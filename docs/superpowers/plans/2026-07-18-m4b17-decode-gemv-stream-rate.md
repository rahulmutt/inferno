# M4b.17 — Decode GEMV Stream-Rate Attribution + Gated Bandwidth Levers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the decode GEMV stream-rate gap (~40–41 GB/s achieved vs 54.39/45.95 GB/s ceilings) into named causes with a shipping-path instrument, then build only the one lever family the pre-registered gate authorizes, and judge it against the split exit criterion (tg ≥ 1.0x vs llama best-of on 16c; measured ceiling statement on 8c).

**Architecture:** A new `gemv_stream` example in `inferno-pool` (beside M4b.10's `bw_curve`) drives the SHIPPING Q8_0 GEMV kernel through the SHIPPING `inferno_par_gemv` dispatch over a decode-shaped working set (the full Qwen2.5-0.5B per-token GEMV sequence, ~525 MiB packed — larger than any target L3, so every pass streams like real decode). Three memory arms (heap / file-backed 4 KiB mmap / THP-backed anonymous) × two kernels (real GEMV / pure-stream touch) give the roofline and page/TLB decompositions; a `perf stat` counter lane and the existing `gate-decode-attr.sh` profiles corroborate. The spec's rules 1–3 then authorize at most one of Lever H (hugepage weight residency in `inferno-core`) or Lever V (AVX-512 VNNI GEMV in `inferno-kernels`).

**Tech Stack:** Rust, the M2 rig, `inferno-pool`'s global pool, rustix `mm`, criterion conventions, quiet-hw gate scripts (`scripts/quiet-hw/`), PhoenixNAP metal via `mise run metal`, Intel SDE (only if Lever V fires).

**Spec:** `docs/superpowers/specs/2026-07-18-m4b17-decode-gemv-stream-rate-design.md` — the pre-registered gate rules, ship-gate thresholds, and exit criteria live THERE. This plan never restates a threshold as authority; when in doubt the spec wins.

## Global Constraints

- **Instrument honesty:** every measuring arm calls the shipping kernel symbols through the shipping `inferno_par_gemv` dispatch — never a bench-local copy (M4b.15 inadmissibility finding).
- **Bit-identity:** Lever H changes no bytes and no kernels. Lever V must be bit-identical to scalar/AVX2 (rig-demanded, never a tolerance). `crates/inferno-graph/src/tolerance.rs` and `gemv_rel_tol` are NOT touched; `git diff main -- crates/inferno-graph/src/tolerance.rs` must be empty at close.
- **`unsafe` only in sanctioned crates:** `inferno-kernels`, `inferno-core`, `inferno-pool` (each has its own `[lints.rust]` table; `unsafe_op_in_unsafe_fn = "deny"` — write `unsafe {}` blocks inside unsafe fns).
- **No lever is built before the Task 6 gate verdict is recorded** in the spec §Amendments with arithmetic shown. Gate verdicts are computed by a human/controller from recorded data.
- **Recorded data points are never edited** (erratum pattern for corrections).
- **Verification commands:** `mise run test`, `mise run lint`, `cargo test -p inferno-kernels --test rig`, `cargo test -p inferno-codegen --test differential`, `cargo test -p inferno-core --test artifact`.
- **Quiet-hw sessions:** never provision two PNAP servers in parallel; after ANY failed session run `mise run metal-gc` and confirm zero servers; retry transient devpod post-create panics once; on 406 check catalog stock and pass `--location`. Commit and push before `mise run metal` (the box clones committed HEAD).
- **Branch:** work on `m4b17-design` (already exists, holds the spec); PR to `main` at the end.
- **Protocol geometry (Qwen2.5-0.5B):** hidden 896, ffn 4864, kv_dim 128, vocab 151936, 24 layers, Q8_0. Best-t = phys cores (16 / 8 on the two boxes).

---

### Task 1: `gemv_stream` — the shipping-path instrument example

**Files:**
- Create: `crates/inferno-pool/examples/gemv_stream.rs`
- Modify: `crates/inferno-pool/Cargo.toml` (dev-dependency)

**Interfaces:**
- Consumes: `inferno_pool::{init_global, inferno_par_gemv, GemvFn}`; `inferno_kernels::{KernelIsa, act::quantize_row_q8a, q8_0::{pack_q8_0_rs8, packed_len_q8_0_rs8}, inferno_gemv_q8_0_rs8_avx2, inferno_gemv_q8_0_rs8_scalar}`; `rustix::mm` (dev-dep).
- Produces: `cargo run --release -p inferno-pool --example gemv_stream -- <lanes> [layers] [reps]` printing per-arm × per-kernel GB/s tables, and a `--spin <arm> <secs>` mode that prints `STREAMING pid=<pid>` then busy-streams (Task 2's perf-attach hook). Task 2's script runs it.

- [ ] **Step 1: Add the dev-dependency**

In `crates/inferno-pool/Cargo.toml` `[dev-dependencies]` (examples build against dev-deps; this adds nothing to the shipping dependency graph):

```toml
# M4b.17 instrument (examples only): the gemv_stream arms need mmap /
# madvise to reproduce the artifact loader's 4 KiB file mapping and the
# THP counterfactual. Same crate+features inferno-core already vendors.
rustix = { workspace = true, features = ["mm", "std"] }
```

- [ ] **Step 2: Write the example**

Create `crates/inferno-pool/examples/gemv_stream.rs`:

```rust
//! M4b.17 arms 1+2: decode-shaped GEMV stream rate, SHIPPING kernel through
//! the SHIPPING dispatch (`inferno_par_gemv` on the global pool) — never a
//! bench-local copy (M4b.15 inadmissibility finding).
//!
//! Working set = the full Qwen2.5-0.5B per-token GEMV sequence (24 layers ×
//! q/k/v/o/gate/up/down + lm_head, ~525 MiB packed Q8_0) so every pass
//! streams from DRAM exactly like real decode — a single small matrix would
//! go cache-hot and measure nothing (the recorded 54.39/45.95 ceilings came
//! from a 143 MiB synthetic image; the achieved ~40 GB/s is in-loop).
//!
//! Three memory arms × two kernels:
//!   heap    — `AlignedBuf` (the bw_curve/ceiling condition)
//!   mmap4k  — file-backed 4 KiB-page PROT_READ/MAP_PRIVATE mmap (the
//!             artifact loader's exact condition, artifact.rs)
//!   thp     — one anonymous region, madvise(MADV_HUGEPAGE), images copied
//!             in (the Lever H counterfactual); AnonHugePages from
//!             /proc/self/smaps is printed so a THP=never box can't fake it
//!   gemv    — the shipping Q8_0 rs8 kernel for the host ISA
//!   stream  — a pure byte-touch loop over the same rs8 group span, through
//!             the same dispatch, same shards (the GEMV-shaped roofline)
//!
//! GATE QUANTITIES ARE HUMAN: paste the tables into the M4b.17 spec
//! §Amendments; rules 1–3 live there. Numbers are only meaningful from
//! quiet hardware.
//!
//! Usage: gemv_stream -- <lanes> [layers=24] [reps=5] [--spin <arm> <secs>]

use inferno_formats::quant::f32_to_f16;
use inferno_kernels::{KernelIsa, act, q8_0};
use inferno_pool::{GemvFn, inferno_par_gemv, init_global};
use std::io::Write as _;
use std::time::{Duration, Instant};

const HIDDEN: usize = 896;
const FFN: usize = 4864;
const KV_DIM: usize = 128;
const VOCAB: usize = 151936;
const N_LAYERS: usize = 24;
/// One rs8 group: 8 lanes × 4 B f32 scale + 8 lanes × 32 B quants.
/// Asserted against `packed_len_q8_0_rs8(8, 32)` in main() so a layout
/// change in inferno-kernels breaks this example loudly, not silently.
const GROUP_BYTES: usize = 288;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Attn,
    Ffn,
    LmHead,
}
const CLASSES: [Class; 3] = [Class::Attn, Class::Ffn, Class::LmHead];

/// (rows, k, class) for one decode token, in dispatch order.
fn token_shapes(layers: usize) -> Vec<(usize, usize, Class)> {
    let mut v = Vec::new();
    for _ in 0..layers {
        v.push((HIDDEN, HIDDEN, Class::Attn)); // q
        v.push((KV_DIM, HIDDEN, Class::Attn)); // k
        v.push((KV_DIM, HIDDEN, Class::Attn)); // v
        v.push((HIDDEN, HIDDEN, Class::Attn)); // o
        v.push((FFN, HIDDEN, Class::Ffn)); // gate
        v.push((FFN, HIDDEN, Class::Ffn)); // up
        v.push((HIDDEN, FFN, Class::Ffn)); // down
    }
    v.push((VOCAB, HIDDEN, Class::LmHead));
    v
}

fn pseudo_bytes(mut seed: u64, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 32) as u8
        })
        .collect()
}

fn pseudo_f32(mut seed: u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// Plausible Q8_0 file bytes without quantizing gigabytes of f32 (perf is
/// value-independent): fixed small scale + random quant payloads. Same
/// trick as inferno-kernels/benches/gemv.rs.
fn gen_q8_0(seed: u64, rows: usize, k: usize) -> Vec<u8> {
    let nb = rows * k / 32;
    let d = f32_to_f16(0.05).to_le_bytes();
    let qs = pseudo_bytes(seed, nb * 32);
    let mut out = Vec::with_capacity(nb * 34);
    for b in 0..nb {
        out.extend_from_slice(&d);
        out.extend_from_slice(&qs[b * 32..(b + 1) * 32]);
    }
    out
}

/// Pure-stream GemvFn: touch exactly the rs8 group span the real GEMV
/// reads for `row_start..row_end`, through the same dispatch and shards.
/// 8-way u64 xor keeps enough ILP to stream at memory rate; the xor lands
/// in `y[row_start]` so the loop can't be dead-code-eliminated.
///
/// # Safety
/// Same contract as the Q8_0 rs8 GEMV ABI (`w` an rs8 image for (rows, k)
/// covering `row_end`, `y` writable at `row_start`).
unsafe extern "C" fn stream_touch(
    y: *mut f32,
    _x: *const u8,
    w: *const u8,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    let nb = k / 32;
    let s0 = row_start / 8;
    let s1 = row_end.div_ceil(8);
    let mut p = unsafe { w.add(s0 * nb * GROUP_BYTES) }.cast::<u64>();
    let end = unsafe { w.add(s1 * nb * GROUP_BYTES) }.cast::<u64>();
    let mut a = [0u64; 8];
    while unsafe { p.add(8) } <= end {
        for (i, ai) in a.iter_mut().enumerate() {
            *ai ^= unsafe { p.add(i).read() };
        }
        p = unsafe { p.add(8) };
    }
    let mut acc = 0u64;
    while p < end {
        acc ^= unsafe { p.read() };
        p = unsafe { p.add(1) };
    }
    for v in a {
        acc ^= v;
    }
    unsafe { y.add(row_start).write(acc as f32) };
}

/// One matrix: its packed image lives in an arm's memory at `ptr`.
struct Mat {
    ptr: *const u8,
    rows: usize,
    k: usize,
    bytes: usize,
    class: Class,
}

/// An arm owns its memory (heap bufs, an mmap, or an anon THP region) and
/// hands out per-matrix pointers.
struct Arm {
    name: &'static str,
    mats: Vec<Mat>,
    // Keeps the backing memory alive; variants are never read directly.
    _backing: Backing,
}

enum Backing {
    // Payload only keeps the allocations alive (never read) — hence the
    // dead_code allow; Drop reads the mmap variants' fields.
    #[allow(dead_code)]
    Heap(Vec<inferno_kernels::AlignedBuf>),
    Mmap { ptr: *mut u8, len: usize },
    Anon { ptr: *mut u8, len: usize },
}

impl Drop for Backing {
    fn drop(&mut self) {
        match *self {
            Backing::Heap(_) => {}
            Backing::Mmap { ptr, len } | Backing::Anon { ptr, len } => {
                // SAFETY: ptr/len are exactly what mmap returned; unmapped once.
                let _ = unsafe { rustix::mm::munmap(ptr.cast(), len) };
            }
        }
    }
}

fn pad4k(n: usize) -> usize {
    n.next_multiple_of(4096)
}

fn build_images(layers: usize) -> Vec<(inferno_kernels::AlignedBuf, usize, usize, Class)> {
    token_shapes(layers)
        .iter()
        .enumerate()
        .map(|(i, &(rows, k, class))| {
            let file = gen_q8_0(0x9e37 + i as u64 * 0x1315, rows, k);
            let img = q8_0::pack_q8_0_rs8(&file, rows, k).expect("pack rs8");
            (img, rows, k, class)
        })
        .collect()
}

fn heap_arm(layers: usize, images: &[(inferno_kernels::AlignedBuf, usize, usize, Class)]) -> Arm {
    // Re-pack into owned bufs (AlignedBuf is not Clone; regeneration is
    // cheap and deterministic).
    let bufs: Vec<inferno_kernels::AlignedBuf> = token_shapes(layers)
        .iter()
        .enumerate()
        .map(|(i, &(rows, k, _))| {
            q8_0::pack_q8_0_rs8(&gen_q8_0(0x9e37 + i as u64 * 0x1315, rows, k), rows, k).unwrap()
        })
        .collect();
    let mats = bufs
        .iter()
        .zip(images)
        .map(|(b, &(_, rows, k, class))| Mat {
            ptr: b.as_ptr(),
            rows,
            k,
            bytes: q8_0::packed_len_q8_0_rs8(rows, k),
            class,
        })
        .collect();
    Arm { name: "heap", mats, _backing: Backing::Heap(bufs) }
}

fn mmap4k_arm(images: &[(inferno_kernels::AlignedBuf, usize, usize, Class)]) -> Arm {
    let total: usize = images.iter().map(|(b, ..)| pad4k(b.as_slice().len())).sum();
    let path = std::env::temp_dir().join(format!("inferno-gemv-stream-{}.bin", std::process::id()));
    let mut f = std::fs::File::create(&path).expect("create temp weights file");
    let mut offsets = Vec::new();
    let mut off = 0usize;
    for (b, ..) in images {
        offsets.push(off);
        f.write_all(b.as_slice()).unwrap();
        let pad = pad4k(b.as_slice().len()) - b.as_slice().len();
        f.write_all(&vec![0u8; pad]).unwrap();
        off += pad4k(b.as_slice().len());
    }
    f.sync_all().unwrap();
    // Mirror artifact.rs exactly: PROT_READ, MAP_PRIVATE, file-backed.
    use std::os::fd::AsFd;
    let f = std::fs::File::open(&path).unwrap();
    // SAFETY: valid fd, real length, read-only private map (artifact.rs's
    // exact call shape); never written through.
    let ptr = unsafe {
        rustix::mm::mmap(
            std::ptr::null_mut(),
            total,
            rustix::mm::ProtFlags::READ,
            rustix::mm::MapFlags::PRIVATE,
            f.as_fd(),
            0,
        )
    }
    .expect("mmap temp weights file")
    .cast::<u8>();
    let _ = std::fs::remove_file(&path); // mapping keeps it alive
    let mats = images
        .iter()
        .zip(&offsets)
        .map(|(&(ref b, rows, k, class), &o)| Mat {
            // SAFETY: o < total by construction.
            ptr: unsafe { ptr.add(o) },
            rows,
            k,
            bytes: b.as_slice().len(),
            class,
        })
        .collect();
    Arm { name: "mmap4k", mats, _backing: Backing::Mmap { ptr, len: total } }
}

fn thp_arm(images: &[(inferno_kernels::AlignedBuf, usize, usize, Class)]) -> Arm {
    let total = pad4k(
        images.iter().map(|(b, ..)| pad4k(b.as_slice().len())).sum::<usize>(),
    )
    .next_multiple_of(2 * 1024 * 1024);
    // SAFETY: fresh anonymous private RW mapping of `total` bytes.
    let ptr = unsafe {
        rustix::mm::mmap_anonymous(
            std::ptr::null_mut(),
            total,
            rustix::mm::ProtFlags::READ | rustix::mm::ProtFlags::WRITE,
            rustix::mm::MapFlags::PRIVATE,
        )
    }
    .expect("mmap anon")
    .cast::<u8>();
    // Advisory: on THP=never this silently does nothing — which is why
    // AnonHugePages is printed below instead of trusted.
    // SAFETY: region just mapped, length exact.
    let _ = unsafe { rustix::mm::madvise(ptr.cast(), total, rustix::mm::Advice::LinuxHugepage) };
    let mut off = 0usize;
    let mut mats = Vec::new();
    for &(ref b, rows, k, class) in images {
        let s = b.as_slice();
        // SAFETY: dst range is inside the fresh mapping; src is a live slice.
        unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), ptr.add(off), s.len()) };
        mats.push(Mat {
            // SAFETY: off < total by construction.
            ptr: unsafe { ptr.add(off) },
            rows,
            k,
            bytes: s.len(),
            class,
        });
        off += pad4k(s.len());
    }
    println!(
        "thp arm: region {total} B, AnonHugePages {} kB (from /proc/self/smaps)",
        anon_huge_kb(ptr as usize)
    );
    Arm { name: "thp", mats, _backing: Backing::Anon { ptr, len: total } }
}

/// AnonHugePages for the smaps region containing `addr` (0 if unreadable —
/// corroboration, not a gate quantity).
fn anon_huge_kb(addr: usize) -> u64 {
    let Ok(smaps) = std::fs::read_to_string("/proc/self/smaps") else {
        return 0;
    };
    let mut in_region = false;
    for line in smaps.lines() {
        if let Some((range, _)) = line.split_once(' ') {
            if let Some((lo, hi)) = range.split_once('-') {
                if let (Ok(lo), Ok(hi)) =
                    (usize::from_str_radix(lo, 16), usize::from_str_radix(hi, 16))
                {
                    in_region = lo <= addr && addr < hi;
                }
            }
        }
        if in_region && line.starts_with("AnonHugePages:") {
            return line
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
        }
    }
    0
}

/// One full token pass; returns per-class accumulated time.
///
/// # Safety
/// Every `Mat` points at a live rs8 image for its (rows, k); `y` holds
/// VOCAB f32s; `xq*` are q8a buffers for k=HIDDEN / k=FFN.
unsafe fn token_pass(
    arm: &Arm,
    kernel: GemvFn,
    y: &mut [f32],
    xq_hidden: &[u8],
    xq_ffn: &[u8],
) -> [Duration; 3] {
    let mut t = [Duration::ZERO; 3];
    for m in &arm.mats {
        let xq = if m.k == HIDDEN { xq_hidden } else { xq_ffn };
        let t0 = Instant::now();
        // SAFETY: forwarding the caller's contract; this is the shipping
        // decode dispatch entry, verbatim.
        unsafe { inferno_par_gemv(kernel, y.as_mut_ptr(), xq.as_ptr(), m.ptr, m.k, m.rows) };
        let ci = CLASSES.iter().position(|&c| c == m.class).unwrap();
        t[ci] += t0.elapsed();
    }
    t
}

fn class_bytes(arm: &Arm) -> [usize; 3] {
    let mut b = [0usize; 3];
    for m in &arm.mats {
        b[CLASSES.iter().position(|&c| c == m.class).unwrap()] += m.bytes;
    }
    b
}

fn main() {
    assert_eq!(
        GROUP_BYTES,
        q8_0::packed_len_q8_0_rs8(8, 32),
        "rs8 group layout changed; fix GROUP_BYTES and stream_touch"
    );
    let args: Vec<String> = std::env::args().skip(1).collect();
    let lanes: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or_else(|| {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    });
    let layers: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(N_LAYERS);
    let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let spin: Option<(String, u64)> = args
        .iter()
        .position(|a| a == "--spin")
        .map(|i| (args[i + 1].clone(), args[i + 2].parse().expect("--spin <arm> <secs>")));

    let isa = if KernelIsa::Avx2.available() { KernelIsa::Avx2 } else { KernelIsa::Scalar };
    let gemv: GemvFn = match isa {
        KernelIsa::Avx2 => inferno_kernels::inferno_gemv_q8_0_rs8_avx2,
        KernelIsa::Scalar => inferno_kernels::inferno_gemv_q8_0_rs8_scalar,
    };

    init_global(lanes).expect("init global pool");
    let images = build_images(layers);
    let total_bytes: usize = images.iter().map(|(b, ..)| b.as_slice().len()).sum();
    println!(
        "gemv_stream: {layers} layers + lm_head, {} matrices, {:.1} MiB packed, {isa:?}, lanes={lanes}, reps={reps}",
        images.len(),
        total_bytes as f64 / (1024.0 * 1024.0)
    );

    let xq_hidden = act::quantize_row_q8a(isa, &pseudo_f32(7, HIDDEN)).unwrap();
    let xq_ffn = act::quantize_row_q8a(isa, &pseudo_f32(11, FFN)).unwrap();
    let mut y = vec![f32::NAN; VOCAB];

    let arms = [heap_arm(layers, &images), mmap4k_arm(&images), thp_arm(&images)];

    if let Some((arm_name, secs)) = spin {
        let arm = arms.iter().find(|a| a.name == arm_name).expect("--spin arm name");
        // Warm one pass so perf attaches to steady-state streaming.
        // SAFETY: buffers built above for exactly these shapes.
        unsafe { token_pass(arm, gemv, &mut y, xq_hidden.as_slice(), xq_ffn.as_slice()) };
        println!("STREAMING pid={}", std::process::id());
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(secs) {
            // SAFETY: as above.
            unsafe { token_pass(arm, gemv, &mut y, xq_hidden.as_slice(), xq_ffn.as_slice()) };
        }
        return;
    }

    println!();
    println!("| arm | kernel | attn GB/s | ffn GB/s | lm_head GB/s | total GB/s | ms/token |");
    println!("|---|---|---|---|---|---|---|");
    for arm in &arms {
        let cb = class_bytes(arm);
        for (kname, kfn) in [("gemv", gemv), ("stream", stream_touch as GemvFn)] {
            // Warm pass (page-faults + first-touch), then reps timed passes;
            // per-class median across reps.
            // SAFETY: buffers built above for exactly these shapes.
            unsafe { token_pass(arm, kfn, &mut y, xq_hidden.as_slice(), xq_ffn.as_slice()) };
            let mut per_class: Vec<[Duration; 3]> = Vec::new();
            for _ in 0..reps {
                // SAFETY: as above.
                per_class.push(unsafe {
                    token_pass(arm, kfn, &mut y, xq_hidden.as_slice(), xq_ffn.as_slice())
                });
            }
            let med = |ci: usize| -> f64 {
                let mut v: Vec<f64> =
                    per_class.iter().map(|t| t[ci].as_secs_f64()).collect();
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            };
            let gbps = |ci: usize| cb[ci] as f64 / med(ci) / 1e9;
            let tot_t: f64 = (0..3).map(med).sum();
            println!(
                "| {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |",
                arm.name,
                kname,
                gbps(0),
                gbps(1),
                gbps(2),
                total_bytes as f64 / tot_t / 1e9,
                tot_t * 1e3,
            );
        }
    }
    println!();
    println!("gate input (human, to the M4b.17 spec §Amendments): roofline = stream");
    println!("rows; page/TLB recovery = thp vs mmap4k on the gemv rows; the heap row");
    println!("is the recorded-ceiling condition (bw_curve used heap buffers).");
}
```

- [ ] **Step 3: Build and smoke it (dev box)**

Run: `cargo run --release -q -p inferno-pool --example gemv_stream -- 2 2 2`
Expected: prints the header (2 layers, 15 matrices), the thp arm's AnonHugePages line, and a 6-row table (3 arms × 2 kernels) with plausible GB/s values (tens); exits 0. Fix compile errors against the pinned toolchain — intrinsic-free, so drift risk is only in rustix call shapes (`mmap_anonymous`, `Advice::LinuxHugepage` naming).

Note: if `AlignedBuf`/`as_ptr` visibility differs from this sketch (e.g. `as_slice().as_ptr()` needed), adjust at the call sites — the structure is the contract, `inferno-kernels`' public surface is the authority.

- [ ] **Step 4: Spin-mode smoke**

Run: `timeout 30 cargo run --release -q -p inferno-pool --example gemv_stream -- 2 2 1 --spin mmap4k 5; echo "exit=$?"`
Expected: `STREAMING pid=<n>` appears, process exits by itself within ~25 s, `exit=0`.

- [ ] **Step 5: Lint + commit**

```bash
mise run lint
git add crates/inferno-pool/Cargo.toml crates/inferno-pool/examples/gemv_stream.rs
git commit -m "M4b.17 Task 1: gemv_stream shipping-path instrument (3 arms x gemv/stream)"
```

---

### Task 2: `gate-gemv-stream.sh` — the quiet-hw session script + counter lane

**Files:**
- Create: `scripts/quiet-hw/gate-gemv-stream.sh` (mode 755)

**Interfaces:**
- Consumes: `scripts/quiet-hw/lib.sh` (`smoke_header`, `machine_block`, `numa_require`, `numa_wrap`, `phys_cores`), Task 1's example.
- Produces: the Round 1 session artifact (arm tables at best-t and t=1, perf-stat tables for mmap4k vs thp, THP-backing line) recorded verbatim into the spec §Amendments. Tasks 3–5 run it.

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# M4b.17 arms 1+2+3 — decode-shaped GEMV stream-rate arms (roofline,
# page/TLB, counter lane) on quiet hardware, via the gemv_stream example
# (shipping kernel through the shipping dispatch). Arm 4 (idle-gap) comes
# from gate-decode-attr.sh's profiles in the same session, not from here.
# VERDICTS ARE HUMAN: paste the tables into the M4b.17 spec §Amendments and
# apply gate rules 1–3 there. Counters are corroboration only (spec).
# Usage: gate-gemv-stream.sh   (env: QHW_OUT QHW_SMOKE QHW_NUMA_NODE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

OUT="${QHW_OUT:-$(mktemp -d)}"
PHYS=$(phys_cores)
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  LANES=2 LAYERS=2 REPS=2
else
  LANES="$PHYS" LAYERS=24 REPS=5
fi

smoke_header "gate-gemv-stream (M4b.17: roofline + page/TLB + counters)"
machine_block
numa_require
[ -n "${QHW_NUMA_NODE:-}" ] && echo "numa: pinned to node ${QHW_NUMA_NODE} (cpubind+membind); phys_cores=$PHYS"
echo

cargo build --release -q -p inferno-pool --example gemv_stream

echo "--- arms at best-t (lanes=$LANES) ---"
$(numa_wrap) cargo run --release -q -p inferno-pool --example gemv_stream -- "$LANES" "$LAYERS" "$REPS" \
  | tee "$OUT/gemv-stream-t$LANES.txt"
echo
echo "--- arms at t=1 (per-thread quality context, not a gate quantity) ---"
$(numa_wrap) cargo run --release -q -p inferno-pool --example gemv_stream -- 1 "$LAYERS" "$REPS" \
  | tee "$OUT/gemv-stream-t1.txt"
echo

if [ "${QHW_SMOKE:-0}" = 1 ]; then
  echo "SMOKE: counter lane skipped"
  exit 0
fi

# --- counter lane (corroboration only; spec §Task 1 arm 3) ---
if ! command -v perf >/dev/null; then
  echo "DEVIATION: perf unavailable — counter lane skipped (record in amendment)"
  exit 0
fi
EVENTS="cycles,instructions,dTLB-load-misses,LLC-load-misses"
for ARM in mmap4k thp; do
  LOG="$OUT/spin-$ARM.log"
  : > "$LOG"
  $(numa_wrap) cargo run --release -q -p inferno-pool --example gemv_stream -- \
    "$LANES" "$LAYERS" 1 --spin "$ARM" 25 > "$LOG" 2>&1 &
  BG=$!
  for _ in $(seq 240); do grep -q "STREAMING" "$LOG" && break; sleep 0.5; done
  grep -q "STREAMING" "$LOG" || { echo "spin never reached STREAMING ($ARM)"; cat "$LOG"; exit 1; }
  PID=$(sed -n 's/.*STREAMING pid=\([0-9]*\).*/\1/p' "$LOG" | head -1)
  echo "--- perf ($ARM, 5 s attach at lanes=$LANES) ---"
  perf stat -e "$EVENTS" -p "$PID" -- sleep 5 2>&1 | tee "$OUT/perf-$ARM.txt"
  wait "$BG" || true
  grep "AnonHugePages" "$LOG" || true
  echo
done
echo "gate arithmetic destination: M4b.17 spec §Amendments (rules 1-3)"
```

- [ ] **Step 2: Smoke it**

Run: `chmod +x scripts/quiet-hw/gate-gemv-stream.sh && QHW_SMOKE=1 bash scripts/quiet-hw/gate-gemv-stream.sh`
Expected: machine block, both arm tables (tiny 2-layer numbers), `SMOKE: counter lane skipped`, exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/quiet-hw/gate-gemv-stream.sh
git commit -m "M4b.17 Task 2: gate-gemv-stream.sh session script + perf counter lane"
```

---

### Task 3: Dev-box full run — plumbing sanity (honestly non-quiet)

No spec-recorded numbers come from this task; it exists so the metal round never debugs the instrument.

**Files:** none (run only).

- [ ] **Step 1: Full non-smoke run**

Run: `bash scripts/quiet-hw/gate-gemv-stream.sh`
Expected: full 24-layer tables at best-t and t=1; perf tables (or the recorded DEVIATION line if the dev box lacks perf); the thp arm reports a nonzero AnonHugePages figure if the dev box has THP enabled (`cat /sys/kernel/mm/transparent_hugepage/enabled` to interpret — `[never]` explains a 0).

- [ ] **Step 2: Sanity checks (fix the instrument if any fail)**

- `stream` ≥ `gemv` GB/s on every arm (a stream row slower than real GEMV means the touch loop is broken).
- The three arms' `stream` rows are within noise of each other OR the difference is page-shaped (thp ≥ mmap4k) — an inverted ordering means warmup is wrong.
- ms/token for the mmap4k gemv row at best-t is in the ballpark of real decode's per-token matmul time (~10–15 ms on the dev box; it is the same kernel over the same byte volume).

- [ ] **Step 3: Push (the metal box clones committed HEAD)**

```bash
git push -u origin m4b17-design
```

---

### Task 4: Round 1 quiet-hw session A — 16c `d2.c1.medium` (6336Y)

**Files:**
- Modify: `docs/superpowers/specs/2026-07-18-m4b17-decode-gemv-stream-rate-design.md` (§Amendments)

- [ ] **Step 1: Provision + run the workload** (one session, one box; never parallel):

```bash
mise run metal -- d2.c1.medium --yes -- '
  set -euo pipefail
  command -v perf >/dev/null || sudo apt-get install -y linux-perf || true
  MODEL=$(bash scripts/fetch-qwen-gguf.sh)
  export QHW_OUT=target/quiet-hw
  bash scripts/quiet-hw/preflight.sh
  bash scripts/quiet-hw/gate-gemv-stream.sh
  bash scripts/quiet-hw/gate-decode-attr.sh "$MODEL"
'
```

(`gate-gemv-stream.sh` = arms 1–3; `gate-decode-attr.sh`'s t=1 + best-t profiles = arm 4's bracket data and the achieved in-loop per-class GB/s.)

- [ ] **Step 2: On ANY failure:** `mise run metal-gc` and confirm zero servers before retrying. Transient devpod post-create panic: retry once. 406: check catalog stock, pass `--location`.

- [ ] **Step 3: Record Session A** in the M4b.17 spec §Amendments (dated `Round 1 Session A — d2.c1.medium`): machine block, both arm tables verbatim, perf tables (or deviation), AnonHugePages line, the decode-attr profile tables, then the human-computed quantities the spec's gate consumes: achieved per-class GB/s (profile), roofline (stream rows), the per-shape-class gaps, THP recovery vs `G/2`, idle-gap (per-token wall − matmul bracket sum), and the dTLB corroboration.

- [ ] **Step 4: Commit + push**

```bash
git add docs/superpowers/specs/2026-07-18-m4b17-decode-gemv-stream-rate-design.md
git commit -m "specs: M4b.17 Round 1 session A (16c) — arm tables, counters, profiles"
git push
```

---

### Task 5: Round 1 quiet-hw session B — 8c `s2.c2.medium` (E-2388G)

Same as Task 4 with `s2.c2.medium`, recorded as `Round 1 Session B — s2.c2.medium`.

**Files:**
- Modify: `docs/superpowers/specs/2026-07-18-m4b17-decode-gemv-stream-rate-design.md` (§Amendments)

- [ ] **Step 1: Provision + run** (Task 4 Step 1 workload, `s2.c2.medium`; remember metal quirks)
- [ ] **Step 2: Record Session B** (Task 4 Step 3 content, per-box numbers) **plus the 8c ceiling statement the spec's exit criterion requires**: from the measured GEMV-shaped roofline and the GEMV share of the decode wall, state with arithmetic whether ANY streaming lever can reach tg 1.0x on this box.
- [ ] **Step 3: Commit + push** (message: `specs: M4b.17 Round 1 session B (8c) — arm tables, counters, profiles, ceiling statement`)

---

### Task 6: Gate verdict amendment — rules 1–3 (human, arithmetic shown once)

**Files:**
- Modify: `docs/superpowers/specs/2026-07-18-m4b17-decode-gemv-stream-rate-design.md` (§Amendments)

- [ ] **Step 1: Apply the spec's pre-registered gate** to the recorded Session A numbers (16c is the deciding box): compute `G`, the THP recovery, and the kernel-vs-roofline margin; walk rules 1→2→3 in order; record the verdict with the arithmetic shown once. Exactly one of: Lever H authorized / Lever V authorized / STOP.
- [ ] **Step 2: Route the plan:**
  - **Rule 1 (Lever H):** Tasks 7 → 10 → 11. Tasks 8–9 are SKIPPED (record as such).
  - **Rule 2 (Lever V):** Tasks 8 → 9 → 10 → 11. Task 7 is SKIPPED (record as such).
  - **Rule 3 (STOP):** Tasks 7–10 are SKIPPED; Round 1 is the closing data; go directly to Task 11.
- [ ] **Step 3: Commit + push** (message: `specs: M4b.17 gate verdict — rule <n>, <lever|STOP>`)

---

### Task 7 (GATED — only if Task 6 fired rule 1): Lever H — hugepage weight residency

**Files:**
- Modify: `crates/inferno-core/src/artifact.rs`

**Interfaces:**
- Consumes: the existing `Mmap` struct and its `open`/`as_slice`; `rustix::mm` (already a dependency with `mm`+`std`).
- Produces: `WeightsMem` (private) with `load(path) -> Result<WeightsMem>` and `as_slice(&self) -> &[u8]`; env knob `INFERNO_HUGEPAGE_WEIGHTS=1` (opt-in until the Task 11 ship verdict; a ship verdict flips the default in a follow-up one-line change recorded there). No CLI plumbing: the env var is read at artifact load, it is not a codegen input and never touches `cache_key`.

- [ ] **Step 1: Add `HugeCopy` and `WeightsMem` beside `Mmap`** in `artifact.rs`:

```rust
/// M4b.17 Lever H: a THP-backed anonymous copy of `weights.bin`.
///
/// Same bytes, same kernels, same dispatch — residency is not a numeric
/// change (bit-neutral by construction). Costs: a one-time load copy and
/// the weights become anonymous RSS instead of evictable page cache.
/// `madvise(MADV_HUGEPAGE)` is advisory: on THP=never kernels the copy
/// still works, just 4 KiB-backed — behavior degrades to the mmap path's
/// page size, never to an error.
struct HugeCopy {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: written once during construction, read-only afterwards; owns its
// region until drop, like `Mmap`.
unsafe impl Send for HugeCopy {}
unsafe impl Sync for HugeCopy {}

impl HugeCopy {
    fn from_slice(src: &[u8]) -> Result<HugeCopy> {
        let len = src.len();
        if len == 0 {
            return Ok(HugeCopy { ptr: NonNull::dangling(), len: 0 });
        }
        // SAFETY: fresh anonymous private RW mapping; length is exact.
        let addr = unsafe {
            rustix::mm::mmap_anonymous(
                std::ptr::null_mut(),
                len,
                rustix::mm::ProtFlags::READ | rustix::mm::ProtFlags::WRITE,
                rustix::mm::MapFlags::PRIVATE,
            )
        }
        .map_err(|e| CoreError::Mmap(std::io::Error::from_raw_os_error(e.raw_os_error())))?;
        // SAFETY: region just mapped, length exact. Advisory — errors ignored.
        let _ = unsafe { rustix::mm::madvise(addr, len, rustix::mm::Advice::LinuxHugepage) };
        // SAFETY: dst is the fresh mapping (page-aligned ≥ 4096, satisfying
        // the kernels' 32-byte base alignment like `Mmap`); src is live.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), addr.cast::<u8>(), len) };
        let ptr = NonNull::new(addr.cast::<u8>())
            .ok_or_else(|| CoreError::Mmap(std::io::Error::other("mmap returned null")))?;
        Ok(HugeCopy { ptr, len })
    }

    fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: single live mapping owned by self; never written after
        // construction; borrow tied to &self.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for HugeCopy {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        // SAFETY: exact base/len from mmap_anonymous; unique final unmap.
        let _ = unsafe { rustix::mm::munmap(self.ptr.as_ptr().cast(), self.len) };
    }
}

/// The weights image backing: the standing file mmap, or (Lever H, opt-in
/// via `INFERNO_HUGEPAGE_WEIGHTS=1`) a THP-backed anonymous copy.
enum WeightsMem {
    Mapped(Mmap),
    Huge(HugeCopy),
}

impl WeightsMem {
    fn load(path: &Path) -> Result<WeightsMem> {
        let m = Mmap::open(path)?;
        if std::env::var("INFERNO_HUGEPAGE_WEIGHTS").is_ok_and(|v| v == "1") {
            // `m` drops at the end of this call: the file pages are released
            // back to the page cache once the copy exists.
            return Ok(WeightsMem::Huge(HugeCopy::from_slice(m.as_slice())?));
        }
        Ok(WeightsMem::Mapped(m))
    }

    fn as_slice(&self) -> &[u8] {
        match self {
            WeightsMem::Mapped(m) => m.as_slice(),
            WeightsMem::Huge(h) => h.as_slice(),
        }
    }
}
```

- [ ] **Step 2: Swap the field and the load site**

- `struct Artifact` field: `weights: Mmap` → `weights: WeightsMem` (the doc comment on the struct's drop-order note stays valid — keep it).
- Load site (`let weights = Mmap::open(&dir.join("weights.bin"))?;`) → `let weights = WeightsMem::load(&dir.join("weights.bin"))?;`
- The two consumers (`self.weights.as_slice().as_ptr()` in `prefill`/`decode_step`) compile unchanged.

- [ ] **Step 3: Unit test (bit-neutrality is byte-equality of the mapping)**

In `artifact.rs`'s existing `#[cfg(test)]` module (create one beside the other unit tests if none exists in this file — check first):

```rust
#[test]
fn huge_copy_is_byte_identical_and_aligned() {
    let src: Vec<u8> = (0..123_457u32).map(|i| (i * 2654435761) as u8).collect();
    let h = HugeCopy::from_slice(&src).unwrap();
    assert_eq!(h.as_slice(), &src[..]);
    assert_eq!(h.as_slice().as_ptr() as usize % 4096, 0);
    assert!(HugeCopy::from_slice(&[]).unwrap().as_slice().is_empty());
}
```

(No env-var test: env mutation races parallel test threads. The env path is exercised end-to-end in Step 5 and in the Round 2 A/B.)

- [ ] **Step 4: Run the gates**

```bash
cargo test -p inferno-core --test artifact
cargo test -p inferno-core huge_copy
cargo test -p inferno-codegen --test differential
mise run test && mise run lint
```
Expected: all green, zero snapshot changes.

- [ ] **Step 5: End-to-end same-logits check (dev box)**

```bash
MODEL=$(bash scripts/fetch-qwen-gguf.sh)
cargo run --release -q -- run "$MODEL" --prompt "The quick brown fox" --max-tokens 8 --threads 2 > /tmp/base.txt
INFERNO_HUGEPAGE_WEIGHTS=1 cargo run --release -q -- run "$MODEL" --prompt "The quick brown fox" --max-tokens 8 --threads 2 > /tmp/huge.txt
diff /tmp/base.txt /tmp/huge.txt && echo SAME
```
Expected: `SAME` (greedy decode, same bytes → identical tokens). If the CLI entry point differs from `cargo run --release -q -- run` (check `cli/`), use the project's actual invocation — the A/B-with-env shape is the contract.

- [ ] **Step 6: Commit**

```bash
git add crates/inferno-core/src/artifact.rs
git commit -m "core: INFERNO_HUGEPAGE_WEIGHTS opt-in THP weight residency (M4b.17 Lever H)"
```

---

### Task 8 (GATED — only if Task 6 fired rule 2): Intel SDE test harness

The dev box is Zen 2; a VNNI kernel cannot execute locally. Land the emulation harness **before** the kernel so Task 9 is test-first-able. This is the M4b.13 plan's Task 6, which was gated out and never executed — build it now exactly as specified there, with one addition below. Follow `docs/superpowers/plans/2026-07-17-m4b13-prefill-gemm-register-tiles-vnni.md` Task 6 Steps 1–6 verbatim (SDE fetchurl derivation in `devenv.nix`/`devenv.yaml`, `scripts/test-vnni.sh`, `mise run test-vnni`, the `vnni-sde` nightly CI lane, commit).

**Files:** (from the M4b.13 plan Task 6)
- Modify: `devenv.nix`, `devenv.yaml`, `mise.toml`, `.github/workflows/nightly.yml`
- Create: `scripts/test-vnni.sh`

**Interfaces:**
- Produces: `sde64` on the devenv PATH; `mise run test-vnni` (Task 9's test runner); the nightly CI lane.

- [ ] **Step 1: Execute the M4b.13 plan's Task 6 Steps 1–6** (pin tarball, derivation, script, pre-kernel PASS run, nightly lane, commit). Its commit message applies with `M4b.13` → `M4b.17`.
- [ ] **Step 2: One deviation to carry:** in the nightly job comment and `mise.toml` task description, attribute the harness to M4b.17 (Lever V), since M4b.13's Lever 2 never shipped.

---

### Task 9 (GATED — only if Task 6 fired rule 2): `KernelIsa::Avx512Vnni` + the VNNI **GEMV** kernel

The 512-bit `vpdpbusd` **GEMV** path (spec: GEMV only — the M4b.13 VNNI *GEMM* stays unbuilt; prefill is out of scope). New host symbol → `HOST_ABI_VERSION` "8" → "9".

**Exactness argument (goes in the kernel doc comment):** weights are clamped to −127 by `pack_q8_0_rs8` and activations to [−127, 127] by the q8a quantizers, so `|w|` fits u8 and mask-negating x never sees −128. `vpdpbusd` sums four u8×i8 products (≤ 4·127·127) into an i32 lane from zero — integer-exact, and integer addition is associative, so any regrouping of the block dot is exact. AVX-512 has no `vpsignb`; `_mm512_mask_sub_epi8` under the w<0 byte mask differs from `_mm256_sign_epi8` only at w == 0 bytes (x not zeroed), where `|w| = 0` zeroes the product anyway. The per-row f32 combine keeps the scalar kernel's block order → bit-identity with scalar/AVX2, demanded by the rig, never a tolerance.

**Files:**
- Modify: `crates/inferno-kernels/src/lib.rs` (enum variant)
- Modify: `crates/inferno-kernels/src/q8_0.rs` (the kernel)
- Modify: `crates/inferno-kernels/src/registry.rs` (kernel set + `kernels_for`)
- Modify: `crates/inferno-kernels/tests/rig.rs` (helper match arms)
- Modify: `crates/inferno-codegen/src/loopir.rs` (symbol selection)
- Modify: `crates/inferno-codegen/src/lib.rs` (`HOST_ABI_VERSION`)
- Modify: `crates/inferno-core/src/artifact.rs` (`ensure_kernels_linked`)

**Interfaces:**
- Consumes: the rs8 layout consts (`STRIP`, `GROUP_BYTES`, `WBLOCK`, `Q8A_BLOCK_BYTES`), `hsum8_i32` (M4b.13 Lever 1's transpose-reduce helper in `q8_0.rs`), `inferno_gemv_q8_0_rs8_avx2` (head/tail delegation), `mise run test-vnni` (Task 8).
- Produces: `pub unsafe extern "C" fn inferno_gemv_q8_0_rs8_avx512vnni(y: *mut f32, x: *const u8, w: *const u8, k: usize, row_start: usize, row_end: usize)` (the M2 GEMV ABI); `KernelIsa::Avx512Vnni`; codegen emits the VNNI symbol for (Q8_0, Avx512Vnni) gemv sites and AVX2 symbols everywhere else.

- [ ] **Step 1: Add the enum variant** in `lib.rs` (compile errors become the to-do list):

```rust
pub enum KernelIsa {
    Scalar,
    Avx2,
    /// AVX-512 F/BW/VL + VNNI decode-GEMV path (M4b.17 Lever V). Only the
    /// Q8_0 GEMV has a VNNI kernel; GEMM, quantize, and attention resolve
    /// to the AVX2 kernels under this variant (spec: decode GEMV only).
    Avx512Vnni,
}
```

`available()` arm (tails delegate to the AVX2 kernel, so AVX2+FMA required too):

```rust
            KernelIsa::Avx512Vnni => {
                std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
                    && std::arch::is_x86_feature_detected!("avx512vl")
                    && std::arch::is_x86_feature_detected!("avx512vnni")
                    && std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
            }
```

`all_available()`: append `KernelIsa::Avx512Vnni` to the candidate array. Then `cargo check --workspace` and fix every non-exhaustive match per Steps 3–5.

- [ ] **Step 2: The kernel in `q8_0.rs`**

```rust
/// # Safety
/// As [`inferno_gemv_q8_0_rs8_scalar`]; additionally requires AVX-512
/// F/BW/VL + VNNI (and AVX2+FMA for the delegated head/tail rows).
///
/// Exactness: pack clamps weights to −127 and q8a clamps activations to
/// [−127, 127], so |w| fits u8 and mask-negating x never sees −128.
/// vpdpbusd sums four u8×i8 products into an i32 lane from zero — integer-
/// exact, and integer addition is associative, so regrouping the block dot
/// is exact. AVX-512 lacks vpsignb; `_mm512_mask_sub_epi8` under the w<0
/// byte mask differs from `_mm256_sign_epi8` only at w == 0 bytes (x not
/// zeroed), where |w| = 0 zeroes the product anyway. The per-row f32
/// combine keeps the scalar kernel's block order → bit-identical.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemv_q8_0_rs8_avx512vnni(
    y: *mut f32,
    x: *const u8,
    w: *const u8,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::x86_64::*;
    let nb = k / WBLOCK;
    // Head rows before the first strip boundary / tail rows after the last:
    // delegate to the AVX2 kernel (bit-identical; vanishing share).
    let first_full = row_start.next_multiple_of(STRIP);
    if first_full >= row_end {
        unsafe { inferno_gemv_q8_0_rs8_avx2(y, x, w, k, row_start, row_end) };
        return;
    }
    if row_start < first_full {
        unsafe { inferno_gemv_q8_0_rs8_avx2(y, x, w, k, row_start, first_full) };
    }
    let full_end = first_full + (row_end - first_full) / STRIP * STRIP;
    if full_end < row_end {
        unsafe { inferno_gemv_q8_0_rs8_avx2(y, x, w, k, full_end, row_end) };
    }
    // VNNI fast path: full strips only.
    let mut r = first_full;
    while r < full_end {
        let strip = r / STRIP;
        let mut acc = _mm256_setzero_ps();
        for b in 0..nb {
            let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
            let qs = unsafe { g.add(32) };
            // SAFETY: group base is 32-aligned (rs8 image contract).
            let dw = unsafe { _mm256_load_ps(g.cast()) };
            let xb = unsafe { x.add(b * Q8A_BLOCK_BYTES) };
            let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
            let x256 = unsafe { _mm256_loadu_si256(xb.add(4).cast()) };
            // Both 256-bit halves = this block's activation quants.
            let xv = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(x256), x256);
            // 8 lanes × 32 B quants = four 64 B zmm loads; zmm j holds
            // lanes 2j and 2j+1. Groups are 32-aligned only → loadu.
            let mut p = [_mm256_setzero_si256(); STRIP];
            for j in 0..4 {
                let wz = unsafe { _mm512_loadu_si512(qs.add(j * 64).cast()) };
                let aw = _mm512_abs_epi8(wz);
                let neg = _mm512_movepi8_mask(wz);
                let sx = _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv);
                let d = _mm512_dpbusd_epi32(_mm512_setzero_si512(), aw, sx);
                // Low ymm = lane 2j's 8 partials, high = lane 2j+1's.
                p[2 * j] = _mm512_castsi512_si256(d);
                p[2 * j + 1] = _mm512_extracti64x4_epi64::<1>(d);
            }
            let isum = _mm256_cvtepi32_ps(hsum8_i32(p));
            let dwdx = _mm256_mul_ps(dw, _mm256_set1_ps(dx));
            acc = _mm256_fmadd_ps(dwdx, isum, acc);
        }
        unsafe { _mm256_storeu_ps(y.add(r), acc) };
        r += STRIP;
    }
}
```

Intrinsic signatures drift between Rust versions (`_mm512_loadu_si512` pointer type, `_mm512_movepi8_mask` return type, `hsum8_i32`'s exact parameter shape from M4b.13 Lever 1) — adjust casts/marshalling to what the pinned toolchain and the existing helper expect; the structure above is the contract. If `hsum8_i32` takes something other than `[__m256i; 8]`, marshal `p` accordingly at the call site rather than duplicating the helper.

- [ ] **Step 3: Registry wiring** in `registry.rs::set()`: the `DType::Q8_0` arm's `gemv` gets `KernelIsa::Avx512Vnni => q8_0::inferno_gemv_q8_0_rs8_avx512vnni`. Every other `match isa` in `set()` (q8_0 gemm/quantize; the whole f32 and q4_k arms) folds the new variant into the AVX2 arm: `KernelIsa::Avx2 | KernelIsa::Avx512Vnni => ...` with the comment `// no VNNI kernel — AVX2 fns (spec: Q8_0 decode GEMV only)`. In `kernels_for()`:

```rust
    let kisa = match isa {
        Isa::X86_64v3 => KernelIsa::Avx2,
        // v4: the VNNI GEMV set when the running CPU has it (M4b.17);
        // otherwise the AVX2 set, as since M2.
        Isa::X86_64v4 => {
            if KernelIsa::Avx512Vnni.available() {
                KernelIsa::Avx512Vnni
            } else {
                KernelIsa::Avx2
            }
        }
    };
```

- [ ] **Step 4: Codegen symbol selection + ABI bump** in `loopir.rs`:

```rust
pub fn gemv_symbol(dtype: &DType, isa: inferno_kernels::KernelIsa) -> String {
    // Only (Q8_0, Avx512Vnni) has a VNNI gemv (M4b.17 Lever V).
    if matches!(dtype, DType::Q8_0) && matches!(isa, inferno_kernels::KernelIsa::Avx512Vnni) {
        return "inferno_gemv_q8_0_rs8_avx512vnni".to_string();
    }
    let i = match isa {
        inferno_kernels::KernelIsa::Scalar => "scalar",
        inferno_kernels::KernelIsa::Avx2 | inferno_kernels::KernelIsa::Avx512Vnni => "avx2",
    };
    // ... existing format
}
```

**CRITICAL:** `gemm_symbol` derives from `gemv_symbol` by string replace (`_gemv_` → `_gemm_`). There is NO VNNI GEMM kernel — `gemm_symbol` must force the AVX2 mapping for `Avx512Vnni` BEFORE the derive, or compiled prefill will reference a nonexistent symbol:

```rust
pub fn gemm_symbol(dtype: &DType, isa: inferno_kernels::KernelIsa) -> String {
    // No VNNI gemm exists (M4b.13 Lever 2 never shipped; M4b.17 is GEMV
    // only) — the v4 variant's gemm derives from the AVX2 gemv symbol.
    let isa = match isa {
        inferno_kernels::KernelIsa::Avx512Vnni => inferno_kernels::KernelIsa::Avx2,
        other => other,
    };
    gemv_symbol(dtype, isa).replace("_gemv_", "_gemm_")
}
```

`attention_symbol` and every other symbol chooser: fold `Avx512Vnni` into the `"avx2"` arm. `crates/inferno-codegen/src/lib.rs`: `HOST_ABI_VERSION` → `"9"`, prepending to its doc comment: `/// "9" = M4b.17's VNNI decode-GEMV symbol (inferno_gemv_q8_0_rs8_avx512vnni);`. `crates/inferno-core/src/artifact.rs::ensure_kernels_linked()`: add `p(inferno_kernels::inferno_gemv_q8_0_rs8_avx512vnni as *const ());` beside the other gemv symbols.

- [ ] **Step 5: Rig helper arms** in `rig.rs`: `gemv_q8_0` gets `KernelIsa::Avx512Vnni => inferno_kernels::inferno_gemv_q8_0_rs8_avx512vnni(...)`; every other `match isa` helper folds into the Avx2 arm (`KernelIsa::Avx2 | KernelIsa::Avx512Vnni =>`) — exactly the registry's dispatch. The rig's existing bit-identity properties (scalar vs SIMD, gemm(m=1) ≡ gemv, hspan tiling) now cover the VNNI variant with no new test code.

- [ ] **Step 6: Test — native (variant skipped) and under SDE (variant exercised)**

```bash
mise run test          # dev box: Avx512Vnni unavailable → rig skips it; snapshots unchanged
mise run test-vnni     # SDE Ice Lake: all rig properties include Avx512Vnni
cargo test -p inferno-codegen --test differential
cargo test -p inferno-core --test artifact
```
Expected: all PASS. If an SDE bit-identity property fails, the kernel is wrong — fix the kernel; never the test or a tolerance.

- [ ] **Step 7: Lint + commit**

```bash
mise run lint
git add crates/inferno-kernels crates/inferno-codegen/src/loopir.rs crates/inferno-codegen/src/lib.rs crates/inferno-core/src/artifact.rs
git commit -m "kernels: AVX-512 VNNI Q8_0 GEMV (KernelIsa::Avx512Vnni, vpdpbusd, ABI v9) — M4b.17 Lever V"
```

---

### Task 10 (GATED — only if a lever was built): Round 2 closing sessions — A/B + fresh llama best-of, both boxes

**Files:**
- Modify: `docs/superpowers/specs/2026-07-18-m4b17-decode-gemv-stream-rate-design.md` (§Amendments)
- Modify: `docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md` (§Amendments, protocol data points verbatim)

**Session inputs fixed BEFORE provisioning:** `BASE_SHA` = the main-merge-base commit the lever branched from (baseline binary); `LEVER_SHA` = the pushed lever HEAD. Both reachable from origin. Lever H's lever run additionally exports `INFERNO_HUGEPAGE_WEIGHTS=1` (it is opt-in until the ship verdict); Lever V needs no env (the v4 registry path activates on VNNI silicon).

- [ ] **Step 1: Session A (16c).** If Lever V shipped, run `mise run test` on the box FIRST (native VNNI rig on real AVX-512 silicon before any recorded number):

```bash
mise run metal -- d2.c1.medium --yes -- '
  set -euo pipefail
  MODEL=$(bash scripts/fetch-qwen-gguf.sh)
  export QHW_OUT=target/quiet-hw
  bash scripts/quiet-hw/preflight.sh
  echo "=== BASELINE (BASE_SHA) ==="
  git checkout <BASE_SHA>
  bash scripts/quiet-hw/gate-bench-protocol.sh "$MODEL"
  echo "=== LEVER (LEVER_SHA) ==="
  git checkout <LEVER_SHA>
  mise run test
  <LEVER_ENV> bash scripts/quiet-hw/gate-bench-protocol.sh "$MODEL"
'
```
(Substitute the SHAs literally; `<LEVER_ENV>` = `INFERNO_HUGEPAGE_WEIGHTS=1` for Lever H, empty for Lever V. `gate-bench-protocol.sh` includes the fresh llama best-of baseline — the ship gate's comparison basis.)

- [ ] **Step 2: Session B (8c).** Same workload on `s2.c2.medium` (serial; metal quirks apply).
- [ ] **Step 3: Record both sessions**: protocol tables verbatim in the M4a spec §Amendments (standing rule, binaries identified by SHA); session narratives + per-box tg ratios in the M4b.17 spec §Amendments.
- [ ] **Step 4: Commit + push** (message: `specs: M4b.17 Round 2 sessions — closing A/B vs fresh llama best-of`)

---

### Task 11: Ship-gate verdict, closing verdict, AGENTS.md, PR (runs on every path)

**Files:**
- Modify: `docs/superpowers/specs/2026-07-18-m4b17-decode-gemv-stream-rate-design.md` (§Amendments)
- Modify: `AGENTS.md` (decode paragraph)

- [ ] **Step 1: Ship-gate verdict** (skip on the rule-3 path): apply the spec's ship gate to the Round 2 numbers, arithmetic shown once. On **ship**: flip the lever default (Lever H: `INFERNO_HUGEPAGE_WEIGHTS` semantics become on-unless-`=0`, one-line change in `WeightsMem::load` with the amendment recording it; Lever V: already default-on via the v4 registry path — record that no flip is needed). On **STOP**: the lever stays opt-in, recorded as a finding (M4b.16 precedent).
- [ ] **Step 2: Closing verdict — exit-criteria walk** in the spec §Amendments, item by item against the spec's §Structure item 4: 16c exit criterion (tg ≥ 1.0x met or the recorded reason not), the 8c ceiling statement (from Task 5), every gate verdict recorded once, standing invariants held (run and cite: `mise run test`, `mise run lint`, rig, differential, artifact; `git diff main -- crates/inferno-graph/src/tolerance.rs` empty), every STOP recorded as a finding, v1 context ratios (never the gate).
- [ ] **Step 3: AGENTS.md**: update the decode paragraph with the M4b.17 outcome (one of: hugepage residency semantics + env knob; the VNNI GEMV variant + its bit-identity guard; or the recorded stream-rate ceiling finding and that GEMV levers are closed). Keep it to the non-obvious-constraint register the file uses.
- [ ] **Step 4: Final gates + PR**

```bash
mise run test && mise run lint
git push
gh pr create --title "M4b.17: decode GEMV stream-rate attribution + gated bandwidth levers" \
  --body "Spec: docs/superpowers/specs/2026-07-18-m4b17-decode-gemv-stream-rate-design.md. Instrument (gemv_stream arms + gate script), Round 1 attribution amendments, gate verdict, <lever-or-STOP summary>, closing verdict."
```
Expected: CI green (nix-cache "path is not valid" failures are runner cache corruption — rerun once before debugging).

---

## Self-Review Notes

- **Spec coverage:** instrument arms 1–4 → Tasks 1–2 (arm 4 rides `gate-decode-attr.sh` in the session workloads); pre-registered gate → Task 6; Lever H → Task 7; Lever V → Tasks 8–9; split exit criterion → Task 5 (8c ceiling statement) + Task 11 (16c walk); ship gate → Tasks 10–11; metal budget (2 provisions STOP / 4 worst) → Tasks 4, 5, 10; every-STOP-recorded + AGENTS.md → Task 11.
- **Gated tasks are pre-written on purpose** (M4b.13 precedent): a rule firing must not stall on plan-writing; a rule NOT firing costs nothing but this text.
- **Known sketch-vs-toolchain seams** (flagged inline, structure-is-the-contract): `AlignedBuf` pointer accessors, rustix call shapes, AVX-512 intrinsic signatures, `hsum8_i32`'s parameter shape, the CLI invocation in Task 7 Step 5.
