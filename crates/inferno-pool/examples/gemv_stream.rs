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
    Mmap {
        ptr: *mut u8,
        len: usize,
    },
    Anon {
        ptr: *mut u8,
        len: usize,
    },
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
    Arm {
        name: "heap",
        mats,
        _backing: Backing::Heap(bufs),
    }
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
    Arm {
        name: "mmap4k",
        mats,
        _backing: Backing::Mmap { ptr, len: total },
    }
}

fn thp_arm(images: &[(inferno_kernels::AlignedBuf, usize, usize, Class)]) -> Arm {
    let total = pad4k(
        images
            .iter()
            .map(|(b, ..)| pad4k(b.as_slice().len()))
            .sum::<usize>(),
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
    Arm {
        name: "thp",
        mats,
        _backing: Backing::Anon { ptr, len: total },
    }
}

/// AnonHugePages for the smaps region containing `addr` (0 if unreadable —
/// corroboration, not a gate quantity).
fn anon_huge_kb(addr: usize) -> u64 {
    let Ok(smaps) = std::fs::read_to_string("/proc/self/smaps") else {
        return 0;
    };
    let mut in_region = false;
    for line in smaps.lines() {
        if let Some((range, _)) = line.split_once(' ')
            && let Some((lo, hi)) = range.split_once('-')
            && let (Ok(lo), Ok(hi)) = (usize::from_str_radix(lo, 16), usize::from_str_radix(hi, 16))
        {
            in_region = lo <= addr && addr < hi;
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
    let lanes: usize = args
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });
    let layers: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(N_LAYERS);
    let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let spin: Option<(String, u64)> = args.iter().position(|a| a == "--spin").map(|i| {
        (
            args[i + 1].clone(),
            args[i + 2].parse().expect("--spin <arm> <secs>"),
        )
    });

    let isa = if KernelIsa::Avx2.available() {
        KernelIsa::Avx2
    } else {
        KernelIsa::Scalar
    };
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

    let arms = [
        heap_arm(layers, &images),
        mmap4k_arm(&images),
        thp_arm(&images),
    ];

    if let Some((arm_name, secs)) = spin {
        let arm = arms
            .iter()
            .find(|a| a.name == arm_name)
            .expect("--spin arm name");
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
                let mut v: Vec<f64> = per_class.iter().map(|t| t[ci].as_secs_f64()).collect();
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
