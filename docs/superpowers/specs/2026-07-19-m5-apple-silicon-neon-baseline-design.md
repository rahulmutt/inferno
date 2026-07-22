# M5 — Apple Silicon NEON Bring-up & Baseline Design

This is the first milestone of v2. v1 closed with the win criterion (beat
llama.cpp on prefill **and** decode) NOT MET on x86, and the M4b campaign
attributed the remaining gap to measured ceilings rather than unexplored
headroom (see
[the v1 close record](2026-07-18-v1-close-design.md)). The v1 design and its
close both name the same successor: the NEON / Apple Silicon target, *"where
the specialization bet gets its second test on hardware whose wall arithmetic
differs."*

M5 is the **bring-up half** of that test. It ports the compiled path to
Apple Silicon and measures one honest baseline against llama.cpp. It ships
**no performance gate** — its deliverable is a *correct compiled path on ARM*
and a *baseline with the gap attributed*, so that the lever campaign after it
can pre-register meaningful gates against a known wall. This mirrors how v1
actually ran: M1–M3 established correctness before M4 chased the win.

## Motivation

The v1 close made "no remaining x86 headroom" a measurement, not an
assertion, by walking each streaming lever family to its wall. Apple Silicon
is a different machine: unified memory (no NUMA), a wide performance-core
cluster beside efficiency cores, and — the crux — an undocumented **AMX**
matrix coprocessor that llama.cpp reaches through Apple's **Accelerate**
framework. Inferno's thesis is *compile the model for the machine via LLVM*,
and pure LLVM codegen targets **NEON, not AMX**. So the ARM head-to-head
carries an asymmetry x86 did not: pure-NEON inferno can lose matmul-heavy
prefill to Accelerate/AMX llama through *hardware access*, independent of
codegen quality.

That asymmetry is exactly what makes the baseline worth measuring before
committing levers. The wrong move is to guess where the wall sits and
pre-register perf gates against the guess; the M4b discipline is to measure
the wall first, then let the gate arithmetic decide which levers are even
authorized. M5 measures the wall. Whether inferno should later emit AMX/SME
to close the asymmetry — eroding the pure-codegen thesis to match the
comparator's weapon — is a real question, but it is the *next* milestone's
question, and it cannot be scoped honestly until this baseline exists.

## Scope Decisions

| Decision | Choice |
|---|---|
| Milestone goal | A correct, differential-clean compiled path on an M1 Mac, plus one honest quiet-hw baseline vs llama.cpp, gap attributed |
| Win gate | **None.** No perf gate; "correct + measured + attributed" is the exit |
| Target chip | Apple M1 (base/Pro/Max — exact tier pinned below). AMX gen1, **no SME**, 128-bit NEON. A conservative floor: if the wall arithmetic works here it generalizes upward |
| Kernel weapon | **Pure LLVM-NEON codegen.** AMX/SME deliberately unused this milestone |
| Comparator | llama.cpp at its genuine best (Accelerate/AMX build) **and** a NEON-only (`LLAMA_NO_ACCELERATE`) build — both CPU-only; Metal/GPU excluded. Report both ratios |
| Relationship to x86 | Strictly **additive**. NEON sits beside AVX2/AVX-512, selected by target descriptor; every x86 gate stays green |
| Measurement | One physical M1 the author owns; a new single-Mac quiet-hw runbook. One data point, not the multi-machine sweep the x86 verdicts used |
| Provisioning | **None.** No `metal/` work, no cloud Macs. Metal budget: zero |

Explicitly out of scope: emitting AMX or SME instructions; calling Accelerate
or any vendor BLAS; a multi-chip Apple Silicon sweep; cloud-Mac provisioning;
a performance gate or win claim of any kind; AOT cross-compilation
(`--target`) and server mode (the later v2 sub-projects). Each is a
legitimate future item; folding one in would make the bring-up contingent on
work it does not need.

**Tier to pin at implementation start:** the exact M1 (core count, P/E split,
memory bandwidth class, chassis/cooling) is recorded in the runbook and cited
by every baseline number. The spec is written against "M1"; the first
Amendment fixes the precise machine.

## Architecture — what changes, crate by crate

The port is additive. NEON is a second SIMD target selected by the target
descriptor; no x86 path is removed or altered in behavior.

- **`inferno-formats`** — no change. Format parsing is arch-independent and
  stays `#![forbid(unsafe_code)]`.
- **`inferno-graph`** — no ISA work. The only change is *tolerance
  re-derivation* (see Correctness Net): ARM FMA contraction and NEON rounding
  differ from AVX, so `LOGIT_TIE_EPSILON`, `gemv_rel_tol`, and
  `logits_abs_tol` are re-derived from observed ARM error distributions.
- **`inferno-kernels`** — the main lift. NEON microkernel variants beside the
  existing scalar/AVX2/AVX-512, holding the rig's **scalar-vs-NEON
  bit-identity** on ARM. Cross-arch bit-identity to x86 is *not* expected and
  *not* asserted — different FP hardware; only same-arch scalar-vs-SIMD
  equality holds. `inferno-kernels` keeps its existing `unsafe` opt-out for
  intrinsics.
- **`inferno-codegen`** — ARM64 (`aarch64-apple-darwin`) target triple, NEON
  vector types/intrinsics in emitted LLVM IR, **AMX unused**. Object emit →
  link to a **`.dylib`** (macOS `-dynamiclib`, not ELF `.so`) → `dlopen` from
  the on-disk artifact cache. Object-emit + `dlopen` remains the only compile
  path; there is no in-memory JIT.
- **`inferno-core`** — `dlopen`, `mmap`, and `madvise` have macOS
  equivalents; the artifact-cache artifact becomes `model.dylib` on darwin.
  Keeps its existing `unsafe` opt-out.
- **Hardware detection** — the `sysctl`-on-macOS path the v1 design already
  anticipated, populating the *same* target descriptor (P/E topology, cache
  sizes, page size, bandwidth class, NEON; no SME on M1).
- **Not touched** — `metal/` (own Mac, no provisioning); GPU anything
  (permanently out of scope for the project).

## Structure — the three slices (Approach A: toolchain first, tune second)

Sequenced so the scariest unknown — does the whole darwin toolchain path work
end to end — is proven before any kernel investment.

**Slice 1 — Toolchain bring-up.** Prove the darwin path end to end: devenv +
LLVM 22.1.8 on `aarch64-darwin`, emit an ARM64 object, link a `.dylib`,
`dlopen` it, run a real model, and pass the **codegen differential natively
on the Mac** (M1 scalar interpreter vs ARM-compiled path) — using
naive/portable NEON kernels. The correctness gate is live from here. If the
toolchain fights, it fails here, cheaply, before kernels.

**Slice 2 — Real NEON microkernels.** Translate the AVX2 tile structure to
NEON, holding rig bit-identity (scalar vs NEON on ARM), so the compiled path
is *competently tuned* — enough that the baseline measures the ARM wall, not
un-tuned laziness. No perf gate; the bar is "a serious NEON kernel, not a
placeholder," judged by the rig and by not leaving obvious NEON headroom on
the table.

**Slice 3 — Baseline.** Stand up the single-Mac quiet-hw runbook, then run
the protocol (Qwen2.5-0.5B-Instruct Q8_0, pp512/tg128, full-thread) against
both llama.cpp builds, and attribute the gap. One recorded baseline; the
milestone's citable output.

## Correctness Net

M5 ships no perf gate, so *correct* is the deliverable and the correctness
net is load-bearing.

1. **Codegen differential is the primary gate**, run natively on the M1: the
   portable scalar interpreter vs the ARM-compiled path, same model, same
   inputs, agreement within re-derived tolerance. This is
   `cargo test -p inferno-codegen --test differential`, now on ARM hardware.
2. **Kernel rig** asserts scalar-vs-NEON **exact** bit-identity on ARM (the
   standing "SIMD variants must stay bit-identical" rule). No cross-arch
   equality against x86 is claimed or tested.
3. **Artifact differential** (`cargo test -p inferno-core --test artifact`)
   green on the Mac.
4. **Tolerance re-derivation is an evidenced, recorded step, not a knob.**
   `logits_abs_tol` / `gemv_rel_tol` / `LOGIT_TIE_EPSILON` are re-derived from
   the rig's `observed_error_*` diagnostics on ARM, and the derivation is
   recorded in Amendments. They are **never** nudged to make a red test green
   — the standing discipline, applied to a new ISA.
5. **x86 non-regression.** Every existing x86 gate — `mise run test` /
   `lint`, kernel rig, both differentials, the nightly speedup gate — stays
   green in CI. The port adds a code path; it removes none.

## Measurement Rig — the single-Mac quiet-hw runbook

A genuinely new artifact; none of the x86 quieting transfers.

- **No numactl / no NUMA** (unified memory). The analog of x86
  socket-pinning is **P-core steering**: drive the benchmark threads onto the
  performance cluster (via QoS class), keep E-cores out of the measured path,
  and record which cores actually ran.
- **Thermal honesty.** Sustained pp512/tg128 on a fanless M1 (Air) throttles;
  a cooled M1 (mini / 14" Pro) is quieter. The runbook records the chassis,
  watches for throttle, and **discards throttled reps** rather than averaging
  them in.
- **Quiescing checklist** — Spotlight/indexing off, background apps closed,
  network idle: the darwin equivalent of the x86 quiet checklist.
- **Comparators, both CPU-only** (Metal/GPU excluded): an Accelerate/AMX
  llama build and a `LLAMA_NO_ACCELERATE` NEON-only build. The baseline
  reports **both** ratios — llama-at-its-best (the criterion's basis) and
  codegen-vs-codegen (NEON-only, isolating codegen quality from the hardware
  gap) — the same best-of honesty v1 used.

## Exit Criteria

All must hold; **none is a performance gate.**

1. The compiled path runs a real model correctly on the M1, with the codegen
   differential, artifact differential, and kernel rig all **green
   on-device**.
2. x86 CI unchanged and green (non-regression).
3. Tolerance re-derivation recorded with its ARM error data.
4. **One** baseline recorded to Amendments: inferno pp512/tg128 vs both llama
   builds, with the gap **attributed** — how much is codegen quality
   (readable from the NEON-only ratio) vs the AMX/Accelerate hardware
   asymmetry vs NEON-kernel headroom still on the table.
5. A recorded verdict naming what the next milestone should attack — the
   scoped input to the "should inferno emit AMX/SME" lever question.

## Verification

1. **On-device green.** All three correctness gates pass on the M1, recorded
   in Amendments with the machine identified.
2. **x86 untouched in behavior.** The x86 gates pass in CI unchanged; the
   diff to x86 code paths is additive only (new NEON branches, no altered
   AVX behavior).
3. **Baseline is protocol-faithful.** Same model, quant, and pp512/tg128
   protocol as the x86 verdicts, full-thread, both llama builds, on quiet
   hardware per the new runbook. Recorded once, append-only.
4. **Attribution is traceable.** Each component of the gap (codegen,
   hardware asymmetry, kernel headroom) cites the measurement it rests on —
   the NEON-only ratio for codegen quality, the best-of ratio for the
   asymmetry.
5. **No scope creep.** No AMX/SME emitted, no Accelerate linked, no cloud
   Mac provisioned, no perf gate asserted. Metal budget: zero.

## Risks

- **The darwin toolchain doesn't cooperate** — LLVM 22.1.8 on
  `aarch64-darwin`, `-dynamiclib` linking, `dlopen` semantics, devenv on
  darwin. *Mitigation: it is Slice 1 by design — fail fast, before kernel
  investment.*
- **One chip, one data point.** v1's verdicts leaned on multi-machine sweeps;
  M5 cannot. *Mitigation: the baseline is stated as a floor that generalizes
  downward, not a proof; the multi-chip sweep is explicitly the lever
  campaign's job.*
- **The AMX asymmetry makes prefill look hopeless.** Pure-NEON pp may sit far
  below Accelerate llama. *Mitigation: that is a finding to attribute, not a
  milestone failure — the NEON-only ratio isolates codegen quality from the
  hardware gap, and attribution is the deliverable.*
- **Tolerance re-derivation becomes a fudge.** Pressure to widen tolerances
  to get green on-device. *Mitigation: the derivation must cite ARM error
  data and is append-only, the same rule every prior milestone held.*
- **A bring-up reads as a win attempt and disappoints.** M5 ships no win.
  *Mitigation: the spec states up front that the win gate is the next
  milestone's; M5's success is a correct path and an attributed baseline.*

## Amendments

_(Append-only. The precise M1 tier, the on-device verification walk, the
tolerance re-derivation data, and the recorded baseline land here at
implementation time. No data point in this section is ever edited.)_

### Task 1 — machine pinned

- **Chip:** Apple M1 Max (`machdep.cpu.brand_string`); `hw.model` = MacBookPro18,2
- **Cores:** 10 physical / 10 logical (no SMT); `hw.perflevel0.physicalcpu` = 8 P-cores, `hw.perflevel1.physicalcpu` = 2 E-cores
- **Caches / memory:** L1d 65536 B (64 KiB), L2 4194304 B (4 MiB), cacheline 128 B, page size 16384 B (16 KiB), RAM 68719476736 B (64 GiB)
- **ISA features:** `hw.optional.neon` = 1, `AdvSIMD` = 1, `FEAT_DotProd` = 1 (present); `FEAT_I8MM` = 0 (absent on M1 — arrived with M2); `FEAT_SME` = 0
- Recorded 2026-07-19 for M5 Task 1 → `Isa::Aarch64Neon`; `Feature::Dotprod` detected, `Feature::I8mm` not.
