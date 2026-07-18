# v1 Close — Criterion Verdict and Closing Record Design

This is the terminal milestone of v1. It ships no lever, no kernel, and no
codegen change. Its deliverable is a **verdict**: the v1 win criterion
("beat llama.cpp tokens/sec on prefill **and** decode") is **NOT MET**, and
the M4b campaign has measured why — well enough that the remaining gap is
attributed to structural walls rather than to unexplored headroom.

M4b.17's closing verdict named this item explicitly:

> With M4b.14's prefill closure and this milestone's decode-GEMV closure,
> every compiled-path streaming lever family is now measured at its wall;
> the v1-reckoning conversation (criterion redefinition or acceptance) is
> the successor item.

This spec is the acceptance branch of that conversation.

## Motivation

Seventeen M4b milestones ran the same discipline: pre-register a gate,
measure on quiet bare-metal hardware, ship the lever only if the gate
fires, and record every verdict — including the STOPs — as a finding. The
campaign produced ten shipped levers and six diagnostic closes, plus one
lever (M4b.5's decode-thread cap) that shipped, was measured a loss six
times, and was removed again by M4b.10, and one infrastructure milestone
(M4b.7's quiet-hardware runbook) that made the rest trustworthy. It did not
produce a win.

The reason to close rather than continue is not exhaustion; it is that the
last three milestones converted the remaining gap from *unknown headroom*
into *measured ceilings*. Continuing would mean building levers that the
project's own gate arithmetic has already ruled out.

| Phase | Where the wall was measured | Recorded quantity |
|---|---|---|
| Prefill GEMM | M4b.13 | Even a perfect VNNI GEMM at the spec's ½-ceiling yields 0.988 < 1.0 on the 8c box — Lever 2 STOP-out |
| Prefill attention | M4b.14 | Hard-ceiling walk: 0.934 < 1.0 on the 8c box; all-STOP on the Lever-2 menu |
| Decode attention | M4b.15/16 | The µbench's 28–37% whole-call headroom transferred as +0.35% / +0.98% e2e — STOP, flag default-off |
| Decode GEMV | M4b.17 | Achieved 40.7 GB/s vs GEMV-shaped roofline 40.49 GB/s (G ≈ 0); rule-3 STOP |

The decode-GEMV result is the decisive one. The shipping kernel runs
**0.5% above** its own GEMV-shaped roofline; the residual gap to the
sequential-stream ceiling is a shape tax (19.8% / 13.6%) plus a 16c-only
file-backing cost (7.2%) that hugepages were measured not to recover. There
is no streaming lever left that the evidence authorizes.

## Scope Decisions

| Decision | Choice |
|---|---|
| Verdict | Accept **NOT MET**; close v1 as-built with the ceiling arithmetic as the finding |
| Deliverable | Documentation only — a closing record, a v1-design amendment, and a repo front-door refresh |
| Release artifact | None. No version bump, no git tag, no GitHub release, no crates.io publish |
| Code changes | None to `crates/` or `cli/`. This milestone must produce an empty diff outside `docs/` and `README.md` |
| New benchmarks | None. No quiet-hw session, no metal provisioning, no new data point |
| v2 direction | Out of scope — its own brainstorm, as the item after this one |

Explicitly out of scope: redefining the criterion to something inferno
passes; re-benchmarking on Q4_K or a larger model to find friendlier
arithmetic; any "one more lever" that the recorded gates STOP'd. Each of
those is a legitimate future item, but folding one into the close would
make the close contingent on a measurement that has not happened.

## The Verdict

**The v1 win criterion is NOT MET.** Standing recorded ratios against
llama.cpp best-of-builds, from the M4b.16 protocol sessions of 2026-07-18
(the most recent protocol runs; M4b.17 ran no protocol session and carries
these forward):

| Machine | pp512 | tg128 |
|---|---|---|
| d2.c1.medium — Xeon Gold 6336Y, 16c | 0.83x | 0.96x |
| s2.c2.medium — Xeon E-2388G, 8c | 0.69x | 0.86x |

Criterion requires pp > 1.0x **and** tg > 1.0x. Decode came within 4% on
the 16c box; prefill did not come close on either.

Two caveats the source tables carry, recorded so the close is not read as
more precise than the data:

- **Prefill has wide error bars.** The 16c session measured inferno at
  873.69 ± 123.66 and llama at 1209.34 ± 243.15 pp tok/s. The two bases
  disagree accordingly — 0.72x against llama's pure-CPU build in the reps=5
  table, 0.83x against best-of-builds from the independent `--json` run.
  This spec cites the best-of figure throughout for consistency with the
  criterion's basis, but the honest statement is that 16c prefill sits
  somewhere in the low 0.7x–0.8x range, not at a sharp 0.83x.
- **The 16c decode figure is 1.00x against one llama build.** Inferno
  measured 58.06 ± 0.33 against the pure-CPU build's 58.05 ± 0.46 — parity.
  It is 0.96x only against best-of, because llama's BLAS build reached
  59.91. The criterion is explicitly "vs llama at its best", so 0.96x is
  the governing number; the 1.00x row is not a win and must not be cited as
  one.

Two properties of these numbers matter for how the close is written. They
are measured against llama.cpp **at its genuine best** — a per-metric
maximum over a pure-CPU ggml build and a BLAS build, a comparator that
M4a's 2026-07-11 fix deliberately strengthened (~4x on pp) after
discovering the original was BLAS-confounded. And they are full-thread
protocol numbers, not the friendlier t=1 diagnostic rows. The close inherits
that honesty; the record must not quietly swap in a weaker basis.

## Structure

Three artifacts, in dependency order.

**1. This spec, as the citable verdict record.** It carries the verdict
above, the campaign findings ledger below, and the closing verification. Any
future reader asking "did inferno beat llama.cpp, and what is actually
known about why not" should be able to answer from this file alone, with
every figure traceable to the milestone spec that recorded it.

**2. A closing amendment in the v1 design doc**
(`2026-07-04-inferno-v1-design.md`). The v1 design states the criterion and
the M0–M4 milestone list; it currently reads as though the outcome is still
open. The amendment appends — never edits — a short section recording that
M4 closed with the criterion NOT MET, pointing at this spec, and noting that
the Risks section's "Beating llama.cpp is hard" entry called the outcome
correctly. The specialization bet was tested, not hand-waved.

**3. A repo front-door refresh** (`README.md`). This is the only
user-visible artifact of the close, and it is currently wrong in ways
unrelated to the verdict: it claims "milestone M3" as status and "LLVM 18"
as the native dependency (the toolchain moved to LLVM 22.1.8 in `ee03def`).
The refresh corrects both, and replaces the status line with an honest
statement of what v1 is: a complete CPU inference engine — GGUF + MLX, LLVM
codegen, artifact cache, threaded compiled path — that reaches roughly
0.83x prefill and 0.96x decode of llama.cpp's best builds on the recorded
quiet hardware, with the gap attributed. A README must not claim a win the
specs record as not met, nor bury the result.

## Campaign Findings Ledger

The consolidated record of what M4b bought and what it proved. Shipped
levers, with their recorded gains:

| Milestone | Lever | Recorded outcome |
|---|---|---|
| M4b.1 | Multi-threaded generated code | Prefill scale 10.63x @ t=12 (gate ≥6x MET, closed by M4b.9); 346.8 → 652.4 pp tok/s (+88%) |
| M4b.2 | Prefill tiles | t=1 prefill ~1.7x; exit criterion 0.70x NOT MET at 0.55x |
| M4b.3 | Vectorized AVX2 attention | pp 1.331x / tg 1.087x vs its own baseline; attention share 35% → 26.4–26.8% |
| M4b.4 | Decode GEMV prefetch@4 | Kept for Q8_0, reverted for Q4_K (measured to hurt there); interleave not authorized |
| M4b.8 | Parallel prefill attention | 5.67x @ t=12 (gate NOT MET); 249.9 → 346.8 pp tok/s |
| M4b.9 | Serial-tail parallelization | 5.67x → 10.63x @ t=12, gate MET; Amdahl serial fraction 10.2% → 1.2% |
| M4b.10 | Decode-cap removal | Removed M4b.5's cap; regret(U) ≤ 5% on all three machines. The bandwidth model behind the cap was **refuted** |
| M4b.11 | Head-sharded decode attention | decode tg **+12.7% (16c) / +13.4% (8c)** — the campaign's largest single decode gain. F16 KV (Lever 2) STOP'd |
| M4b.13 | Register-tiled Q8_0 prefill GEMM | µbench MR=4 geomean +11.9%; t=1 pp +6.8% / +5.0% |
| M4b.14 | Query-blocked prefill attention | t=1 prefill −16.8% / −18.4%; attention bracket −44%; pp 0.74x → 0.79x on 16c, bit-identical |

Diagnostic closes — no lever shipped, a finding recorded instead. These are
the load-bearing part of the close, because they are what makes "no
remaining headroom" a measurement rather than an assertion:

| Milestone | Finding |
|---|---|
| M4b.5 | The phase-aware decode cap was a **loss**, missing the best fixed cap six times (−11.72% / −7.11% / −1.62% latterly). Removed in M4b.10 |
| M4b.6 | GEMV op-reduction is dead cross-vendor — a loss on Zen 2, a 6-rep wash on 6336Y, a loss on E-2388G. NO-SHIP final |
| M4b.12 | Decode-attention dispatch overhead is not the wall: wake 0.000%, alloc ≤0.057%, publish ≤0.179%. All three gates STOP. Kernel is 72.5% / 94.2% of the instrumented call |
| M4b.15 | A const-geometry-specialized compile of the same kernel source leaves 28–37% whole-call on the table; the phase-marginal instrument itself failed admissibility on all three machines |
| M4b.16 | That 28–37% µbench headroom transfers to **+0.35% / +0.98%** e2e — a naive transfer predicted 6–10% / 4–7%. Ships default-off behind `INFERNO_EMITTED_ATTN` |
| M4b.17 | Decode GEMV runs 0.5% **above** its GEMV-shaped roofline (G = −0.2 GB/s). Residual = shape tax (19.8% / 13.6%) + 16c file-backing cost (7.2%), and hugepages recover none of it. Both bandwidth lever families foreclosed |

Two methodology findings outlive the campaign and belong in the record.
First, M4a's comparator fix (2026-07-11): the original llama.cpp baseline
was BLAS-confounded and roughly 4x too weak on pp; every ratio before that
date is not comparable to the ones after. Second, the recurring gap between
µbench headroom and end-to-end transfer — M4b.16 is the sharpest instance,
where a real 28–37% kernel-level win produced under 1% e2e. A µbench
number is a hypothesis about the wall, not a measurement of it.

## Verification

Because the deliverable is documentation, verification is about the
*absence* of change and the *accuracy* of claims, not about behavior.

1. **Empty code diff.** `git diff main -- crates/ cli/ fuzz/ scripts/` must
   be empty at close. The close cannot smuggle in a change.
2. **Standing invariants still hold**, recorded once here as every prior
   milestone does: `mise run test`, `mise run lint`, the kernel rig, the
   codegen differential, and the artifact differential. These are expected
   to pass unchanged from `708050d`, and the point of running them is to
   record that v1 closes green rather than to detect a regression from work
   this milestone did not do.
3. **Every ledger figure traces to its source spec.** Each number in the
   findings ledger and the verdict table is quoted from the milestone spec
   that recorded it. No figure may be recomputed, rounded differently, or
   reconstructed from memory — recorded data points are append-only, and
   that rule extends to citing them.
4. **README claims match the specs.** The status line's ratios must equal
   the verdict table's; the LLVM version must equal `devenv.nix`'s.
5. **No new data point is created.** This milestone provisions no metal and
   records no session. Metal budget: zero.

## Risks

- **A close reads as an ending.** It is not: v1 is a working engine, and the
  v2 direction (NEON/Apple Silicon, AOT cross-compilation, then server mode)
  is where the specialization bet gets its second test on hardware whose
  wall arithmetic differs. Mitigation: the close states this, and the v2
  brainstorm is the immediately following item.
- **Temptation to soften the verdict.** The strongest recorded tg is 0.96x,
  and "within noise of parity on decode" is a true sentence that would read
  as a near-win. It is not the criterion. Mitigation: the criterion is
  quoted verbatim and the verdict stated before any mitigating context.
- **Selective ratio citation.** Several bases exist (plain build vs
  best-of-builds, t=1 vs full-thread, protocol vs gate A/B tables), and
  M4b.15 already needed an erratum for citing the plain-build basis. The
  close uses best-of-builds full-thread protocol numbers throughout, and
  says so.
- **The ledger is a re-derivation risk.** Compressing seventeen milestones
  into two tables invites paraphrase drift. Mitigation: verification item 3,
  plus the rule that the owning spec always wins on any discrepancy.
