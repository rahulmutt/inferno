//! `inferno bench-compiled`: the M3 "faster than the interpreter" nightly
//! gate. Generates the same number of tokens from the same prompt with the
//! compiled backend and with the M1 interpreter, measures decode tok/s for
//! each via `GenStats::decode_secs`, and asserts the compiled path clears a
//! conservative multiple of the interpreter's throughput.
//!
//! Not part of the blocking PR tier: it needs a real (non-fixture) model and
//! a first compile, both too slow for the 5-minute budget. Wired into
//! `.github/workflows/nightly.yml` instead (see `mise run speedup`).

use std::ops::ControlFlow;
use std::path::Path;
use std::process::ExitCode;

use inferno_runtime::{Generator, Greedy};

use crate::run::load_compiled;

/// Conservative floor for compiled-vs-interpreter decode throughput.
///
/// Chosen BELOW the first observed real-model speedup — see the "First
/// speedup data point" amendment in
/// `docs/superpowers/specs/2026-07-05-m3-compiler-design.md` for the
/// measurement this was set from (model, quant, machine, date). Leaves
/// headroom against machine noise while still catching a real regression.
/// Never lower this to turn a red nightly run green; if the compiled path
/// stops clearing it, that's a real finding, not a test to fix.
pub const MARGIN: f64 = 3.0;

/// Runs the compiled-vs-interpreter decode-throughput comparison and prints
/// the result. Returns `ExitCode::FAILURE` if the compiled path's decode
/// tok/s is not at least `MARGIN * interpreter tok/s`, or if generation
/// errors out on either backend.
pub fn bench_compiled(
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    max_seq_len: usize,
) -> ExitCode {
    let inner = || -> Result<(f64, f64), Box<dyn std::error::Error>> {
        // Compiled first: this is also the first compile, so its cost (LLVM
        // codegen + link) lands here rather than skewing the interpreter run.
        let mut compiled = load_compiled(model, max_seq_len)?;
        let (_, compiled_stats) =
            compiled.generate(prompt, max_tokens, &mut Greedy, &mut |_| {
                ControlFlow::Continue(())
            })?;
        let compiled_tok_s = compiled_stats.generated as f64 / compiled_stats.decode_secs.max(1e-9);

        let mut interp = Generator::load(model, max_seq_len)?;
        let (_, interp_stats) = interp.generate(prompt, max_tokens, &mut Greedy, &mut |_| {
            ControlFlow::Continue(())
        })?;
        let interp_tok_s = interp_stats.generated as f64 / interp_stats.decode_secs.max(1e-9);

        Ok((compiled_tok_s, interp_tok_s))
    };

    match inner() {
        Ok((compiled_tok_s, interp_tok_s)) => {
            let speedup = compiled_tok_s / interp_tok_s.max(1e-9);
            println!(
                "compiled: {compiled_tok_s:.2} tok/s | interp: {interp_tok_s:.2} tok/s | speedup: {speedup:.2}x (margin {MARGIN:.1}x)"
            );
            if compiled_tok_s >= interp_tok_s * MARGIN {
                ExitCode::SUCCESS
            } else {
                eprintln!(
                    "FAIL: compiled decode throughput {compiled_tok_s:.2} tok/s does not clear \
                     {MARGIN:.1}x interpreter throughput {interp_tok_s:.2} tok/s (needed >= {:.2} tok/s)",
                    interp_tok_s * MARGIN
                );
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
