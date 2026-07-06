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

use std::time::Instant;

use inferno_core::Engine;
// `Greedy` is already brought into scope by the `use inferno_runtime::{Generator,
// Greedy};` above (frozen `bench_compiled` code) — importing only what's new here
// avoids an E0252 duplicate-import error without touching that line.
use inferno_runtime::{Backend, Sampler};

use crate::run::clamp_max_seq_len;

#[allow(dead_code)] // fields consumed in Task 8
pub struct Measurement {
    pub mean_tok_s: f64,
    pub stddev_tok_s: f64,
}

#[allow(dead_code)] // fields consumed in Task 8
pub struct InfernoNumbers {
    pub pp: Measurement,
    pub tg: Measurement,
}

/// Sample mean and sample (n-1) standard deviation; stddev is 0 for n < 2.
fn mean_stddev(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    if samples.len() < 2 {
        return (mean, 0.0);
    }
    let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

/// Measure compiled prefill (pp synthetic tokens) and decode (tg greedy
/// steps) throughput, `reps` timed repetitions after one untimed warmup.
/// Drives the `Backend` directly: no tokenizer/EOS/UTF-8 in the timed path.
#[allow(dead_code)] // consumed in Task 8
pub fn measure_inferno(
    model: &Path,
    pp: usize,
    tg: usize,
    reps: usize,
) -> Result<InfernoNumbers, Box<dyn std::error::Error>> {
    let needed = pp + tg;
    let max_seq_len = clamp_max_seq_len(model, needed)?;
    if max_seq_len < needed {
        return Err(format!(
            "model context length {max_seq_len} is too small for pp={pp} + tg={tg}"
        )
        .into());
    }
    let desc = inferno_formats::load_desc(model)?;
    // vocab_size is u64 in ModelDesc; token ids are u32.
    let vocab = u32::try_from(desc.hyperparams.vocab_size).map_err(|_| "vocab size exceeds u32")?;
    if vocab < 2 {
        return Err("model vocab too small for synthetic prompt".into());
    }
    // Synthetic prompt: valid ids cycling [1, vocab). Content is irrelevant
    // to throughput (mirrors llama-bench's approach).
    let ids: Vec<u32> = (0..pp).map(|i| 1 + (i as u32 % (vocab - 1))).collect();

    // Compile (or cache-hit) happens here, outside any timed region.
    let engine = Engine::load(model, max_seq_len)?;
    let mut backend = engine.compiled_backend()?;

    let run_once = |backend: &mut dyn Backend| -> Result<(f64, f64), Box<dyn std::error::Error>> {
        backend.reset();
        let t0 = Instant::now();
        let mut last = backend.forward(&ids)?;
        let pp_secs = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let mut greedy = Greedy;
        for _ in 0..tg {
            let next = greedy.sample(&last);
            last = backend.forward(&[next])?;
        }
        let tg_secs = t1.elapsed().as_secs_f64();
        Ok((pp as f64 / pp_secs.max(1e-9), tg as f64 / tg_secs.max(1e-9)))
    };

    run_once(&mut backend)?; // warmup: touches every mmap'd weight page
    let mut pp_samples = Vec::with_capacity(reps);
    let mut tg_samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let (p, t) = run_once(&mut backend)?;
        pp_samples.push(p);
        tg_samples.push(t);
    }
    let (pm, ps) = mean_stddev(&pp_samples);
    let (tm, ts) = mean_stddev(&tg_samples);
    Ok(InfernoNumbers {
        pp: Measurement {
            mean_tok_s: pm,
            stddev_tok_s: ps,
        },
        tg: Measurement {
            mean_tok_s: tm,
            stddev_tok_s: ts,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_stddev_matches_hand_computation() {
        let (m, s) = mean_stddev(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((m - 5.0).abs() < 1e-12);
        // Sample stddev (n-1) of that set is sqrt(32/7).
        assert!((s - (32.0f64 / 7.0).sqrt()).abs() < 1e-12);
        let (m1, s1) = mean_stddev(&[3.5]);
        assert_eq!((m1, s1), (3.5, 0.0));
    }

    /// End-to-end smoke on the tiny fixture: compiles once (artifact cache),
    /// then measures. Numbers must be finite and positive; anything else
    /// means the timed path is broken, not slow.
    #[test]
    fn measure_inferno_smoke_on_fixture() {
        let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../crates/inferno-formats/tests/fixtures/tiny.gguf");
        let n = measure_inferno(&model, 8, 4, 2).unwrap();
        for m in [&n.pp, &n.tg] {
            assert!(m.mean_tok_s.is_finite() && m.mean_tok_s > 0.0);
            assert!(m.stddev_tok_s.is_finite() && m.stddev_tok_s >= 0.0);
        }
    }

    #[test]
    fn measure_inferno_rejects_prompt_beyond_context() {
        let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../crates/inferno-formats/tests/fixtures/tiny.gguf");
        // tiny.gguf's context length is far below 1<<20.
        assert!(measure_inferno(&model, 1 << 20, 4, 1).is_err());
    }
}
