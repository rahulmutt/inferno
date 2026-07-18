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
        // Pinned to 1 thread ON PURPOSE (M4b.1 spec): this gate measures
        // codegen quality against the interpreter; letting threading
        // inflate the ratio would hide codegen regressions behind
        // parallelism. Never "fix" a red nightly by unpinning this.
        let mut compiled = load_compiled(model, max_seq_len, 1)?;
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

pub struct Measurement {
    pub mean_tok_s: f64,
    pub stddev_tok_s: f64,
}

pub struct InfernoNumbers {
    pub pp: Measurement,
    pub tg: Measurement,
}

pub struct InfernoRun {
    pub headline: InfernoNumbers,
    /// Per-thread parity diagnostic: same backend, active threads capped to
    /// 1. None when the headline itself ran at t=1.
    pub t1: Option<InfernoNumbers>,
    /// The thread count the engine actually ran at (`Engine::threads()`,
    /// post-clamp to `1..=logical_cores`) — what the report must record,
    /// since the raw CLI-resolved value can exceed logical cores.
    pub threads: usize,
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

/// Measure compiled prefill/decode throughput at `threads` lanes, plus an
/// optional t=1 diagnostic pass over the SAME process-global pool (capped
/// via `set_global_active_threads` — the pool is sized once per process).
pub fn measure_inferno(
    model: &Path,
    pp: usize,
    tg: usize,
    reps: usize,
    threads: usize,
    t1_diag: bool,
) -> Result<InfernoRun, Box<dyn std::error::Error>> {
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
    let mut engine = Engine::load(model, max_seq_len)?;
    engine.set_threads(threads);
    engine.set_emitted_attn(std::env::var("INFERNO_EMITTED_ATTN").is_ok_and(|v| v == "1"));
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
    let headline = InfernoNumbers {
        pp: Measurement {
            mean_tok_s: pm,
            stddev_tok_s: ps,
        },
        tg: Measurement {
            mean_tok_s: tm,
            stddev_tok_s: ts,
        },
    };

    let t1 = if t1_diag && engine.threads() > 1 {
        assert!(
            inferno_pool::set_global_active_threads(1),
            "pool initialized by compiled_backend above"
        );
        run_once(&mut backend)?; // warmup at the new lane count
        let mut pp_s = Vec::with_capacity(reps);
        let mut tg_s = Vec::with_capacity(reps);
        for _ in 0..reps {
            let (p, t) = run_once(&mut backend)?;
            pp_s.push(p);
            tg_s.push(t);
        }
        inferno_pool::set_global_active_threads(engine.threads());
        let (pm, ps) = mean_stddev(&pp_s);
        let (tm, ts) = mean_stddev(&tg_s);
        Some(InfernoNumbers {
            pp: Measurement {
                mean_tok_s: pm,
                stddev_tok_s: ps,
            },
            tg: Measurement {
                mean_tok_s: tm,
                stddev_tok_s: ts,
            },
        })
    } else {
        None
    };
    Ok(InfernoRun {
        headline,
        t1,
        threads: engine.threads(),
    })
}

/// One recorded comparison data point (the `--json` shape; the human table
/// is rendered from the same struct). Recorded in the M4a spec's
/// Amendments section per the protocol.
#[derive(serde::Serialize)]
pub struct BenchReport {
    pub model: String,
    pub model_type: String,
    pub cpu_info: String,
    pub physical_cores: u32,
    pub logical_cores: u32,
    pub inferno_version: String,
    pub inferno_git: String,
    pub llama_build_commit: String,
    pub pp: u64,
    pub tg: u64,
    pub reps: u64,
    /// Headline inferno thread count (matched to llama.cpp's since M4b.1).
    /// The actual post-clamp count the engine ran at (`Engine::threads()`),
    /// not the raw CLI-resolved value — so this stays true even when
    /// `--threads` exceeds logical cores.
    pub inferno_threads: u64,
    pub llama_threads: u64,
    pub inferno_pp_tok_s: f64,
    pub inferno_pp_stddev: f64,
    pub inferno_tg_tok_s: f64,
    pub inferno_tg_stddev: f64,
    pub llama_pp_tok_s: f64,
    pub llama_pp_stddev: f64,
    pub llama_tg_tok_s: f64,
    pub llama_tg_stddev: f64,
    /// The `-t 1` per-thread-parity diagnostic rows (None when the
    /// full-thread run already was `-t 1`).
    pub llama_t1_pp_tok_s: Option<f64>,
    pub llama_t1_pp_stddev: Option<f64>,
    pub llama_t1_tg_tok_s: Option<f64>,
    pub llama_t1_tg_stddev: Option<f64>,
    /// inferno's own t=1 diagnostic (same pool, active threads capped to 1);
    /// None when the headline run was already t=1. Reads directly as the
    /// M4b.1 prefill-scaling measurement: headline pp / t1 pp.
    pub inferno_t1_pp_tok_s: Option<f64>,
    pub inferno_t1_pp_stddev: Option<f64>,
    pub inferno_t1_tg_tok_s: Option<f64>,
    pub inferno_t1_tg_stddev: Option<f64>,
}

fn render_table(r: &BenchReport) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(s, "model: {} ({})", r.model, r.model_type);
    let _ = writeln!(
        s,
        "cpu: {} ({} physical / {} logical cores)",
        r.cpu_info, r.physical_cores, r.logical_cores
    );
    let _ = writeln!(
        s,
        "inferno {} ({}) vs llama.cpp {} | pp={} tg={} reps={}",
        r.inferno_version, r.inferno_git, r.llama_build_commit, r.pp, r.tg, r.reps
    );
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "{:<22} {:>7} {:>18} {:>18}",
        "engine",
        "threads",
        format!("pp{} tok/s", r.pp),
        format!("tg{} tok/s", r.tg)
    );
    let mut row = |name: &str, threads: u64, pp: f64, pps: f64, tg: f64, tgs: f64| {
        let _ = writeln!(
            s,
            "{:<22} {:>7} {:>11.2} ± {:<5.2} {:>11.2} ± {:<5.2}",
            name, threads, pp, pps, tg, tgs
        );
    };
    row(
        "inferno (compiled)",
        r.inferno_threads,
        r.inferno_pp_tok_s,
        r.inferno_pp_stddev,
        r.inferno_tg_tok_s,
        r.inferno_tg_stddev,
    );
    if let (Some(pp), Some(pps), Some(tg), Some(tgs)) = (
        r.inferno_t1_pp_tok_s,
        r.inferno_t1_pp_stddev,
        r.inferno_t1_tg_tok_s,
        r.inferno_t1_tg_stddev,
    ) {
        row("inferno (t=1 diag)", 1, pp, pps, tg, tgs);
    }
    row(
        "llama.cpp",
        r.llama_threads,
        r.llama_pp_tok_s,
        r.llama_pp_stddev,
        r.llama_tg_tok_s,
        r.llama_tg_stddev,
    );
    if let (Some(pp), Some(pps), Some(tg), Some(tgs)) = (
        r.llama_t1_pp_tok_s,
        r.llama_t1_pp_stddev,
        r.llama_t1_tg_tok_s,
        r.llama_t1_tg_stddev,
    ) {
        row("llama.cpp (t=1 diag)", 1, pp, pps, tg, tgs);
    }
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "ratio (inferno/llama.cpp): pp {:.2}x | tg {:.2}x",
        r.inferno_pp_tok_s / r.llama_pp_tok_s.max(1e-9),
        r.inferno_tg_tok_s / r.llama_tg_tok_s.max(1e-9),
    );
    s
}

fn git_short_hash() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// `inferno bench`: the M4a manual comparison protocol (see the spec —
/// quiet hardware, devenv shell, release build; data points recorded in
/// the spec's Amendments, never a CI gate).
#[allow(clippy::too_many_arguments)]
pub fn bench(
    model: &Path,
    pp: u64,
    tg: u64,
    reps: u64,
    threads: u64,
    llama_bench_bin: Option<&Path>,
    json: bool,
) -> ExitCode {
    let inner = || -> Result<BenchReport, Box<dyn std::error::Error>> {
        if pp == 0 || tg == 0 || reps == 0 {
            return Err("--pp, --tg, and --reps must all be > 0".into());
        }
        let target = inferno_target::TargetDesc::detect()?;
        let threads = if threads == 0 {
            u64::from(target.topology.physical_cores)
        } else {
            threads
        };
        let inferno = measure_inferno(
            model,
            pp as usize,
            tg as usize,
            reps as usize,
            threads as usize,
            true,
        )?;
        let bin = llama_bench_bin
            .map(Path::to_path_buf)
            .unwrap_or_else(|| "llama-bench".into());
        let t_list: Vec<u64> = if threads == 1 {
            vec![1]
        } else {
            vec![threads, 1]
        };
        let rows = crate::llama_bench::run_llama_bench(&bin, model, pp, tg, &t_list, reps)?;
        let pick = |n_prompt: u64, n_gen: u64, t: u64| {
            crate::llama_bench::find_row(&rows, n_prompt, n_gen, t).ok_or_else(|| {
                format!("llama-bench output missing the (pp={n_prompt}, tg={n_gen}, t={t}) row")
            })
        };
        let lpp = pick(pp, 0, threads)?;
        let ltg = pick(0, tg, threads)?;
        let (t1pp, t1tg) = if threads == 1 {
            (None, None)
        } else {
            (Some(pick(pp, 0, 1)?), Some(pick(0, tg, 1)?))
        };
        Ok(BenchReport {
            model: model
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| model.display().to_string()),
            model_type: lpp.model_type.clone(),
            cpu_info: lpp.cpu_info.clone(),
            physical_cores: target.topology.physical_cores,
            logical_cores: target.topology.logical_cores,
            inferno_version: env!("CARGO_PKG_VERSION").into(),
            inferno_git: git_short_hash(),
            llama_build_commit: lpp.build_commit.clone(),
            pp,
            tg,
            reps,
            inferno_threads: inferno.threads as u64,
            llama_threads: threads,
            inferno_pp_tok_s: inferno.headline.pp.mean_tok_s,
            inferno_pp_stddev: inferno.headline.pp.stddev_tok_s,
            inferno_tg_tok_s: inferno.headline.tg.mean_tok_s,
            inferno_tg_stddev: inferno.headline.tg.stddev_tok_s,
            llama_pp_tok_s: lpp.avg_ts,
            llama_pp_stddev: lpp.stddev_ts,
            llama_tg_tok_s: ltg.avg_ts,
            llama_tg_stddev: ltg.stddev_ts,
            llama_t1_pp_tok_s: t1pp.map(|r| r.avg_ts),
            llama_t1_pp_stddev: t1pp.map(|r| r.stddev_ts),
            llama_t1_tg_tok_s: t1tg.map(|r| r.avg_ts),
            llama_t1_tg_stddev: t1tg.map(|r| r.stddev_ts),
            inferno_t1_pp_tok_s: inferno.t1.as_ref().map(|n| n.pp.mean_tok_s),
            inferno_t1_pp_stddev: inferno.t1.as_ref().map(|n| n.pp.stddev_tok_s),
            inferno_t1_tg_tok_s: inferno.t1.as_ref().map(|n| n.tg.mean_tok_s),
            inferno_t1_tg_stddev: inferno.t1.as_ref().map(|n| n.tg.stddev_tok_s),
        })
    };
    match inner() {
        Ok(report) => {
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report)
                        .expect("BenchReport serializes: plain numbers and strings")
                );
            } else {
                print!("{}", render_table(&report));
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
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
        let n = measure_inferno(&model, 8, 4, 2, 2, true).unwrap();
        for m in [&n.headline.pp, &n.headline.tg] {
            assert!(m.mean_tok_s.is_finite() && m.mean_tok_s > 0.0);
            assert!(m.stddev_tok_s.is_finite() && m.stddev_tok_s >= 0.0);
        }
        assert!(n.t1.is_some());
        let t1 = n.t1.unwrap();
        for m in [&t1.pp, &t1.tg] {
            assert!(m.mean_tok_s.is_finite() && m.mean_tok_s > 0.0);
            assert!(m.stddev_tok_s.is_finite() && m.stddev_tok_s >= 0.0);
        }
    }

    #[test]
    fn measure_inferno_rejects_prompt_beyond_context() {
        let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../crates/inferno-formats/tests/fixtures/tiny.gguf");
        // tiny.gguf's context length is far below 1<<20.
        assert!(measure_inferno(&model, 1 << 20, 4, 1, 1, false).is_err());
    }

    #[test]
    fn render_table_snapshot() {
        let r = BenchReport {
            model: "qwen2.5-0.5b-instruct-q8_0.gguf".into(),
            model_type: "qwen2 1B Q8_0".into(),
            cpu_info: "AMD Ryzen 9 3900 12-Core Processor".into(),
            physical_cores: 12,
            logical_cores: 24,
            inferno_version: "0.1.0".into(),
            inferno_git: "0b09ece".into(),
            llama_build_commit: "3ab8b3a9".into(),
            pp: 512,
            tg: 128,
            reps: 5,
            inferno_threads: 1,
            llama_threads: 12,
            inferno_pp_tok_s: 110.2,
            inferno_pp_stddev: 1.4,
            inferno_tg_tok_s: 26.1,
            inferno_tg_stddev: 0.3,
            llama_pp_tok_s: 486.4,
            llama_pp_stddev: 4.9,
            llama_tg_tok_s: 84.0,
            llama_tg_stddev: 0.8,
            llama_t1_pp_tok_s: Some(52.1),
            llama_t1_pp_stddev: Some(0.5),
            llama_t1_tg_tok_s: Some(9.3),
            llama_t1_tg_stddev: Some(0.1),
            inferno_t1_pp_tok_s: Some(58.0),
            inferno_t1_pp_stddev: Some(0.9),
            inferno_t1_tg_tok_s: Some(21.4),
            inferno_t1_tg_stddev: Some(0.2),
        };
        insta::assert_snapshot!("bench_report_table", render_table(&r));
    }
}
