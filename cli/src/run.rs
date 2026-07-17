use std::io::Write;
use std::ops::ControlFlow;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use inferno_core::Engine;
use inferno_runtime::{
    Backend, ChainSampler, Generator, Greedy, RuntimeError, Sampler, SamplerConfig,
};

#[allow(clippy::too_many_arguments)]
pub fn run(
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    max_seq_len: usize,
    interp: bool,
    threads: u64,
    sampling: SamplerConfig,
    profile: bool,
) -> ExitCode {
    if let Err(e) = sampling.validate() {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    if profile {
        if interp {
            eprintln!("error: --profile requires the compiled path (incompatible with --interp)");
            return ExitCode::FAILURE;
        }
        return match run_profile(model, prompt, max_tokens, max_seq_len, threads) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        };
    }
    let mut sampler = ChainSampler::new(sampling);
    let generator = if interp {
        Generator::load(model, max_seq_len).map_err(|e| e.to_string())
    } else {
        load_compiled(model, max_seq_len, threads).map_err(|e| e.to_string())
    };
    let mut generator = match generator {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut stdout = std::io::stdout().lock();
    // Track the first stdout write/flush failure (e.g. `| head` closing the
    // pipe) so we can tell the interpreter to stop generating immediately
    // instead of grinding through the remaining --max-tokens with a dead
    // consumer, then report failure — never SUCCESS on a broken pipe.
    let mut stdout_err: Option<std::io::Error> = None;
    let result = generator.generate(prompt, max_tokens, &mut sampler, &mut |bytes| match stdout
        .write_all(bytes)
        .and_then(|()| stdout.flush())
    {
        Ok(()) => ControlFlow::Continue(()),
        Err(e) => {
            stdout_err = Some(e);
            ControlFlow::Break(())
        }
    });
    if let Some(e) = stdout_err {
        eprintln!("error: failed to write to stdout: {e}");
        return ExitCode::FAILURE;
    }
    match result {
        Ok((_, stats)) => {
            let _ = writeln!(stdout);
            eprintln!(
                "prefill: {} tok in {:.1}s ({:.2} tok/s) | decode: {} tok in {:.1}s ({:.2} tok/s)",
                stats.prompt_tokens,
                stats.prefill_secs,
                stats.prompt_tokens as f64 / stats.prefill_secs.max(1e-9),
                stats.generated,
                stats.decode_secs,
                stats.generated as f64 / stats.decode_secs.max(1e-9),
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The dedicated `--profile` measurement path (Task 4, per the M4b.2 plan's
/// own recommendation): bypasses `Generator`/streaming entirely and drives
/// a `CompiledBackend` directly, so the normal generation loop stays
/// untouched by profiling concerns.
///
/// Measures exactly two phases: one prefill call over the whole prompt, then
/// up to `max_tokens` greedy decode steps (stopping early on EOS or a full
/// context, mirroring `Generator::generate`'s own stop conditions). The
/// profiler counters are snapshotted and reset between the two phases so
/// each table reflects only its own phase. Prints both tables to stdout via
/// `cli::profile::render`; never streams generated text (this is a
/// measurement run, not a user-facing generation).
fn run_profile(
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    max_seq_len: usize,
    threads: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let max_seq_len = clamp_max_seq_len(model, max_seq_len)?;
    let mut engine = Engine::load(model, max_seq_len)?;
    if threads != 0 {
        engine.set_threads(threads as usize);
    }
    // Distinct (profiled) cache entry — see `CompileOptions.profile`.
    engine.set_profile(true);
    // M4b.12: dispatch-split recording (no-op unless built with
    // --features pool-profile).
    inferno_pool::set_pool_profiling(true);
    let mut backend = engine.compiled_backend()?;

    let desc = inferno_formats::load_desc(model)?;
    let spec = desc.tokenizer.as_ref().ok_or(RuntimeError::NoTokenizer)?;
    let tokenizer = inferno_runtime::tokenizer_for(spec)?;
    let prompt_ids = tokenizer.encode(prompt, tokenizer.default_add_bos())?;
    if prompt_ids.is_empty() || prompt_ids.len() >= max_seq_len {
        return Err(Box::new(RuntimeError::PromptTooLong {
            got: prompt_ids.len(),
            max: max_seq_len,
        }));
    }
    let eos = tokenizer.eos();

    backend.reset();

    let t0 = Instant::now();
    let mut last = backend.forward(&prompt_ids)?;
    let prefill_secs = t0.elapsed().as_secs_f64();

    let slots = backend.profile_slots().to_vec();
    let prefill_counts = backend.profile_snapshot().unwrap_or_default();
    backend.profile_reset();
    inferno_pool::pool_prof_reset();

    let mut sampler = Greedy;
    let mut generated = 0usize;
    let t1 = Instant::now();
    for step in 0..max_tokens {
        let next = sampler.sample(&last);
        if Some(next) == eos {
            break;
        }
        generated += 1;
        let seq_len = prompt_ids.len() + step;
        if seq_len + 1 > max_seq_len {
            break; // context full
        }
        last = backend.forward(&[next])?;
    }
    let decode_secs = t1.elapsed().as_secs_f64();
    let decode_counts = backend.profile_snapshot().unwrap_or_default();

    // Per-slot weight bytes for ONE forward invocation, scaled by each
    // phase's per-token invocation count (see `Engine::profile_matmul_bytes`
    // doc comment for the approximation this makes).
    let per_invocation_bytes = engine.profile_matmul_bytes(&slots)?;
    let prefill_bytes: Vec<u64> = per_invocation_bytes
        .iter()
        .map(|b| b * prompt_ids.len() as u64)
        .collect();
    let decode_bytes: Vec<u64> = per_invocation_bytes
        .iter()
        .map(|b| b * generated as u64)
        .collect();

    print!(
        "{}",
        crate::profile::render(
            "prefill",
            &slots,
            &prefill_counts,
            &prefill_bytes,
            prefill_secs
        )
    );
    print!(
        "{}",
        crate::profile::render("decode", &slots, &decode_counts, &decode_bytes, decode_secs)
    );
    // M4b.12 dispatch-split section (only prints on a pool-profile build).
    if let Some(snap) = inferno_pool::pool_prof_snapshot()
        && snap.calls > 0
    {
        let attn_cyc = slots
            .iter()
            .position(|s| s == "attention")
            .map(|i| decode_counts[i])
            .unwrap_or(0);
        print!("{}", crate::profile::render_pool(&snap, attn_cyc));
    }
    Ok(())
}

/// Build a `Generator` driven by a `CompiledBackend` (the default, non
/// `--interp` path).
///
/// `max_seq_len` MUST be the exact same value used to build the `Engine`
/// (which sizes the `CompiledBackend`'s KV cache) and the `Generator`
/// (which uses it for its own context-full bookkeeping) — a mismatch would
/// desync the decode loop's stop condition from the backend's real KV
/// capacity. We clamp it ONCE here (mirroring `Generator::load`'s own
/// clamp against the model's context length) and hand the identical clamped
/// value to both `Engine::load` and `Generator::load_with_backend` below, so
/// there is exactly one number in play.
pub(crate) fn load_compiled(
    model: &Path,
    max_seq_len: usize,
    threads: u64,
) -> Result<Generator, Box<dyn std::error::Error>> {
    let max_seq_len = clamp_max_seq_len(model, max_seq_len)?;
    let mut engine = Engine::load(model, max_seq_len)?;
    if threads != 0 {
        engine.set_threads(threads as usize);
    }
    let backend = engine.compiled_backend()?;
    let generator = Generator::load_with_backend(model, max_seq_len, Box::new(backend))?;
    Ok(generator)
}

/// Clamp a requested `max_seq_len` to the model's declared context length
/// (mirroring `Generator::load`'s own clamp), so the compiled `Engine` and the
/// `Generator` are keyed on the SAME effective sequence length. A model with
/// `context_length == 0` (unknown) keeps the requested value. Shared by the
/// `inferno run` compiled path and `inferno diff-compiled` so both compile and
/// key the identical artifact for a given model + requested `--max-seq-len`.
pub(crate) fn clamp_max_seq_len(
    model: &Path,
    requested: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    let desc = inferno_formats::load_desc(model)?;
    let ctx = desc.hyperparams.context_length as usize;
    Ok(if ctx > 0 {
        requested.min(ctx)
    } else {
        requested
    })
}
