use std::io::Write;
use std::ops::ControlFlow;
use std::path::Path;
use std::process::ExitCode;

use inferno_core::Engine;
use inferno_runtime::{Generator, Greedy};

pub fn run(
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    max_seq_len: usize,
    interp: bool,
) -> ExitCode {
    let generator = if interp {
        Generator::load(model, max_seq_len).map_err(|e| e.to_string())
    } else {
        load_compiled(model, max_seq_len).map_err(|e| e.to_string())
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
    let result = generator.generate(prompt, max_tokens, &mut Greedy, &mut |bytes| match stdout
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
) -> Result<Generator, Box<dyn std::error::Error>> {
    let desc = inferno_formats::load_desc(model)?;
    let ctx = desc.hyperparams.context_length as usize;
    let max_seq_len = if ctx > 0 {
        max_seq_len.min(ctx)
    } else {
        max_seq_len
    };
    let engine = Engine::load(model, max_seq_len)?;
    let backend = engine.compiled_backend()?;
    let generator = Generator::load_with_backend(model, max_seq_len, Box::new(backend))?;
    Ok(generator)
}
