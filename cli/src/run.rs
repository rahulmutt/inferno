use std::io::Write;
use std::ops::ControlFlow;
use std::path::Path;
use std::process::ExitCode;

use inferno_runtime::{Generator, Greedy};

pub fn run(model: &Path, prompt: &str, max_tokens: usize, max_seq_len: usize) -> ExitCode {
    let mut generator = match Generator::load(model, max_seq_len) {
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
