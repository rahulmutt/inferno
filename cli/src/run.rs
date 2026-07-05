use std::io::Write;
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
    let result = generator.generate(prompt, max_tokens, &mut Greedy, &mut |bytes| {
        let _ = stdout.write_all(bytes);
        let _ = stdout.flush();
    });
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
