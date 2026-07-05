use std::path::Path;
use std::process::ExitCode;

use inferno_runtime::{Generator, teacher_forced};
use serde::Deserialize;

#[derive(Deserialize)]
struct TokensFile {
    prompt_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
}

pub fn diff(model: &Path, prompt_file: &Path, tokens_file: &Path) -> ExitCode {
    let inner = || -> Result<bool, Box<dyn std::error::Error>> {
        let prompt = std::fs::read_to_string(prompt_file)?;
        let tf: TokensFile = serde_json::from_str(&std::fs::read_to_string(tokens_file)?)?;
        let mut generator = Generator::load(model, 4096)?;

        // Gate 0: our tokenization must match the reference's exactly, or
        // every later position is comparing different sequences.
        let ours = generator.encode(&prompt)?;
        if ours != tf.prompt_tokens {
            let first = ours
                .iter()
                .zip(&tf.prompt_tokens)
                .position(|(a, b)| a != b)
                .unwrap_or(ours.len().min(tf.prompt_tokens.len()));
            eprintln!(
                "TOKENIZATION MISMATCH at prompt position {first}:\n  ours:  {ours:?}\n  llama: {:?}",
                tf.prompt_tokens
            );
            return Ok(false);
        }
        println!("prompt tokenization matches ({} tokens)", ours.len());

        let out = teacher_forced(&mut generator, &tf.prompt_tokens, &tf.generated_tokens)?;
        println!(
            "teacher-forced: {} checked, {} matched, {} ties, min top-2 gap {:.4}",
            out.checked, out.matched, out.ties, out.min_gap
        );
        for m in &out.mismatches {
            eprintln!(
                "MISMATCH at generated position {}: expected {}, got {} (gap {:.4})\n  our top-5: {:?}",
                m.position, m.expected, m.got, m.gap, m.top
            );
        }
        Ok(out.passed())
    };
    match inner() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
