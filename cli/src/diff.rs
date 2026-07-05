use std::ops::ControlFlow;
use std::path::Path;
use std::process::ExitCode;

use inferno_core::Engine;
use inferno_graph::tolerance::LOGIT_TIE_EPSILON;
use inferno_runtime::{Backend, Generator, Greedy, teacher_forced};
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

/// Compiled-vs-interpreter last-token-logit differential over a
/// teacher-forced sequence (M3 compiled-path gate, `inferno diff-compiled`).
///
/// The interpreter greedily generates a deterministic continuation of
/// `prompt` — that continuation IS the teacher-forced reference sequence, no
/// external tokens file needed. The compiled backend then replays
/// prompt+continuation through the exact `prefill`/`decode_step` path
/// `inferno run` (compiled, default) uses, one `forward` per new position,
/// comparing its per-position next-token argmax against the interpreter's
/// own choice at that position. A mismatch is tolerated as a "tie" — not a
/// failure — when the INTERPRETER's own top-2 logit gap at that position is
/// under `LOGIT_TIE_EPSILON`: greedy argmax of near-identical logits (Task
/// 12/14 already bound compiled-vs-interpreter |Δlogit|) can legitimately
/// pick either token on a near-tie.
pub fn diff_compiled(
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    max_seq_len: usize,
) -> ExitCode {
    let inner = || -> Result<bool, Box<dyn std::error::Error>> {
        let mut generator = Generator::load(model, max_seq_len)?;
        let prompt_tokens = generator.encode(prompt)?;
        if prompt_tokens.is_empty() {
            return Err(Box::new(inferno_runtime::RuntimeError::Tokenizer(
                "prompt tokenizes to no tokens".to_string(),
            )));
        }

        let (generated, _) = generator.generate(prompt, max_tokens, &mut Greedy, &mut |_| {
            ControlFlow::Continue(())
        })?;
        let full: Vec<u32> = prompt_tokens.iter().chain(&generated).copied().collect();
        if full.len() <= prompt_tokens.len() {
            println!("interpreter generated 0 tokens (immediate EOS) — nothing to compare");
            return Ok(true);
        }

        // All-position interpreter logits: row r predicts token r+1.
        let logits = generator.full_logits(&full)?;
        let vocab = generator.vocab_size();

        let engine = Engine::load(model, max_seq_len)?;
        let mut backend = engine.compiled_backend()?;

        let mut checked = 0usize;
        let mut matched = 0usize;
        let mut ties = 0usize;
        let mut min_gap = f32::INFINITY;
        let mut mismatches: Vec<(usize, u32, u32, f32)> = Vec::new();

        // First forward = prefill over the prompt; `last` predicts full[prompt_tokens.len()].
        let mut last = backend.forward(&prompt_tokens)?;
        for i in prompt_tokens.len()..full.len() {
            let expected = full[i];
            let interp_row = &logits.data[(i - 1) * vocab..i * vocab];
            let (top0, top1) = top2(interp_row);
            let gap = top0.1 - top1.1;
            let got = argmax(&last);

            checked += 1;
            min_gap = min_gap.min(gap);
            if got == expected {
                matched += 1;
            } else if gap < LOGIT_TIE_EPSILON {
                ties += 1;
            } else {
                mismatches.push((i, expected, got, gap));
            }

            if i + 1 < full.len() {
                last = backend.forward(&[expected])?;
            }
        }

        println!(
            "compiled-vs-interp: {checked} checked, {matched} matched, {ties} ties, min top-2 gap {min_gap:.4}"
        );
        for (pos, expected, got, gap) in &mismatches {
            eprintln!("MISMATCH at position {pos}: expected {expected}, got {got} (gap {gap:.4})");
        }
        Ok(mismatches.is_empty())
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

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// The top-2 `(index, value)` pairs of `row`, highest first.
fn top2(row: &[f32]) -> ((u32, f32), (u32, f32)) {
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    idx.sort_by(|a, b| row[*b as usize].total_cmp(&row[*a as usize]));
    let a = idx[0];
    let b = idx.get(1).copied().unwrap_or(a);
    ((a, row[a as usize]), (b, row[b as usize]))
}
