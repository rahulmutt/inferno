//! Teacher-forced differential against an external reference (llama.cpp).
//! One full-sequence pass; per-position argmax comparison with a tie
//! tolerance computed from OUR top-2 logit gap — no reference-logit
//! extraction needed (spec §Nightly tier).

use inferno_graph::tolerance::LOGIT_TIE_EPSILON;

use crate::Result;
use crate::generate::Generator;

#[derive(Debug)]
pub struct Mismatch {
    pub position: usize,
    pub expected: u32,
    pub got: u32,
    pub gap: f32,
    pub top: Vec<(u32, f32)>,
}

#[derive(Debug)]
pub struct DiffOutcome {
    pub checked: usize,
    pub matched: usize,
    pub ties: usize,
    pub min_gap: f32,
    pub mismatches: Vec<Mismatch>,
}

impl DiffOutcome {
    pub fn passed(&self) -> bool {
        self.mismatches.is_empty()
    }
}

fn top_n(row: &[f32], n: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    idx.sort_by(|a, b| row[*b as usize].total_cmp(&row[*a as usize]));
    idx.into_iter()
        .take(n)
        .map(|i| (i, row[i as usize]))
        .collect()
}

pub fn teacher_forced(
    generator: &mut Generator,
    prompt_tokens: &[u32],
    forced: &[u32],
) -> Result<DiffOutcome> {
    if prompt_tokens.is_empty() {
        // Position 0 would have no predicting row (logits[p] predict p+1).
        return Err(crate::RuntimeError::Tokenizer(
            "teacher forcing needs a non-empty prompt".into(),
        ));
    }
    let full: Vec<u32> = prompt_tokens.iter().chain(forced).copied().collect();
    let logits = generator.full_logits(&full)?;
    let vocab = generator.vocab_size();
    let mut out = DiffOutcome {
        checked: forced.len(),
        matched: 0,
        ties: 0,
        min_gap: f32::INFINITY,
        mismatches: Vec::new(),
    };
    for (i, &expected) in forced.iter().enumerate() {
        // logits row at position p predict token p+1; forced[i] sits at
        // absolute position prompt.len()+i, predicted by row prompt.len()+i-1.
        let row_idx = prompt_tokens.len() + i - 1;
        let row = &logits.data[row_idx * vocab..(row_idx + 1) * vocab];
        let top = top_n(row, 5);
        let got = top[0].0;
        let gap = top[0].1 - top[1].1;
        out.min_gap = out.min_gap.min(gap);
        if got == expected {
            out.matched += 1;
        } else if gap < LOGIT_TIE_EPSILON {
            out.ties += 1;
        } else {
            out.mismatches.push(Mismatch {
                position: i,
                expected,
                got,
                gap,
                top,
            });
        }
    }
    Ok(out)
}
