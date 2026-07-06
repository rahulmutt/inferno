//! The generation loop: tokenize → prefill → [sample → decode]* → stream.

use std::ops::ControlFlow;
use std::path::Path;
use std::time::Instant;

use inferno_formats::{ModelDesc, load_desc};
use inferno_graph::{Graph, Interpreter, KvCache, Tensor, build_graph};

use crate::backend::{Backend, InterpBackend};
use crate::sampler::Sampler;
use crate::tokenizer::{Tokenizer, tokenizer_for};
use crate::{Result, RuntimeError};

/// Buffers streamed token bytes, emitting only complete UTF-8 sequences.
/// Invalid bytes (impossible from a well-formed vocab, cheap to guard)
/// become U+FFFD.
#[derive(Default)]
pub(crate) struct Utf8Buffer {
    pending: Vec<u8>,
}

impl Utf8Buffer {
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Vec<u8> {
        self.pending.extend_from_slice(bytes);
        let mut out = Vec::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(_) => {
                    out.append(&mut self.pending);
                    return out;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    out.extend_from_slice(&self.pending[..valid]);
                    match e.error_len() {
                        None => {
                            // Incomplete tail — keep it pending.
                            self.pending.drain(..valid);
                            return out;
                        }
                        Some(bad) => {
                            out.extend_from_slice("\u{FFFD}".as_bytes());
                            self.pending.drain(..valid + bad);
                        }
                    }
                }
            }
        }
    }
}

pub struct GenStats {
    pub prompt_tokens: usize,
    pub generated: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
}

pub struct Generator {
    desc: ModelDesc,
    graph: Graph,
    backend: Box<dyn Backend>,
    tokenizer: Box<dyn Tokenizer>,
    max_seq_len: usize,
}

impl Generator {
    pub fn load(model: &Path, max_seq_len: usize) -> Result<Generator> {
        let desc = load_desc(model)?;
        let graph = build_graph(&desc)?;
        let spec = desc.tokenizer.as_ref().ok_or(RuntimeError::NoTokenizer)?;
        let tokenizer = tokenizer_for(spec)?;
        let ctx = desc.hyperparams.context_length as usize;
        let max_seq_len = if ctx > 0 {
            max_seq_len.min(ctx)
        } else {
            max_seq_len
        };
        let backend = Box::new(InterpBackend::new(
            desc.clone(),
            graph.clone(),
            max_seq_len,
        )?);
        Ok(Generator {
            desc,
            graph,
            backend,
            tokenizer,
            max_seq_len,
        })
    }

    /// Like [`Generator::load`], but drives generation through a
    /// caller-supplied [`Backend`] instead of the default `InterpBackend`
    /// (e.g. the M3 CLI injecting a `CompiledBackend`). `full_logits`
    /// (teacher forcing) always uses its own local interpreter regardless
    /// of which backend drives `generate`.
    pub fn load_with_backend(
        model: &Path,
        max_seq_len: usize,
        backend: Box<dyn Backend>,
    ) -> Result<Generator> {
        let desc = load_desc(model)?;
        let graph = build_graph(&desc)?;
        let spec = desc.tokenizer.as_ref().ok_or(RuntimeError::NoTokenizer)?;
        let tokenizer = tokenizer_for(spec)?;
        let ctx = desc.hyperparams.context_length as usize;
        let max_seq_len = if ctx > 0 {
            max_seq_len.min(ctx)
        } else {
            max_seq_len
        };
        Ok(Generator {
            desc,
            graph,
            backend,
            tokenizer,
            max_seq_len,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, self.tokenizer.default_add_bos())
    }

    pub fn vocab_size(&self) -> usize {
        self.desc.hyperparams.vocab_size as usize
    }

    /// Single full-sequence pass returning logits at every position
    /// (teacher forcing / diff harness). Always uses a local interpreter —
    /// a last-token-only backend (e.g. the compiled path) cannot produce
    /// the all-position logits teacher forcing needs.
    pub fn full_logits(&mut self, tokens: &[u32]) -> Result<Tensor> {
        let mut kv = KvCache::new(&self.graph, self.max_seq_len)?;
        if tokens.len() > self.max_seq_len {
            return Err(RuntimeError::PromptTooLong {
                got: tokens.len(),
                max: self.max_seq_len,
            });
        }
        let mut interp = Interpreter::new();
        Ok(interp.run(&self.desc, &self.graph, tokens, &mut kv)?)
    }

    /// Runs generation, streaming decoded bytes to `on_bytes` as they become
    /// available. `on_bytes` returns `ControlFlow::Break(())` to signal that
    /// the consumer is gone (e.g. a broken stdout pipe) — the decode loop
    /// stops immediately rather than grinding through the remaining
    /// `max_tokens`. Returning `ControlFlow::Continue(())` keeps generating.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampler: &mut dyn Sampler,
        on_bytes: &mut dyn FnMut(&[u8]) -> ControlFlow<()>,
    ) -> Result<(Vec<u32>, GenStats)> {
        let prompt_ids = self.encode(prompt)?;
        if prompt_ids.is_empty() || prompt_ids.len() >= self.max_seq_len {
            return Err(RuntimeError::PromptTooLong {
                got: prompt_ids.len(),
                max: self.max_seq_len,
            });
        }
        for &t in &prompt_ids {
            sampler.accept(t);
        }
        self.backend.reset();
        let eos = self.tokenizer.eos();
        let mut buf = Utf8Buffer::default();
        let mut out_ids = Vec::new();

        let t0 = Instant::now();
        let mut last = self.backend.forward(&prompt_ids)?;
        let prefill_secs = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        for step in 0..max_tokens {
            let next = sampler.sample(&last);
            if Some(next) == eos {
                break;
            }
            out_ids.push(next);
            sampler.accept(next);
            let chunk = buf.push(&self.tokenizer.decode_token(next));
            if !chunk.is_empty() && on_bytes(&chunk).is_break() {
                break; // consumer signaled stop (e.g. broken pipe)
            }
            // Mirrors the backend's own KV length (prompt + steps decoded so
            // far) so this matches the interpreter's
            // `kv.len() + 1 > max_seq_len` check exactly, without the
            // Generator reaching into the backend.
            let seq_len = prompt_ids.len() + step;
            if seq_len + 1 > self.max_seq_len {
                break; // context full
            }
            last = self.backend.forward(&[next])?;
        }
        let stats = GenStats {
            prompt_tokens: prompt_ids.len(),
            generated: out_ids.len(),
            prefill_secs,
            decode_secs: t1.elapsed().as_secs_f64(),
        };
        Ok((out_ids, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::Greedy;

    #[test]
    fn utf8_buffer_holds_split_codepoints() {
        let mut b = Utf8Buffer::default();
        let euro = "€".as_bytes(); // 3 bytes
        assert_eq!(b.push(&euro[..1]), b"");
        assert_eq!(b.push(&euro[1..2]), b"");
        assert_eq!(b.push(&euro[2..]), "€".as_bytes());
        assert_eq!(b.push(b"ab"), b"ab");
    }

    #[test]
    fn utf8_buffer_replaces_invalid_bytes() {
        let mut b = Utf8Buffer::default();
        // 0xFF can never start a UTF-8 sequence → replacement char.
        assert_eq!(b.push(&[0xFF, b'a']), "\u{FFFD}a".as_bytes());
    }

    fn fixture(p: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../inferno-formats/tests/fixtures")
            .join(p)
    }

    /// A consumer that signals stop (e.g. a closed stdout pipe) must halt
    /// the decode loop immediately rather than grinding through the
    /// remaining `max_tokens` — the bug this contract change fixes.
    #[test]
    fn on_bytes_break_stops_generation_early() {
        let mut g = Generator::load(&fixture("tiny.gguf"), 64).unwrap();
        let mut calls = 0usize;
        let (ids, stats) = g
            .generate("the", 50, &mut Greedy, &mut |_| {
                calls += 1;
                if calls == 2 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .unwrap();
        assert_eq!(calls, 2, "loop must stop right after the break signal");
        assert!(
            ids.len() < 50,
            "generation should halt long before max_tokens: got {} ids",
            ids.len()
        );
        assert_eq!(stats.generated, ids.len());
    }

    /// The generator must feed every prompt token and every sampled token to
    /// `Sampler::accept` — that is how repeat penalty (M4a) sees context.
    #[test]
    fn generator_accepts_prompt_and_sampled_tokens() {
        struct Recording {
            inner: Greedy,
            accepted: Vec<u32>,
        }
        impl crate::sampler::Sampler for Recording {
            fn sample(&mut self, logits: &[f32]) -> u32 {
                self.inner.sample(logits)
            }
            fn accept(&mut self, token: u32) {
                self.accepted.push(token);
            }
        }

        let mut g = Generator::load(&fixture("tiny.gguf"), 64).unwrap();
        let prompt_ids = g.encode("the").unwrap();
        let mut s = Recording {
            inner: Greedy,
            accepted: Vec::new(),
        };
        let (out_ids, _) = g
            .generate("the", 4, &mut s, &mut |_| ControlFlow::Continue(()))
            .unwrap();
        let mut want = prompt_ids;
        want.extend_from_slice(&out_ids);
        assert_eq!(s.accepted, want);
    }

    fn sampled_ids(seed: u64) -> Vec<u32> {
        use crate::sampler::{ChainSampler, SamplerConfig};
        let mut g = Generator::load(&fixture("tiny.gguf"), 64).unwrap();
        let mut s = ChainSampler::new(SamplerConfig {
            temperature: 5.0, // near-uniform: forces real draws
            seed,
            ..Default::default()
        });
        g.generate("the", 8, &mut s, &mut |_| ControlFlow::Continue(()))
            .unwrap()
            .0
    }

    /// Blocking-tier determinism gate from the M4a spec: same seed → same
    /// token sequence; different seeds diverge at temperature > 0.
    #[test]
    fn same_seed_same_tokens_different_seed_diverges() {
        assert_eq!(sampled_ids(7), sampled_ids(7));
        // At temperature 5 over the whole vocab, 8 identical draws across
        // two seeds is astronomically unlikely; a collision here means the
        // seed is being ignored.
        assert_ne!(sampled_ids(1), sampled_ids(2));
    }
}
