# Threat model — inferno v1

**Assets:** the user's machine (inference runs locally with the user's
privileges) and the integrity of inference results.

**Adversary:** whoever produced a model file the user downloads. Model files
(GGUF, safetensors, config.json) are **untrusted input** — the primary
boundary. A malicious file must at worst produce a clean error, never memory
corruption, panic-DoS from a tiny crafted input, or resource exhaustion from
lying headers.

**Controls at this boundary** (`crates/inferno-formats`):
- `#![forbid(unsafe_code)]` in the parsing crate.
- Bounded reads everywhere: explicit limits (`src/limits.rs`) on string
  lengths, tensor/KV counts, array sizes and nesting depth, header sizes; no
  allocation sized by an unvalidated length (arrays parse element-by-element
  so a lying count hits EOF cheaply).
- Checked arithmetic on all attacker-controlled sizes and offsets;
  offset/length consistency validated (safetensors spans vs dtype size,
  GGUF offset alignment).
- Continuous fuzzing of both parsers (nightly CI, `fuzz/`): any panic is a
  bug by definition.

**Second boundary: the artifact cache** (`~/.cache/inferno/`, from M3).
Compiled artifacts are native code loaded via `dlopen` — running a cached
artifact is running code from that directory. Artifacts are looked up and
verified by content hash (model × target × inferno version); a hash mismatch
means recompile, never load. The cache directory has user-only permissions.

**Explicitly out of scope (v1):**
- Sandboxing generated code against a hostile *local* user (same trust
  domain as the inferno process itself).
- Network boundaries — v1 has no server and makes no network calls.
- Tokenizer-content attacks on downstream systems (prompt/output content is
  the embedding application's concern).

**Supply chain:** `cargo audit` + `cargo deny` gate CI and run on a nightly
schedule; `Cargo.lock` committed; `gitleaks` on pre-commit and CI.

Revisit when: the HTTP server lands (v2), AOT artifact *distribution* lands
(signing story needed), or any parser gains `unsafe`.
