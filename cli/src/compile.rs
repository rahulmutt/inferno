use std::path::Path;
use std::process::ExitCode;

use inferno_core::Engine;

/// `inferno compile`: force a compile (or verify + reuse a cached compile) of
/// `model` for the host target at `max_seq_len`, then print the cache
/// directory the resulting `model.so`/`weights.bin`/`meta.json` live in.
pub fn compile(model: &Path, max_seq_len: usize) -> ExitCode {
    let inner = || -> inferno_core::Result<std::path::PathBuf> {
        let engine = Engine::load(model, max_seq_len)?;
        // Forces the compile-or-cache: builds (or verifies a cached) Artifact
        // and wraps it in a CompiledBackend, which is otherwise discarded —
        // `inferno compile` only cares about the cache directory landing on
        // disk with a verified artifact in it.
        let _ = engine.compiled_backend()?;
        engine.cache_dir()
    };
    match inner() {
        Ok(dir) => {
            println!("{}", dir.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
