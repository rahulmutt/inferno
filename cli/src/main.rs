mod bench;
mod compile;
mod diff;
mod inspect;
mod run;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "inferno", about = "CPU-first LLM inference engine", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Show a model file's architecture, hyperparameters, and tensors.
    Inspect {
        /// Path to a .gguf file, an MLX directory, or a .safetensors file.
        model: PathBuf,
        /// How many tensors to list (0 = none).
        #[arg(long, default_value_t = 10)]
        tensors: usize,
    },
    /// Generate text from a prompt. Runs the compiled path by default
    /// (`inferno compile`'s cache, compiling on first use); `--interp` forces
    /// the M1 reference interpreter instead (slow by design; useful as a
    /// cross-check against the compiled path).
    Run {
        /// Path to a .gguf file, an MLX directory, or a .safetensors file.
        model: PathBuf,
        /// Prompt text (raw completion; no chat template).
        #[arg(long, short)]
        prompt: String,
        /// Maximum tokens to generate.
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// KV-cache capacity (clamped to the model's context length).
        #[arg(long, default_value_t = 4096)]
        max_seq_len: usize,
        /// Use the M1 scalar interpreter instead of the compiled path.
        #[arg(long)]
        interp: bool,
        /// Sampling temperature; 0 = greedy (the default).
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// Keep only the k highest-logit tokens; 0 = disabled.
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        /// Nucleus sampling mass in (0, 1]; 1.0 = disabled.
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,
        /// Drop tokens below min-p × max-probability; 0 = disabled.
        #[arg(long, default_value_t = 0.0)]
        min_p: f32,
        /// Penalty for recently seen tokens; 1.0 = disabled.
        #[arg(long, default_value_t = 1.0)]
        repeat_penalty: f32,
        /// Repeat-penalty window length.
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
        /// RNG seed for sampling.
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Compile a model for the host target and print the cache directory
    /// (`model.so`/`weights.bin`/`meta.json`) the artifact lands in. Reuses a
    /// verified cached compile if one already exists for this
    /// model/target/`max_seq_len`.
    Compile {
        /// Path to a .gguf file, an MLX directory, or a .safetensors file.
        model: PathBuf,
        /// KV-cache capacity to compile for (part of the cache key).
        #[arg(long, default_value_t = 4096)]
        max_seq_len: usize,
    },
    /// Teacher-forced differential vs an external reference (nightly harness).
    #[command(hide = true)]
    Diff {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt_file: PathBuf,
        #[arg(long)]
        tokens_file: PathBuf,
    },
    /// Compiled-vs-interpreter last-token-logit differential over a
    /// teacher-forced sequence (M3 compiled-path gate).
    #[command(hide = true)]
    DiffCompiled {
        model: PathBuf,
        #[arg(long, short)]
        prompt: String,
        /// How many tokens the interpreter greedily generates to build the
        /// teacher-forced sequence the compiled backend is checked against.
        #[arg(long, default_value_t = 8)]
        max_tokens: usize,
        #[arg(long, default_value_t = 4096)]
        max_seq_len: usize,
    },
    /// Decode-throughput speedup gate: generates `max_tokens` with the
    /// compiled backend and with the interpreter and asserts compiled tok/s
    /// clears a conservative multiple of interpreter tok/s (M3 nightly
    /// "faster than the interpreter" gate; not part of the blocking PR
    /// tier — see `mise run speedup`).
    #[command(hide = true)]
    BenchCompiled {
        model: PathBuf,
        #[arg(long, short)]
        prompt: String,
        #[arg(long, default_value_t = 48)]
        max_tokens: usize,
        #[arg(long, default_value_t = 4096)]
        max_seq_len: usize,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect { model, tensors } => match inferno_formats::load_desc(&model) {
            Ok(desc) => {
                print!("{}", inspect::render(&desc, tensors));
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Command::Run {
            model,
            prompt,
            max_tokens,
            max_seq_len,
            interp,
            temperature,
            top_k,
            top_p,
            min_p,
            repeat_penalty,
            repeat_last_n,
            seed,
        } => run::run(
            &model,
            &prompt,
            max_tokens,
            max_seq_len,
            interp,
            inferno_runtime::SamplerConfig {
                temperature,
                top_k,
                top_p,
                min_p,
                repeat_penalty,
                repeat_last_n,
                seed,
            },
        ),
        Command::Compile { model, max_seq_len } => compile::compile(&model, max_seq_len),
        Command::Diff {
            model,
            prompt_file,
            tokens_file,
        } => diff::diff(&model, &prompt_file, &tokens_file),
        Command::DiffCompiled {
            model,
            prompt,
            max_tokens,
            max_seq_len,
        } => diff::diff_compiled(&model, &prompt, max_tokens, max_seq_len),
        Command::BenchCompiled {
            model,
            prompt,
            max_tokens,
            max_seq_len,
        } => bench::bench_compiled(&model, &prompt, max_tokens, max_seq_len),
    }
}
