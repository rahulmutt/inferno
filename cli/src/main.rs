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
    /// Generate text from a prompt (M1: reference interpreter — slow by
    /// design; the compiler arrives in M3).
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
        } => run::run(&model, &prompt, max_tokens, max_seq_len),
    }
}
