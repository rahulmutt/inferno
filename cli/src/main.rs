mod inspect;

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
    }
}
