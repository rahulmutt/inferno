//! Drive the devenv-pinned `llama-bench` and parse its `-o json` output.
//! The parser is strict on required fields (schema drift fails loudly,
//! citing the field) and tolerant of extra fields (serde's default).

use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LlamaBenchRow {
    pub build_commit: String,
    pub cpu_info: String,
    pub model_type: String,
    pub n_prompt: u64,
    pub n_gen: u64,
    pub n_threads: u64,
    /// Average tokens/sec for this test row.
    pub avg_ts: f64,
    pub stddev_ts: f64,
}

pub fn parse(json: &str) -> Result<Vec<LlamaBenchRow>, String> {
    serde_json::from_str(json).map_err(|e| {
        format!(
            "unparseable llama-bench JSON (schema drift vs the devenv-pinned \
             llama.cpp? see the M4a spec): {e}"
        )
    })
}

pub fn find_row(
    rows: &[LlamaBenchRow],
    n_prompt: u64,
    n_gen: u64,
    n_threads: u64,
) -> Option<&LlamaBenchRow> {
    rows.iter()
        .find(|r| r.n_prompt == n_prompt && r.n_gen == n_gen && r.n_threads == n_threads)
}

pub fn run_llama_bench(
    bin: &Path,
    model: &Path,
    pp: u64,
    tg: u64,
    threads: &[u64],
    reps: u64,
) -> Result<Vec<LlamaBenchRow>, String> {
    let t_list = threads
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let out = Command::new(bin)
        .arg("-m")
        .arg(model)
        .args(["-p", &pp.to_string(), "-n", &tg.to_string()])
        .args(["-t", &t_list, "-r", &reps.to_string(), "-o", "json"])
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                format!(
                    "llama-bench not found at `{}` — run inside `devenv shell` \
                     (it provides the pinned llama.cpp) or pass --llama-bench <path>",
                    bin.display()
                )
            } else {
                format!("failed to spawn llama-bench: {e}")
            }
        })?;
    if !out.status.success() {
        return Err(format!(
            "llama-bench exited with {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    parse(&String::from_utf8_lossy(&out.stdout))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_json() -> String {
        std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/llama-bench.json"
        ))
        .unwrap()
    }

    #[test]
    fn parses_golden_fixture() {
        let rows = parse(&fixture_json()).unwrap();
        assert_eq!(rows.len(), 2);
        let pp = find_row(&rows, 512, 0, 12).unwrap();
        assert_eq!(pp.avg_ts, 486.4);
        assert_eq!(pp.stddev_ts, 4.9);
        assert_eq!(pp.build_commit, "3ab8b3a9");
        let tg = find_row(&rows, 0, 128, 12).unwrap();
        assert_eq!(tg.avg_ts, 84.0);
        assert!(find_row(&rows, 512, 0, 1).is_none());
    }

    /// Schema drift (a required field vanishing) must fail loudly with the
    /// field name, not produce a half-report.
    #[test]
    fn missing_required_field_is_a_loud_error() {
        let broken = fixture_json().replace("\"avg_ts\"", "\"renamed_ts\"");
        let err = parse(&broken).unwrap_err();
        assert!(err.contains("avg_ts"), "error should name the field: {err}");
    }

    #[test]
    fn non_json_input_is_an_error() {
        assert!(parse("ggml_init: using CPU backend\n").is_err());
    }
}
