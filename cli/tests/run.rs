use assert_cmd::Command;
use predicates::prelude::*;

fn fixture(p: &str) -> String {
    format!(
        "{}/../crates/inferno-formats/tests/fixtures/{p}",
        env!("CARGO_MANIFEST_DIR")
    )
}

#[test]
fn run_streams_tokens_from_gguf_fixture() {
    Command::cargo_bin("inferno")
        .unwrap()
        .args([
            "run",
            &fixture("tiny.gguf"),
            "--prompt",
            "the",
            "--max-tokens",
            "4",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("decode:"));
}

#[test]
fn run_works_on_mlx_dir() {
    Command::cargo_bin("inferno")
        .unwrap()
        .args([
            "run",
            &fixture("mlx"),
            "--prompt",
            "the",
            "--max-tokens",
            "2",
        ])
        .assert()
        .success();
}

#[test]
fn run_reports_model_errors_cleanly() {
    Command::cargo_bin("inferno")
        .unwrap()
        .args(["run", "/nonexistent.gguf", "--prompt", "x"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("error:"));
}
