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

/// Task 16 gate: the compiled path (default `inferno run`) and the
/// interpreter path (`--interp`) must produce IDENTICAL greedy token text.
/// This runs the REAL `inferno` binary through the compiled `dlopen` path —
/// it is the end-to-end proof that `-rdynamic` + kernel retention +
/// `CompiledBackend` all work together (a resolution failure here surfaces as
/// an `undefined symbol: inferno_gemv_*` dlopen error, not a silent mismatch).
#[test]
fn compiled_and_interp_agree_on_greedy_tokens() {
    // Point both invocations at an isolated cache dir so this test doesn't
    // race other tests / prior runs over the default `~/.cache/inferno`.
    let cache = tempfile::tempdir().unwrap();

    let compiled = Command::cargo_bin("inferno")
        .unwrap()
        .env("XDG_CACHE_HOME", cache.path())
        .args([
            "run",
            &fixture("tiny.gguf"),
            "--prompt",
            "the",
            "--max-tokens",
            "8",
        ])
        .assert()
        .success();
    let compiled_stdout = String::from_utf8(compiled.get_output().stdout.clone()).unwrap();

    let interp = Command::cargo_bin("inferno")
        .unwrap()
        .env("XDG_CACHE_HOME", cache.path())
        .args([
            "run",
            &fixture("tiny.gguf"),
            "--prompt",
            "the",
            "--max-tokens",
            "8",
            "--interp",
        ])
        .assert()
        .success();
    let interp_stdout = String::from_utf8(interp.get_output().stdout.clone()).unwrap();

    assert_eq!(
        compiled_stdout, interp_stdout,
        "compiled and interpreter paths must produce identical greedy token text"
    );
}

#[test]
fn run_sampling_same_seed_is_reproducible() {
    let out = |seed: &str| {
        let a = Command::cargo_bin("inferno")
            .unwrap()
            .args([
                "run",
                &fixture("tiny.gguf"),
                "--interp",
                "--prompt",
                "the",
                "--max-tokens",
                "8",
                "--max-seq-len",
                "64",
                "--temperature",
                "5.0",
                "--seed",
                seed,
            ])
            .assert()
            .success();
        String::from_utf8(a.get_output().stdout.clone()).unwrap()
    };
    assert_eq!(out("7"), out("7"));
}

#[test]
fn run_rejects_invalid_sampling_flags() {
    Command::cargo_bin("inferno")
        .unwrap()
        .args([
            "run",
            &fixture("tiny.gguf"),
            "--interp",
            "--prompt",
            "the",
            "--top-p",
            "0",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("top-p"));
}
