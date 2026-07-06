use assert_cmd::Command;
use predicates::prelude::*;

fn fixture(p: &str) -> String {
    format!(
        "{}/../crates/inferno-formats/tests/fixtures/{p}",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// Validation must reject zero sizes before any measurement or llama-bench
/// lookup happens (so this test needs neither LLVM-compiled artifacts nor
/// a llama-bench binary).
#[test]
fn bench_rejects_zero_sizes() {
    for flag in ["--pp", "--tg", "--reps"] {
        Command::cargo_bin("inferno")
            .unwrap()
            .args(["bench", &fixture("tiny.gguf"), flag, "0"])
            .assert()
            .failure()
            .stderr(predicate::str::contains("must all be > 0"));
    }
}
