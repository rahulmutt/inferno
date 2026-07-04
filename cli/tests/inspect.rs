use assert_cmd::Command;

fn fixture(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../crates/inferno-formats/tests/fixtures")
        .join(rel)
}

#[test]
fn inspect_gguf_snapshot() {
    let out = Command::cargo_bin("inferno")
        .unwrap()
        .args([
            "inspect",
            fixture("tiny.gguf").to_str().unwrap(),
            "--tensors",
            "3",
        ])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    insta::assert_snapshot!("inspect_gguf", stdout);
}

#[test]
fn inspect_mlx_dir() {
    Command::cargo_bin("inferno")
        .unwrap()
        .args(["inspect", fixture("mlx").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicates::str::contains("architecture: llama"));
}

#[test]
fn inspect_missing_file_fails_cleanly() {
    Command::cargo_bin("inferno")
        .unwrap()
        .args(["inspect", "/nonexistent/model.gguf"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("error"));
}
