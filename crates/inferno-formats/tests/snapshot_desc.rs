use std::path::Path;

fn fixture(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(rel)
}

#[test]
fn gguf_desc_snapshot() {
    let desc = inferno_formats::load_desc(&fixture("tiny.gguf")).unwrap();
    insta::assert_yaml_snapshot!("tiny_gguf_desc", desc);
}

#[test]
fn mlx_desc_snapshot() {
    let desc = inferno_formats::load_desc(&fixture("mlx")).unwrap();
    insta::assert_yaml_snapshot!("tiny_mlx_desc", desc);
}

#[test]
fn gguf_weight_file_and_offset_recorded() {
    let path = fixture("tiny.gguf");
    let desc = inferno_formats::load_desc(&path).unwrap();
    assert_eq!(desc.weight_files, vec![path]);
    assert_eq!(desc.data_section_offsets.len(), 1);
}

#[test]
fn unknown_format_is_clear_error() {
    let err = inferno_formats::load_desc(Path::new("Cargo.toml")).unwrap_err();
    assert!(matches!(
        err,
        inferno_formats::FormatError::UnknownFormat(_)
    ));
}
