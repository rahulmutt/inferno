#![no_main]
use libfuzzer_sys::fuzz_target;

// The parser must be total over arbitrary bytes: any panic is a finding.
fuzz_target!(|data: &[u8]| {
    let _ = inferno_formats::gguf::parse(&mut std::io::Cursor::new(data));
});
