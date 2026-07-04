#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = inferno_formats::safetensors::parse(&mut std::io::Cursor::new(data), 0);
});
