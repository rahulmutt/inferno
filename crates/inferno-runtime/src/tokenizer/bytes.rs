//! GPT-2 byte↔unicode mapping and token-string → raw-bytes decoding, shared
//! by the native BPE/SPM tokenizers and the HF wrapper.

use std::collections::HashMap;
use std::sync::OnceLock;

pub(crate) fn byte_to_unicode() -> &'static [char; 256] {
    static TABLE: OnceLock<[char; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let printable =
            |b: u8| (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || b >= 0xAE;
        let mut table = ['\0'; 256];
        let mut n = 0u32;
        for b in 0u16..=255 {
            let b8 = b as u8;
            table[b as usize] = if printable(b8) {
                char::from_u32(u32::from(b)).unwrap()
            } else {
                n += 1;
                char::from_u32(255 + n).unwrap()
            };
        }
        table
    })
}

pub(crate) fn unicode_to_byte() -> &'static HashMap<char, u8> {
    static MAP: OnceLock<HashMap<char, u8>> = OnceLock::new();
    MAP.get_or_init(|| {
        byte_to_unicode()
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect()
    })
}

/// GPT-2-form BPE token string → raw bytes. None if any char is outside the
/// byte-unicode alphabet (e.g. a control/added token that slipped through —
/// callers filter specials by token type before decoding).
pub(crate) fn bpe_token_to_bytes(token: &str) -> Option<Vec<u8>> {
    let map = unicode_to_byte();
    token.chars().map(|c| map.get(&c).copied()).collect()
}

/// SPM token string → raw bytes: ▁ (U+2581) → space, <0xNN> byte-fallback
/// tokens → their byte, everything else passes through as UTF-8.
pub(crate) fn spm_token_to_bytes(token: &str) -> Vec<u8> {
    if token.len() == 6
        && token.starts_with("<0x")
        && token.ends_with('>')
        && let Ok(b) = u8::from_str_radix(&token[3..5], 16)
    {
        return vec![b];
    }
    token.replace('\u{2581}', " ").into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_table_roundtrips_all_bytes() {
        let enc = byte_to_unicode();
        let dec = unicode_to_byte();
        for b in 0u16..=255 {
            assert_eq!(dec[&enc[b as usize]], b as u8);
        }
        assert_eq!(enc[b'a' as usize], 'a'); // printables map to themselves
        assert_eq!(enc[b' ' as usize], '\u{120}'); // space → Ġ
    }

    #[test]
    fn bpe_token_to_bytes_decodes_gpt2_form() {
        assert_eq!(bpe_token_to_bytes("Ġhello").unwrap(), b" hello");
        assert_eq!(bpe_token_to_bytes("the").unwrap(), b"the");
        // "<|bos|>" is all printable ASCII, so it decodes to its own bytes —
        // special tokens are filtered by token TYPE before decoding (Tasks
        // 11/13), never detected here.
        assert_eq!(bpe_token_to_bytes("<|bos|>").unwrap(), b"<|bos|>");
    }

    #[test]
    fn spm_token_to_bytes_handles_space_and_byte_fallback() {
        assert_eq!(spm_token_to_bytes("\u{2581}the"), b" the");
        assert_eq!(spm_token_to_bytes("<0x0A>"), b"\n");
        assert_eq!(spm_token_to_bytes("x"), b"x");
    }
}
