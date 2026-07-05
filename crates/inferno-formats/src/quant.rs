//! Scalar quant codecs: the reference implementations of the v1 dtypes.
//! `dequant` follows the exact ggml block layouts (so real files decode);
//! `pack` is a *simple* min/max reference quantizer for fixtures and tests,
//! not ggml's error-minimizing quantizer. Both are the semantic ground truth
//! that M2 kernels are property-tested against.

use crate::{DType, FormatError, Result};

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15);
    let exp = u32::from((h >> 10) & 0x1F);
    let man = u32::from(h & 0x3FF);
    let bits = match (exp, man) {
        (0, 0) => sign << 31,
        (0, mut m) => {
            // Subnormal: renormalize into f32.
            let mut e: i32 = 113; // 127 - 15 + 1
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((m & 0x3FF) << 13)
        }
        (0x1F, 0) => (sign << 31) | 0x7F80_0000,
        (0x1F, m) => (sign << 31) | 0x7F80_0000 | (m << 13),
        (e, m) => (sign << 31) | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x7F_FFFF;
    if exp == 0xFF {
        // Inf/NaN; keep NaN-ness with a quiet bit.
        return sign | 0x7C00 | u16::from(man != 0) << 9;
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00; // overflow → inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // underflow → signed zero
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let midpoint = 1u32 << (shift - 1);
        let round = u32::from(rem > midpoint || (rem == midpoint && half & 1 == 1));
        return sign | (half + round) as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1FFF;
    let round = u32::from(rem > 0x1000 || (rem == 0x1000 && half & 1 == 1));
    sign | (half + round) as u16 // rounding carry correctly bumps the exponent
}

pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x0040; // quiet NaN
    }
    let half = bits >> 16;
    let rem = bits & 0xFFFF;
    let round = u32::from(rem > 0x8000 || (rem == 0x8000 && half & 1 == 1));
    (half + round) as u16
}

fn bad(detail: String) -> FormatError {
    FormatError::Malformed {
        context: "quant data",
        detail,
    }
}

/// ggml Q4_K scale/min extraction: 8 six-bit (scale, min) pairs in 12 bytes.
/// Public because `inferno-kernels` decodes scales at pack time (M2).
pub fn get_scale_min_k4(j: usize, s: &[u8]) -> (u8, u8) {
    if j < 4 {
        (s[j] & 63, s[j + 4] & 63)
    } else {
        (
            (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4),
            (s[j + 4] >> 4) | ((s[j] >> 6) << 4),
        )
    }
}

pub fn dequant(dtype: &DType, bytes: &[u8], n_elems: usize) -> Result<Vec<f32>> {
    let expected = dtype
        .byte_len(n_elems as u64)
        .ok_or_else(|| bad(format!("{dtype:?}: {n_elems} elements not representable")))?;
    if bytes.len() as u64 != expected {
        return Err(bad(format!(
            "{dtype:?}: got {} bytes, expected {expected} for {n_elems} elements",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(n_elems);
    match dtype {
        DType::F32 => {
            for c in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes(c.try_into().unwrap()));
            }
        }
        DType::F16 => {
            for c in bytes.chunks_exact(2) {
                out.push(f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())));
            }
        }
        DType::BF16 => {
            for c in bytes.chunks_exact(2) {
                out.push(bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap())));
            }
        }
        DType::Q8_0 => {
            // 34-byte block: f16 scale d, then 32 × i8. y = d * q.
            for block in bytes.chunks_exact(34) {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                for &q in &block[2..34] {
                    out.push(d * f32::from(q as i8));
                }
            }
        }
        DType::Q4_K => {
            // 144-byte super-block: f16 d, f16 dmin, 12 bytes of 6-bit
            // (scale, min) pairs, 128 bytes of 4-bit quants (256 elements).
            // y = d*sc*q - dmin*m, in chunks of 64 (32 low nibbles then 32 high).
            for block in bytes.chunks_exact(144) {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                let scales = &block[4..16];
                let qs = &block[16..144];
                let mut is = 0;
                let mut qoff = 0;
                for _ in 0..4 {
                    let (sc1, m1) = get_scale_min_k4(is, scales);
                    let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                    let (d1, min1) = (d * f32::from(sc1), dmin * f32::from(m1));
                    let (d2, min2) = (d * f32::from(sc2), dmin * f32::from(m2));
                    for l in 0..32 {
                        out.push(d1 * f32::from(qs[qoff + l] & 0xF) - min1);
                    }
                    for l in 0..32 {
                        out.push(d2 * f32::from(qs[qoff + l] >> 4) - min2);
                    }
                    qoff += 32;
                    is += 2;
                }
            }
        }
        DType::Unsupported(s) => return Err(bad(format!("unsupported dtype {s}"))),
    }
    Ok(out)
}

pub fn pack(dtype: &DType, values: &[f32]) -> Result<Vec<u8>> {
    let expected = dtype
        .byte_len(values.len() as u64)
        .ok_or_else(|| bad(format!("{dtype:?}: {} elements not packable", values.len())))?;
    let mut out = Vec::with_capacity(expected as usize);
    match dtype {
        DType::F32 => {
            for v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        DType::F16 => {
            for v in values {
                out.extend_from_slice(&f32_to_f16(*v).to_le_bytes());
            }
        }
        DType::BF16 => {
            for v in values {
                out.extend_from_slice(&f32_to_bf16(*v).to_le_bytes());
            }
        }
        DType::Q8_0 => {
            for block in values.chunks_exact(32) {
                let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                let dh = f32_to_f16(d);
                out.extend_from_slice(&dh.to_le_bytes());
                let d = f16_to_f32(dh); // quantize against the stored scale
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                for v in block {
                    out.push((v * inv).round().clamp(-127.0, 127.0) as i8 as u8);
                }
            }
        }
        DType::Q4_K => {
            for sb in values.chunks_exact(256) {
                // Per 32-elem sub-block: value = d*sc*q - dmin*m with q ∈ 0..=15.
                let mut effs = [0f32; 8]; // effective scale per sub-block
                let mut mins = [0f32; 8]; // effective (positive) min offset
                for (j, blk) in sb.chunks_exact(32).enumerate() {
                    let mn = blk.iter().fold(f32::INFINITY, |m, v| m.min(*v));
                    let mx = blk.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
                    mins[j] = (-mn).max(0.0);
                    effs[j] = (mx + mins[j]).max(0.0) / 15.0;
                }
                let dsup = effs.iter().fold(0f32, |m, v| m.max(*v)) / 63.0;
                let msup = mins.iter().fold(0f32, |m, v| m.max(*v)) / 63.0;
                let dh = f32_to_f16(dsup);
                let mh = f32_to_f16(msup);
                out.extend_from_slice(&dh.to_le_bytes());
                out.extend_from_slice(&mh.to_le_bytes());
                let (dsup, msup) = (f16_to_f32(dh), f16_to_f32(mh));
                let q6 = |x: f32, s: f32| -> u8 {
                    if s > 0.0 {
                        (x / s).round().clamp(0.0, 63.0) as u8
                    } else {
                        0
                    }
                };
                let lsc: Vec<u8> = effs.iter().map(|&e| q6(e, dsup)).collect();
                let lm: Vec<u8> = mins.iter().map(|&m| q6(m, msup)).collect();
                // Inverse of get_scale_min_k4.
                let mut scales = [0u8; 12];
                scales[..4].copy_from_slice(&lsc[..4]);
                scales[4..(4 + 4)].copy_from_slice(&lm[..4]);
                for j in 4..8 {
                    scales[j + 4] = (lsc[j] & 0xF) | ((lm[j] & 0xF) << 4);
                    scales[j - 4] |= (lsc[j] >> 4) << 6;
                    scales[j] |= (lm[j] >> 4) << 6;
                }
                out.extend_from_slice(&scales);
                // Quantize elements, packing nibbles in ggml's chunk-of-64 order.
                let quant = |j: usize, x: f32| -> u8 {
                    let sc = dsup * f32::from(lsc[j]);
                    let m = msup * f32::from(lm[j]);
                    if sc > 0.0 {
                        ((x + m) / sc).round().clamp(0.0, 15.0) as u8
                    } else {
                        0
                    }
                };
                for pair in 0..4 {
                    let (j1, j2) = (pair * 2, pair * 2 + 1);
                    for l in 0..32 {
                        let lo = quant(j1, sb[j1 * 32 + l]);
                        let hi = quant(j2, sb[j2 * 32 + l]);
                        out.push(lo | (hi << 4));
                    }
                }
            }
        }
        DType::Unsupported(s) => return Err(bad(format!("unsupported dtype {s}"))),
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn f16_known_vectors() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x7BFF), 65504.0); // f16 max
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8); // smallest subnormal
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(-2.0), 0xC000);
        assert_eq!(f32_to_f16(65504.0), 0x7BFF);
        assert_eq!(f32_to_f16(1e6), 0x7C00); // overflow → +inf
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn bf16_known_vectors() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xC040), -3.0);
        assert_eq!(f32_to_bf16(1.0), 0x3F80);
        // RNE: 1.0039063 is exactly between 0x3F80 and 0x3F81 → even (0x3F80)
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8000)), 0x3F80);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    }

    #[test]
    fn q8_0_roundtrip_exactish() {
        // 32 values in [-1, 1]; max abs error after roundtrip ≤ d/2 = amax/254.
        let vals: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) / 15.5).collect();
        let packed = pack(&DType::Q8_0, &vals).unwrap();
        assert_eq!(packed.len(), 34);
        let out = dequant(&DType::Q8_0, &packed, 32).unwrap();
        for (a, b) in vals.iter().zip(&out) {
            assert!((a - b).abs() <= 1.0 / 254.0 + 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn q4_k_roundtrip_block() {
        let vals: Vec<f32> = (0..256)
            .map(|i| ((i * 37 % 256) as f32 / 128.0) - 1.0)
            .collect();
        let packed = pack(&DType::Q4_K, &vals).unwrap();
        assert_eq!(packed.len(), 144);
        let out = dequant(&DType::Q4_K, &packed, 256).unwrap();
        // Simple min/max quantizer worst case: 4-bit step ≤ 2·amax/15 (half-
        // step error ~6.7% of amax) plus 6-bit scale quantization → 11%.
        let amax = vals.iter().fold(0f32, |m, v| m.max(v.abs()));
        for (a, b) in vals.iter().zip(&out) {
            assert!((a - b).abs() <= 0.11 * amax, "{a} vs {b}");
        }
    }

    #[test]
    fn dequant_rejects_bad_lengths() {
        assert!(dequant(&DType::F32, &[0u8; 7], 2).is_err()); // 2 f32 = 8 bytes
        assert!(dequant(&DType::Q8_0, &[0u8; 34], 31).is_err()); // not block-aligned
        assert!(pack(&DType::Q4_K, &[0f32; 100]).is_err()); // not multiple of 256
        assert!(dequant(&DType::Unsupported("x".into()), &[], 0).is_err());
    }
}
