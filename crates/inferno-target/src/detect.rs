//! Live hardware detection, layered for testability (spec §inferno-target):
//! pure functions parse a captured `/sys/devices/system/cpu` tree passed as a
//! root path; a thin live layer supplies the real root, `is_x86_feature_detected!`
//! for the ISA, and the page size. No downstream crate re-probes hardware.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use crate::{CacheLevel, CoreTopology, Feature, Isa, Result, TargetDesc, TargetError};

impl TargetDesc {
    pub fn detect() -> Result<TargetDesc> {
        let (isa, features) = detect_isa()?;
        let root = Path::new("/sys/devices/system/cpu");
        Ok(TargetDesc {
            isa,
            features,
            page_size: rustix::param::page_size() as u64,
            memory_bw_class: None,
            topology: parse_topology(root)?,
            caches: parse_caches(root)?,
        })
    }
}

#[cfg(target_arch = "x86_64")]
fn detect_isa() -> Result<(Isa, BTreeSet<Feature>)> {
    macro_rules! det {
        ($f:tt) => {
            std::arch::is_x86_feature_detected!($f)
        };
    }
    let v3 = det!("avx2")
        && det!("fma")
        && det!("bmi1")
        && det!("bmi2")
        && det!("f16c")
        && det!("lzcnt")
        && det!("movbe");
    if !v3 {
        return Err(TargetError::UnsupportedPlatform {
            detail: "CPU below x86-64-v3 (needs AVX2+FMA)".to_string(),
        });
    }
    let v4 = det!("avx512f")
        && det!("avx512bw")
        && det!("avx512cd")
        && det!("avx512dq")
        && det!("avx512vl");
    let mut features = BTreeSet::new();
    if det!("avx512vnni") {
        features.insert(Feature::Vnni);
    }
    if det!("avx512bf16") {
        features.insert(Feature::Bf16);
    }
    Ok((if v4 { Isa::X86_64v4 } else { Isa::X86_64v3 }, features))
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_isa() -> Result<(Isa, BTreeSet<Feature>)> {
    Err(TargetError::UnsupportedPlatform {
        detail: "only x86-64 detection is implemented (M2)".to_string(),
    })
}

fn bad(path: &Path, detail: impl Into<String>) -> TargetError {
    TargetError::MalformedSysfs {
        path: path.display().to_string(),
        detail: detail.into(),
    }
}

fn read_trim(path: &Path) -> Result<String> {
    fs::read_to_string(path)
        .map(|s| s.trim().to_string())
        .map_err(|e| bad(path, e.to_string()))
}

fn parse_topology(root: &Path) -> Result<CoreTopology> {
    let entries = fs::read_dir(root).map_err(|e| bad(root, e.to_string()))?;
    let mut logical = 0u32;
    let mut cores = BTreeSet::new();
    for entry in entries {
        let entry = entry.map_err(|e| bad(root, e.to_string()))?;
        let name = entry.file_name();
        let Some(idx) = name
            .to_string_lossy()
            .strip_prefix("cpu")
            .map(str::to_string)
        else {
            continue;
        };
        if idx.is_empty() || !idx.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let topo = entry.path().join("topology");
        let pkg = read_trim(&topo.join("physical_package_id"))?;
        let core = read_trim(&topo.join("core_id"))?;
        cores.insert((pkg, core));
        logical += 1;
    }
    if logical == 0 {
        return Err(bad(root, "no cpuN directories"));
    }
    let physical = cores.len() as u32;
    Ok(CoreTopology {
        physical_cores: physical,
        logical_cores: logical,
        smt: logical > physical,
    })
}

fn parse_caches(root: &Path) -> Result<Vec<CacheLevel>> {
    let dir = root.join("cpu0/cache");
    let entries = fs::read_dir(&dir).map_err(|e| bad(&dir, e.to_string()))?;
    let mut out = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| bad(&dir, e.to_string()))?;
        if !entry.file_name().to_string_lossy().starts_with("index") {
            continue;
        }
        let p = entry.path();
        let ty = read_trim(&p.join("type"))?;
        if ty != "Data" && ty != "Unified" {
            continue;
        }
        let level = read_trim(&p.join("level"))?
            .parse::<u8>()
            .map_err(|e| bad(&p, format!("level: {e}")))?;
        let size_bytes = parse_size(&read_trim(&p.join("size"))?, &p.display().to_string())?;
        let line_bytes = read_trim(&p.join("coherency_line_size"))?
            .parse::<u32>()
            .map_err(|e| bad(&p, format!("coherency_line_size: {e}")))?;
        let shared_by = parse_cpu_list(
            &read_trim(&p.join("shared_cpu_list"))?,
            &p.display().to_string(),
        )?;
        out.push(CacheLevel {
            level,
            size_bytes,
            line_bytes,
            shared_by,
        });
    }
    if out.is_empty() {
        return Err(bad(&dir, "no Data/Unified cache index directories"));
    }
    out.sort_by_key(|c| c.level);
    Ok(out)
}

/// "32K" | "4M" | "512" → bytes.
fn parse_size(s: &str, ctx: &str) -> Result<u64> {
    let err = || TargetError::MalformedSysfs {
        path: ctx.to_string(),
        detail: format!("unparseable cache size `{s}`"),
    };
    if let Some(n) = s.strip_suffix('K') {
        return n.parse::<u64>().map(|n| n * 1024).map_err(|_| err());
    }
    if let Some(n) = s.strip_suffix('M') {
        return n.parse::<u64>().map(|n| n * 1024 * 1024).map_err(|_| err());
    }
    s.parse::<u64>().map_err(|_| err())
}

/// "0-2,12-14" → count of listed CPUs.
fn parse_cpu_list(s: &str, ctx: &str) -> Result<u32> {
    let err = |d: String| TargetError::MalformedSysfs {
        path: ctx.to_string(),
        detail: d,
    };
    let mut count = 0u32;
    for part in s.split(',') {
        match part.split_once('-') {
            Some((a, b)) => {
                let a: u32 = a.parse().map_err(|_| err(format!("bad range `{part}`")))?;
                let b: u32 = b.parse().map_err(|_| err(format!("bad range `{part}`")))?;
                if b < a {
                    return Err(err(format!("inverted range `{part}`")));
                }
                count += b - a + 1;
            }
            None => {
                part.parse::<u32>()
                    .map_err(|_| err(format!("bad cpu id `{part}`")))?;
                count += 1;
            }
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CacheLevel, CoreTopology, TargetDesc};
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sys-cpu-ryzen-3900")
    }

    #[test]
    fn fixture_topology() {
        assert_eq!(
            parse_topology(&fixture()).unwrap(),
            CoreTopology {
                physical_cores: 12,
                logical_cores: 24,
                smt: true
            }
        );
    }

    #[test]
    fn fixture_caches() {
        // index1 (32K Instruction) must be skipped; Data/Unified kept, sorted by level.
        assert_eq!(
            parse_caches(&fixture()).unwrap(),
            vec![
                CacheLevel {
                    level: 1,
                    size_bytes: 32768,
                    line_bytes: 64,
                    shared_by: 2
                },
                CacheLevel {
                    level: 2,
                    size_bytes: 524288,
                    line_bytes: 64,
                    shared_by: 2
                },
                CacheLevel {
                    level: 3,
                    size_bytes: 16777216,
                    line_bytes: 64,
                    shared_by: 6
                },
            ]
        );
    }

    #[test]
    fn missing_root_is_typed_error() {
        let err = parse_topology(std::path::Path::new("/nonexistent-sys")).unwrap_err();
        assert!(
            matches!(err, crate::TargetError::MalformedSysfs { .. }),
            "{err}"
        );
    }

    #[test]
    fn cpu_list_and_size_parsers() {
        assert_eq!(parse_cpu_list("0,12", "t").unwrap(), 2);
        assert_eq!(parse_cpu_list("0-2,12-14", "t").unwrap(), 6);
        assert_eq!(parse_cpu_list("7", "t").unwrap(), 1);
        assert_eq!(parse_size("32K", "t").unwrap(), 32768);
        assert_eq!(parse_size("16384K", "t").unwrap(), 16777216);
        assert_eq!(parse_size("4M", "t").unwrap(), 4 * 1024 * 1024);
        assert_eq!(parse_size("512", "t").unwrap(), 512);
        assert!(parse_size("32Q", "t").is_err());
    }

    /// Live detection must succeed and be internally coherent wherever the
    /// suite runs (CI runners and the dev box are both x86-64-v3+ Linux).
    #[test]
    fn live_detect_is_coherent() {
        let t = TargetDesc::detect().unwrap();
        assert!(t.topology.logical_cores >= t.topology.physical_cores);
        assert!(t.topology.physical_cores >= 1);
        assert!(!t.caches.is_empty());
        assert!(t.page_size >= 4096);
        assert!(t.memory_bw_class.is_none());
    }

    /// detect == profile equivalence (spec §inferno-target). Machine-specific:
    /// gated on INFERNO_EXPECT_PROFILE, set on the dev box; vacuous elsewhere.
    /// memory_bw_class is profile-only, so it is cleared before comparing.
    #[test]
    fn detect_matches_expected_profile() {
        let Ok(name) = std::env::var("INFERNO_EXPECT_PROFILE") else {
            return;
        };
        let mut profile = TargetDesc::from_profile(&name).unwrap();
        profile.memory_bw_class = None;
        assert_eq!(TargetDesc::detect().unwrap(), profile);
    }
}
