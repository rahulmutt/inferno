//! macOS/aarch64 hardware detection via sysctl. Populates the same
//! `TargetDesc` the x86 sysfs path produces (crate-doc equivalence contract).
#![cfg(target_os = "macos")]

use crate::desc::{CacheLevel, CoreTopology, Feature, Isa, TargetDesc};
use crate::error::{Result, TargetError};
use std::collections::BTreeSet;
use std::ffi::CString;

fn sysctl_u64(name: &str) -> Option<u64> {
    let cname = CString::new(name).ok()?;
    let mut val: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: writing a u64 out-param sized exactly; name is NUL-terminated.
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            (&mut val as *mut u64).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 && len == std::mem::size_of::<u64>() {
        Some(val)
    } else {
        None
    }
}

fn sysctl_flag(name: &str) -> bool {
    sysctl_u64(name).unwrap_or(0) != 0
}

pub fn detect_macos() -> Result<TargetDesc> {
    // NEON is mandatory on ARMv8; treat its absence as a non-Apple-Silicon host.
    if !sysctl_flag("hw.optional.neon") && !sysctl_flag("hw.optional.AdvSIMD") {
        return Err(TargetError::UnsupportedPlatform {
            detail: "NEON not reported by sysctl; not an Apple Silicon host".into(),
        });
    }
    let mut features = BTreeSet::new();
    if sysctl_flag("hw.optional.arm.FEAT_DotProd") {
        features.insert(Feature::Dotprod);
    }
    if sysctl_flag("hw.optional.arm.FEAT_I8MM") {
        features.insert(Feature::I8mm);
    }

    let physical = sysctl_u64("hw.physicalcpu").unwrap_or(0) as u32;
    let logical = sysctl_u64("hw.logicalcpu").unwrap_or(physical as u64) as u32;
    let perf = sysctl_u64("hw.perflevel0.physicalcpu").map(|v| v as u32);
    let eff = sysctl_u64("hw.perflevel1.physicalcpu").map(|v| v as u32);
    if physical == 0 {
        return Err(TargetError::UnsupportedPlatform {
            detail: "hw.physicalcpu unavailable".into(),
        });
    }
    let topology = CoreTopology {
        physical_cores: physical,
        logical_cores: logical,
        smt: logical > physical, // Apple has no SMT; this stays false
        perf_cores: perf,
        eff_cores: eff,
    };

    let line = sysctl_u64("hw.cachelinesize").unwrap_or(128) as u32;
    let mut caches = Vec::new();
    if let Some(l1) =
        sysctl_u64("hw.perflevel0.l1dcachesize").or_else(|| sysctl_u64("hw.l1dcachesize"))
    {
        caches.push(CacheLevel {
            level: 1,
            size_bytes: l1,
            line_bytes: line,
            shared_by: 1,
        });
    }
    if let Some(l2) =
        sysctl_u64("hw.perflevel0.l2cachesize").or_else(|| sysctl_u64("hw.l2cachesize"))
    {
        // Apple L2 is shared per performance cluster.
        caches.push(CacheLevel {
            level: 2,
            size_bytes: l2,
            line_bytes: line,
            shared_by: perf.unwrap_or(1),
        });
    }

    Ok(TargetDesc {
        isa: Isa::Aarch64Neon,
        features,
        page_size: sysctl_u64("hw.pagesize").unwrap_or(16384),
        memory_bw_class: None, // profile-only, never detected (matches x86 path)
        topology,
        caches,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desc::{Feature, Isa};

    #[test]
    fn detect_populates_apple_silicon_descriptor() {
        let d = detect_macos().expect("detection must succeed on Apple Silicon");
        assert_eq!(d.isa, Isa::Aarch64Neon);
        assert!(d.features.contains(&Feature::Dotprod));
        assert!(d.topology.physical_cores >= 4);
        assert!(
            d.topology.perf_cores.unwrap_or(0) >= 1,
            "P-core count must be detected"
        );
        assert!(d.page_size >= 4096);
        assert!(!d.caches.is_empty(), "at least L1d + L2 must be detected");
    }
}
