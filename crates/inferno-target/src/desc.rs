//! `TargetDesc`: plain serde-able data describing a machine. The same struct
//! whether probed live or loaded from a named profile — that equivalence is
//! the future cross-compile interface (spec §inferno-target). Always an
//! explicit input to planning/codegen; nothing downstream re-probes.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// ISA at kernel-dispatch granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Isa {
    /// AVX2 + FMA (+BMI1/2, F16C, LZCNT, MOVBE). All M2 kernels target this.
    #[serde(rename = "x86-64-v3")]
    X86_64v3,
    /// v3 + AVX-512 F/BW/CD/DQ/VL. Defined for dispatch; no M2 kernels.
    #[serde(rename = "x86-64-v4")]
    X86_64v4,
    /// Apple Silicon / ARMv8-A with mandatory NEON (128-bit Advanced SIMD).
    /// AMX/SME are deliberately not modeled — inferno emits pure NEON (M5).
    Aarch64Neon,
}

/// Features outside the ISA level that future kernels may dispatch on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Feature {
    Vnni,
    Bf16,
    /// ARM dotprod (SDOT/UDOT) — present on Apple M1+.
    Dotprod,
    /// ARM i8mm (SMMLA) — absent on M1; gated for M2+.
    I8mm,
}

/// One data/unified cache level as seen by a single core.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheLevel {
    pub level: u8,
    pub size_bytes: u64,
    pub line_bytes: u32,
    /// Logical CPUs sharing this cache instance.
    pub shared_by: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreTopology {
    pub physical_cores: u32,
    pub logical_cores: u32,
    pub smt: bool,
    /// Performance-core count on heterogeneous chips (Apple P/E). `None` on flat SMP.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub perf_cores: Option<u32>,
    /// Efficiency-core count on heterogeneous chips. `None` on flat SMP.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eff_cores: Option<u32>,
}

/// Coarse memory-bandwidth class. Profile-only: nothing detects it; the M3
/// planner may consume it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BwClass {
    Consumer,
    Workstation,
    Server,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetDesc {
    pub isa: Isa,
    pub features: BTreeSet<Feature>,
    pub page_size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_bw_class: Option<BwClass>,
    pub topology: CoreTopology,
    pub caches: Vec<CacheLevel>,
}
