//! `TargetDesc` + hardware detection + named target profiles. Pure data and
//! probing: always an explicit input to planning and codegen (spec
//! §inferno-target). A detected target and a profile-loaded target are the
//! same struct — that equivalence is the cross-compilation interface.

mod desc;
mod detect;
#[cfg(target_os = "macos")]
mod detect_macos;
mod error;
mod profile;

pub use desc::{BwClass, CacheLevel, CoreTopology, Feature, Isa, TargetDesc};
pub use error::{Result, TargetError};
pub use profile::available_profiles;
