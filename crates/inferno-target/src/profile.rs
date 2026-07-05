//! Named target profiles: TOML files embedded at build time. A profile and a
//! live detection produce the same `TargetDesc`.

use crate::{Result, TargetDesc, TargetError};

const PROFILES: &[(&str, &str)] = &[("ryzen-3900", include_str!("../profiles/ryzen-3900.toml"))];

pub fn available_profiles() -> Vec<&'static str> {
    PROFILES.iter().map(|(n, _)| *n).collect()
}

impl TargetDesc {
    pub fn from_profile(name: &str) -> Result<TargetDesc> {
        let Some((_, text)) = PROFILES.iter().find(|(n, _)| *n == name) else {
            return Err(TargetError::UnknownProfile {
                name: name.to_string(),
                available: available_profiles().join(", "),
            });
        };
        toml::from_str(text).map_err(|e| TargetError::MalformedProfile {
            name: name.to_string(),
            detail: e.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BwClass, CacheLevel, CoreTopology, Isa, TargetError};

    #[test]
    fn ryzen_profile_loads() {
        let t = TargetDesc::from_profile("ryzen-3900").unwrap();
        assert_eq!(t.isa, Isa::X86_64v3);
        assert!(t.features.is_empty());
        assert_eq!(t.page_size, 4096);
        assert_eq!(t.memory_bw_class, Some(BwClass::Consumer));
        assert_eq!(
            t.topology,
            CoreTopology {
                physical_cores: 12,
                logical_cores: 24,
                smt: true
            }
        );
        assert_eq!(
            t.caches,
            vec![
                CacheLevel {
                    level: 1,
                    size_bytes: 32 * 1024,
                    line_bytes: 64,
                    shared_by: 2
                },
                CacheLevel {
                    level: 2,
                    size_bytes: 512 * 1024,
                    line_bytes: 64,
                    shared_by: 2
                },
                CacheLevel {
                    level: 3,
                    size_bytes: 16 * 1024 * 1024,
                    line_bytes: 64,
                    shared_by: 6
                },
            ]
        );
    }

    #[test]
    fn unknown_profile_lists_available() {
        let err = TargetDesc::from_profile("m3").unwrap_err();
        let TargetError::UnknownProfile { name, available } = err else {
            panic!("wrong variant: {err}");
        };
        assert_eq!(name, "m3");
        assert!(available.contains("ryzen-3900"));
    }

    #[test]
    fn toml_roundtrip_is_identity() {
        let t = TargetDesc::from_profile("ryzen-3900").unwrap();
        let text = toml::to_string_pretty(&t).unwrap();
        let back: TargetDesc = toml::from_str(&text).unwrap();
        assert_eq!(t, back);
    }
}
