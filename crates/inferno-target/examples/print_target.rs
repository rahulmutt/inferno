//! Print the detected TargetDesc as profile-ready TOML.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    print!(
        "{}",
        toml::to_string_pretty(&inferno_target::TargetDesc::detect()?)?
    );
    Ok(())
}
