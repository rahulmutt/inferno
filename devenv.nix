# Native dependencies mise cannot supply (see AGENTS.md / developer-environment
# skill). Everything else — rust, cargo tools — is pinned in mise.toml.
{ pkgs, ... }:

{
  packages = [
    # LLVM for inferno-codegen (llvm-sys/inkwell). Major version MUST match
    # the inkwell feature flag in crates/inferno-codegen (llvm18-1).
    pkgs.llvmPackages_18.llvm.dev
    pkgs.libffi
    pkgs.libxml2
    pkgs.zlib
    # Pinned benchmark opponent for `mise run bench` (M4).
    pkgs.llama-cpp
  ];

  env.LLVM_SYS_180_PREFIX = "${pkgs.llvmPackages_18.llvm.dev}";

  enterShell = ''
    echo "inferno devenv: LLVM $(llvm-config --version)"
  '';
}
