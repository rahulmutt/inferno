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

  env.LLVM_SYS_181_PREFIX = "${pkgs.llvmPackages_18.llvm.dev}";
  # ggml CPU backend for `mise run bench-kernels` (--features ggml-compare).
  # haswell = AVX2+FMA — the same ISA class as inferno's M2 kernels, so the
  # comparison is apples-to-apples. The per-arch backends live under bin/.
  env.INFERNO_GGML_CPU_LIB = "${pkgs.llama-cpp}/bin/libggml-cpu-haswell.so";

  # Runtime deps of the statically-linked LLVM libs that inferno-codegen pulls
  # in via llvm-sys/inkwell (libstdc++ since LLVM is C++, libffi/libxml2/zlib
  # since LLVM is built against them, ncurses for terminfo). Without this,
  # `cargo test -p inferno-codegen` links fine but fails at runtime with
  # "error while loading shared libraries" because the devenv shell's glibc
  # doesn't consult the host's /etc/ld.so.cache.
  env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.libffi
    pkgs.libxml2
    pkgs.zlib
    pkgs.ncurses
  ];

  enterShell = ''
    echo "inferno devenv: LLVM $(llvm-config --version)"
  '';
}
