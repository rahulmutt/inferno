# Native dependencies mise cannot supply (see AGENTS.md / developer-environment
# skill). Everything else — rust, cargo tools — is pinned in mise.toml.
{ pkgs, ... }:

let
  # The controlled benchmark opponent MUST be the pure-CPU ggml build:
  # nixpkgs' stock llama-cpp links OpenBLAS (blasSupport defaults on for
  # CPU-only builds), and OpenBLAS's internal all-core thread pool ignores
  # llama-bench's -t pin — it breaks every thread-controlled comparison and
  # oversubscribes at pinned t (M4a spec, 2026-07-11 amendment). This
  # override misses the NixOS binary cache, so it compiles locally once per
  # fresh environment (a few minutes, metal boxes included).
  llama-cpp-cpu = pkgs.llama-cpp.override { blasSupport = false; };
in
{
  packages = [
    # LLVM for inferno-codegen (llvm-sys/inkwell). Major version MUST match
    # the inkwell feature flag in crates/inferno-codegen (llvm18-1).
    pkgs.llvmPackages_18.llvm.dev
    pkgs.libffi
    pkgs.libxml2
    pkgs.zlib
    # Pinned benchmark opponent for `mise run bench` (M4) — pure-CPU build.
    llama-cpp-cpu
  ]
  # Socket pinning for the quiet-hw gates (numa_wrap). A dual-socket box must
  # take its point on one socket or NUMA effects contaminate it, and without
  # this the pinned gates die with exit 127 — after the box is paid for.
  ++ pkgs.lib.optionals pkgs.stdenv.isLinux [ pkgs.numactl ];

  env.LLVM_SYS_181_PREFIX = "${pkgs.llvmPackages_18.llvm.dev}";
  # ggml CPU backend for `mise run bench-kernels` (--features ggml-compare).
  # haswell = AVX2+FMA — the same ISA class as inferno's M2 kernels, so the
  # comparison is apples-to-apples. The per-arch backends live under bin/.
  env.INFERNO_GGML_CPU_LIB = "${llama-cpp-cpu}/bin/libggml-cpu-haswell.so";
  # Stock (BLAS) llama-bench, deliberately NOT on PATH: only
  # gate-bench-protocol.sh reads this, to record a "llama at its best"
  # reference row next to the controlled pure-CPU comparison.
  env.INFERNO_LLAMA_BENCH_BLAS = "${pkgs.llama-cpp}/bin/llama-bench";

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
