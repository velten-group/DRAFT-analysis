#!/usr/bin/env bash
# Compile the unchanged official source against one explicit Conda sysroot.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || {
  echo 'scMTNI build requires Linux x86_64.' >&2; exit 1;
}
: "${CONDA_PREFIX:?Run with conda run -n simulation_scmtni bash comparisons/simulation_scmtni/build.sh}"
[[ "$CONDA_PREFIX" != *[[:space:]]* ]] || {
  echo 'The official Makefile requires a Conda prefix without whitespace.' >&2; exit 1;
}
COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
SYSROOT="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"
[[ -x "$COMPILER" && -f "$SYSROOT/usr/include/features.h" ]] || {
  echo 'Missing compiler/sysroot. Run bash comparisons/simulation_benchmark/setup.sh scmtni.' >&2
  exit 1
}
[[ -f "$HERE/upstream/Code/Makefile" ]] || {
  echo 'Missing official source; run the scMTNI setup first.' >&2; exit 1;
}

mkdir -p "$HERE/runs/build"
BUILD_DIR=$(mktemp -d "$HERE/runs/build/build_$(date +%Y%m%d_%H%M%S)_XXXXXX")
printf 'scMTNI build diagnostics: %s\n' "$BUILD_DIR"
{
  printf 'compiler=%s\nsysroot=%s\n' "$COMPILER" "$SYSROOT"
  "$COMPILER" --version
  getconf GNU_LIBC_VERSION
} | tee "$BUILD_DIR/toolchain.txt"

cat > "$BUILD_DIR/link_probe.cpp" <<'CPP'
#include <features.h>
#if !defined(__GLIBC__) || __GLIBC__ < 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ < 17)
#error "scMTNI dependencies require a GLIBC sysroot of at least 2.17"
#endif
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_sf_bessel.h>
#include <iostream>
#include <time.h>
int main() {
    double x[2] = {1.0, 2.0}, y[2] = {3.0, 4.0};
    timespec stamp;
    if (clock_gettime(CLOCK_MONOTONIC, &stamp) != 0) return 1;
    if (cblas_ddot(2, x, 1, y, 1) != 11.0) return 2;
    std::cout << "GSL link/run probe passed: " << gsl_sf_bessel_J0(1.0) << '\n';
    return 0;
}
CPP

# rpath points only to the Conda dependency libraries. The sysroot is a LINK-TIME
# target; never place its libc in the runtime loader path.
FLAGS=(-O2 -std=gnu++14 "--sysroot=$SYSROOT" "-I$CONDA_PREFIX/include")
LINKS=("-L$CONDA_PREFIX/lib" "-Wl,-rpath,$CONDA_PREFIX/lib"
       "-Wl,-rpath-link,$CONDA_PREFIX/lib"
       "-Wl,-rpath-link,$SYSROOT/lib64" "-Wl,-rpath-link,$SYSROOT/usr/lib64"
       -lgsl -lgslcblas -lm)
if ! "$COMPILER" "${FLAGS[@]}" "$BUILD_DIR/link_probe.cpp" -o "$BUILD_DIR/link_probe" "${LINKS[@]}" > "$BUILD_DIR/link_probe.log" 2>&1; then
  cat "$BUILD_DIR/link_probe.log" >&2
  printf 'GSL link probe failed; scMTNI compilation was not started. Diagnostics: %s\n' "$BUILD_DIR" >&2
  exit 1
fi
"$BUILD_DIR/link_probe" | tee "$BUILD_DIR/link_probe_run.log"
make -C "$HERE/upstream/Code" CC="$COMPILER" CFLAGS="${FLAGS[*]}" LFLAG="${LINKS[*]}" 2>&1 | tee "$BUILD_DIR/make.log"
test -x "$HERE/upstream/Code/scMTNI"
test -s "$HERE/upstream/Code/scMTNI"
# Code/scMTNI is tracked upstream but is a mutable build output. Keep a separate
# successful-build hash so training cannot accept a stale/unverified executable.
BINARY_SHA256=$(sha256sum "$HERE/upstream/Code/scMTNI")
BINARY_SHA256=${BINARY_SHA256%% *}
printf '%s\n' "$BINARY_SHA256" > "$BUILD_DIR/scMTNI.sha256"
cp "$BUILD_DIR/scMTNI.sha256" "$HERE/runs/build/verified_binary.sha256.part"
mv -- "$HERE/runs/build/verified_binary.sha256.part" "$HERE/runs/build/verified_binary.sha256"
printf 'Official scMTNI build complete. Diagnostics: %s\n' "$BUILD_DIR"
