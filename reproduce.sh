#!/usr/bin/env bash
# One command for the checkpoint line (Linux, macOS):   bash reproduce.sh
#
# Run it from this folder after placing the five draft_analysis_data_*.tar.gz archives here (next to README.md); if
# they are missing it tries to download them from Zenodo. It then
#   1. checks and unpacks the archives into data/,
#   2. builds the environment: with conda, a local environment .tools/ (Python 3.12, R, Java 17); without conda, a
#      .venv/ from python3.12/python3.11 and the R and Java found on PATH; packages in ~/.local are ignored,
#   3. downloads the scMTNI evaluator and checks every input file (SHA-256),
#   4. runs `draft-analysis checkpoint` (about 4 hours on one CPU thread) and
#   5. ends with REPRODUCED or NOT REPRODUCED. Everything is also written to reproduce.log.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONNOUSERSITE=1  # an older drosophila_gnn or dependency in ~/.local must not be used
unset PYTHONPATH
DOI_RECORD="https://zenodo.org/records/23678662/files"
ARCHIVES="draft_analysis_data_simulation.tar.gz draft_analysis_data_fly_raw.tar.gz
draft_analysis_data_fly_checkpoints.tar.gz draft_analysis_data_fly_frozen_external.tar.gz
draft_analysis_data_mouse.tar.gz"

fail() { echo; echo "!! $*"; echo "NOT REPRODUCED: the set-up stopped (see reproduce.log)"; exit 1; }

main() {
  echo "== 1/5 data"
  if [ ! -f data/fly/raw/A_rna_focus.h5ad ]; then
    for f in $ARCHIVES SHA256SUMS; do
      [ -f "$f" ] && continue
      echo "   downloading $f"
      curl -fL --retry 3 -o "$f.part" "$DOI_RECORD/$f?download=1" && mv "$f.part" "$f" || {
        rm -f "$f.part"
        [ "$f" = SHA256SUMS ] && continue
        fail "$f is missing and could not be downloaded: put the five archives in $(pwd) and run again"; }
    done
    if [ -f SHA256SUMS ]; then
      if command -v sha256sum >/dev/null; then sha256sum -c SHA256SUMS; else shasum -a 256 -c SHA256SUMS; fi \
        || fail "an archive is damaged (checksum differs): download it again"
    fi
    for a in $ARCHIVES; do echo "   unpacking $a"; tar -xzf "$a" -C data || fail "could not unpack $a"; done
  else
    echo "   data/ already unpacked"
  fi

  echo "== 2/5 environment"
  if command -v conda >/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    [ -x .tools/bin/python ] || conda create -y -p "$PWD/.tools" -c conda-forge python=3.12 r-base openjdk=17 \
      || fail "conda could not create .tools/"
    conda activate "$PWD/.tools"
    PY="$PWD/.tools/bin/python"
  else
    BASE=""
    for p in python3.12 python3.11 python3; do
      if command -v "$p" >/dev/null && "$p" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then BASE=$p; break; fi
    done
    [ -n "$BASE" ] || fail "Python 3.11 or newer is needed (or install conda and run again)"
    [ -x .venv/bin/python ] || "$BASE" -m venv .venv || fail "could not create .venv/"
    PY="$PWD/.venv/bin/python"
    command -v Rscript >/dev/null || fail "R (Rscript) is needed: install R, or install conda and run again"
    command -v java >/dev/null || [ -n "${DRAFT_JAVA:-}" ] || fail "Java 17 is needed: install it or set DRAFT_JAVA"
  fi
  "$PY" -m pip install -q -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu \
    || fail "pip could not install requirements.lock"
  "$PY" -m pip install -q -e . --no-deps || fail "pip could not install this repository"
  "$PY" --version
  { Rscript --version 2>&1 | head -1; java -version 2>&1 | head -1; } || true

  echo "== 3/5 inputs"
  "$PY" -m draft_analysis.cli data --download --verify || fail "inputs are missing or differ (see above)"

  echo "== 4/5 checkpoint line (about 4 hours)"
  set +e
  "$PY" -m draft_analysis.cli checkpoint
  status=$?
  set -e

  echo "== 5/5 result"
  grep -hE "^(main|supplementary): |^Supplementary Tables: " reproduce.log || true
  if [ "$status" -eq 0 ]; then
    echo "REPRODUCED: every table matches the published values (results/checkpoint/comparison_with_source_data.tsv)"
  else
    echo "NOT REPRODUCED (exit $status): see the FAIL lines above and results/checkpoint/comparison_with_source_data.tsv"
  fi
  return "$status"
}

main 2>&1 | tee reproduce.log
exit "${PIPESTATUS[0]}"
