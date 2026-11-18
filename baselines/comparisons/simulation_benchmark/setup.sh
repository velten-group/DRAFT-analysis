#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
command -v conda >/dev/null || { echo 'conda is required.' >&2; exit 1; }
methods=("$@")
if (( ${#methods[@]} == 0 )); then methods=(scenicplus scmtni scdori dictys); fi
for method in "${methods[@]}"; do
  case "$method" in scenicplus|scmtni|scdori|dictys) ;; *) echo "Unknown method: $method" >&2; exit 1;; esac
  envname="simulation_${method}"
  python_version=3.11
  if [[ "$method" == scenicplus ]]; then
    # Reuse the environment that already runs the real-data SCENIC+ workflow.
    envname=${SCENICPLUS_ENV:-scenicplus_r2g}
    python_version=$(< comparisons/simulation_scenicplus/python-version.txt)
  fi
  created=0
  if conda run -n "$envname" python -c 'import sys' >/dev/null 2>&1; then
    if [[ "$method" == scenicplus ]]; then
      conda run -n "$envname" python -c "import platform; assert platform.python_version() == '$python_version', 'Existing real-data SCENIC+ environment must use Python $python_version; it is preserved unchanged'"
    else
      conda run -n "$envname" python -c 'import sys; assert sys.version_info[:2] == (3,11), "Existing baseline environment must use Python 3.11; preserved unchanged"'
    fi
  else
    conda create -y -n "$envname" -c conda-forge "python=$python_version" pip
    created=1
  fi
  if [[ "$method" == scenicplus ]]; then
    if (( created )); then
      conda run --no-capture-output -n "$envname" python -m pip install -r comparisons/simulation_scenicplus/requirements.txt
    elif ! conda run -n "$envname" python -c 'import yaml' >/dev/null 2>&1; then
      # Configuration parsing only; the native scientific stack is left intact.
      conda run --no-capture-output -n "$envname" python -m pip install 'PyYAML>=6,<7'
    fi
  else
    conda run --no-capture-output -n "$envname" python -m pip install -r "comparisons/${envname}/requirements.txt"
  fi
  case "$method" in
    scdori|dictys)
      # CPU by default. For CUDA, supply an official PyTorch 2.6 wheel index explicitly.
      conda run --no-capture-output -n "$envname" python -m pip install torch==2.6.0 --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
      ;;
  esac
  conda run --no-capture-output -n "$envname" python comparisons/simulation_benchmark/bootstrap.py "$method"
  if [[ "$method" == scmtni ]]; then
    [[ $(uname -s) == Linux ]] || { echo 'Official scMTNI must be built on Linux.' >&2; exit 1; }
    conda install -y -n "$envname" --override-channels -c conda-forge --strict-channel-priority \
      --file comparisons/simulation_scmtni/toolchain.txt
    conda run --no-capture-output -n "$envname" bash comparisons/simulation_scmtni/build.sh
  fi
  conda run --no-capture-output -n "$envname" python -m pip check
  if [[ "$method" == scenicplus ]]; then
    conda run --no-capture-output -n "$envname" python comparisons/simulation_scenicplus/check_runtime.py
  fi
  echo "Ready: $envname"
done
