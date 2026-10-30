#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CF="$REPO/third_party/libero-cf"
VENV="$REPO/envs/sim/.venv"

mkdir -p "$(dirname "$VENV")"
uv venv --python 3.8 --prompt cf_sim "$VENV"
export VIRTUAL_ENV="$VENV"
uv pip sync "$CF/requirements.txt" "$CF/requirements_libero.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install -e "$CF/packages/openpi-client"
uv pip install -e "$CF"

BENCH="$CF/libero/libero"
mkdir -p "$REPO/.libero" "$CF/libero/datasets"
cat > "$REPO/.libero/config.yaml" <<YAML
assets: $BENCH/assets
bddl_files: $BENCH/bddl_files
benchmark_root: $BENCH
conditions: $BENCH/conditions
datasets: $CF/libero/datasets
init_states: $BENCH/init_files
YAML
