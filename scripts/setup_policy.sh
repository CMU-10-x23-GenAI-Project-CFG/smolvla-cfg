#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO/policy_server"
UV_PROJECT_ENVIRONMENT="$REPO/envs/policy/.venv" uv sync --locked --python 3.10
