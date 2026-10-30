REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
export CF="$REPO/third_party/libero-cf"
export SIM_PY="$REPO/envs/sim/.venv/bin/python"
export POLICY_PY="$REPO/envs/policy/.venv/bin/python"
export LIBERO_CONFIG_PATH="$REPO/.libero"
export PYTHONPATH="$CF${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
[ "$MUJOCO_GL" = osmesa ] && export PYOPENGL_PLATFORM=osmesa
true
