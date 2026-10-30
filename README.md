# smolvla-cfg

Language guidance for SmolVLA on [LIBERO-CF](https://github.com/yuffish/libero-cf) ([paper](https://arxiv.org/abs/2602.17659)).

```
a = a_neg + w * (a_cond - a_neg)
```

Both passes share starting noise. They mix on the final action chunk, in normalized action space.

## Layout

| Path | Contents |
|---|---|
| `policy_server/smolvla_cf/policy.py` | Inference and guidance |
| `policy_server/smolvla_cf/serve.py` | WebSocket server |
| `policy_server/smolvla_cf/negatives.json` | Instruction → distractor prompt |
| `third_party/libero-cf` | Benchmark, evaluator, π0.5 servers (unmodified) |
| `third_party/lerobot` | SmolVLA source, v0.3.3 (unmodified) |

The evaluator (Python 3.8) and policy server (Python 3.10) are separate processes.

## Setup

Needs Ubuntu, an NVIDIA GPU and [`uv`](https://docs.astral.sh/uv/). Allow about 40 GB of disk.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recursive git@github.com:CMU-10-x23-GenAI-Project-CFG/smolvla-cfg.git
cd smolvla-cfg
scripts/setup_sim.sh
scripts/setup_policy.sh
```

Run `source scripts/env.sh` in every terminal. Checkpoints download on first use.

Rendering defaults to EGL. Without it, run `export MUJOCO_GL=osmesa` before sourcing.

## Run SmolVLA

```bash
# Terminal 1
$POLICY_PY -m smolvla_cf.serve --port 8000 --null negative --scale 2.0

# Terminal 2
mkdir -p results/run && cd results/run
$SIM_PY $CF/eval/main_cf.py --args.task_suite_name libero_cf_object --args.port 8000 --args.resize_size 256
```

| Experiment | Server flags |
|---|---|
| Baseline | `--null none` |
| Null vector | `--null mask --scale w` |
| Negative prompt | `--null negative --scale w` |
| Stronger guidance | Raise `--scale` |

- `--args.resize_size 256` is required: it matches the checkpoint's training images.
- `--null negative` only supports `libero_cf_object`. Other suites raise `KeyError`.
- Quick check: add `--args.num_trials_per_task 1`.
- Default is 50 trials × 10 tasks, about 2–3 h per configuration on one GPU.

## Read results

Output goes to `client.log` and `rollouts/` in the working directory.

| Paper term | Meaning | `libero_cf_object` log line |
|---|---|---|
| Grounding | Gripper touched the object | `touches` |
| Success | Object reached its goal | `successes` |
| Faithful | Instructed object | First condition |
| Biased | Trained-behavior object | Second condition |

- Ignore `Total success rate`: it also counts placing the distractor.
- `grep -c "Caught exception" client.log` must return 0. The evaluator otherwise records server errors as failed episodes.
- Check the process is alive, not only the log. If it dies, resume in a new directory with `--args.starting_task_id N` and redo task `N` from its first trial.
- In other suites, the first condition is not always the instructed object. Read the task's instruction and BDDL `:goal` before mapping conditions.

## Replicate the paper (π0.5)

Upstream code only. Use one server and one evaluator per configuration.

```bash
cd third_party/libero-cf/openpi-cf
GIT_LFS_SKIP_SMUDGE=1 UV_PYTHON_PREFERENCE=only-managed uv sync --locked --no-install-project
export PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false

# Baseline
.venv/bin/python scripts/serve_policy.py --env LIBERO --port 8000

# Training-free guidance
.venv/bin/python scripts/serve_policy_TF.py --port 8000 --guidance_scale 1.5 \
  policy:checkpoint --policy.config=pi05_libero --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

`--no-install-project` works around a missing `README.md` in `openpi-cf`, so never use `uv run` here. Run these in a shell that has not sourced `scripts/env.sh`. Each server needs about 9 GB of VRAM. The checkpoint is about 12 GB.

Evaluate from the repo root, with `env.sh` sourced. Use `libero_cf_spatial`, `libero_cf_object`, `libero_cf_long` or `libero_cf_ood`:

```bash
mkdir -p results/paper && cd results/paper
$SIM_PY $CF/eval/main_cf.py --args.task_suite_name libero_cf_object --args.port 8000
```

Compare against Table III of the paper.

- The paper uses ω = 1.5 for π0.5. Upstream's README example uses 2.0.
- For the VA variant, get the [unconditioned checkpoint](https://huggingface.co/VLA-CF/pi05-libero-VA) and follow upstream's `serve_policy_VA.py` command.

## Test

```bash
CUDA_VISIBLE_DEVICES="" envs/policy/.venv/bin/python -m pytest policy_server/tests -q
```
