# Overnight run — Praneet's guidance screens

Thanks for running these. Everything is on branch `praneet/per-step-guidance`
(PR #1). Needs the Linux + NVIDIA box. Roughly 4–5 h, mostly unattended.

Run the steps in order. If a step fails, stop and send me the error — don't
skip ahead.

---

## 0. Get the code

```bash
git fetch origin
git checkout praneet/per-step-guidance
```

## 1. One-time setup (skip if the envs already exist)

```bash
scripts/setup_sim.sh
scripts/setup_policy.sh
```

Then, **in every terminal below**, from the repo root:

```bash
source scripts/env.sh
```

## 2. Sanity check — does my sampler match the real checkpoint?

This is the one test I couldn't run (no GPU on my end). It must pass before
the screens mean anything.

```bash
CUDA_VISIBLE_DEVICES="" $POLICY_PY -m pytest policy_server/tests -q
```

Expected: all pass. If `test_policy_guidance.py` fails, **stop** and send me
the output — don't run the screens.

---

## 3. The screens (2 terminals each: server, then evaluator)

All screens are 10 trials, on `libero_cf_object`. Each writes to its own
results folder so nothing overwrites.

### 3a. Baseline (reference row)

```bash
# Terminal 1 — server
$POLICY_PY -m smolvla_cf.serve --port 8000 --null negative --scale 2.0

# Terminal 2 — evaluator
mkdir -p results/base && cd results/base
$SIM_PY $CF/eval/main_cf.py --args.task_suite_name libero_cf_object \
  --args.port 8000 --args.resize_size 256 --args.num_trials_per_task 10
```

Stop the server (Ctrl-C in Terminal 1) before starting the next one.

### 3b. Early-only schedule — sweep k = 2, 4, 6, 8

For each `k`, change `--early-k` on the server and the folder name:

```bash
# Terminal 1 — server (set K to 2, then 4, then 6, then 8)
$POLICY_PY -m smolvla_cf.serve --port 8000 --null negative --scale 2.0 \
  --schedule early --early-k 2

# Terminal 2 — evaluator (match the folder to the k)
mkdir -p results/early_k2 && cd results/early_k2
$SIM_PY $CF/eval/main_cf.py --args.task_suite_name libero_cf_object \
  --args.port 8000 --args.resize_size 256 --args.num_trials_per_task 10
```

Repeat for `--early-k 4` → `results/early_k4`, `6` → `results/early_k6`,
`8` → `results/early_k8`. Restart the server between each.

### 3c. Fitted schedule (only if you have the weights file)

If I've sent you `ws.json`, run:

```bash
# Terminal 1 — server
$POLICY_PY -m smolvla_cf.serve --port 8000 --null negative \
  --schedule fitted --fitted-file ws.json

# Terminal 2 — evaluator
mkdir -p results/fitted && cd results/fitted
$SIM_PY $CF/eval/main_cf.py --args.task_suite_name libero_cf_object \
  --args.port 8000 --args.resize_size 256 --args.num_trials_per_task 10
```

If I haven't sent `ws.json` yet, **skip 3c** — I'll fit it from your run
outputs afterward.

### 3d. A-CFG — skip for now

A-CFG isn't wired as a runnable server mode yet (it's implemented and
unit-tested, but I want to verify it against the checkpoint before a real
run). Please skip it tonight; I'll follow up.

---

## 4. What to send me back

For each `results/<name>` folder, I need:

1. The last ~30 lines of `client.log` (it prints `touches` / `successes`
   counts per task).
2. The output of this, which must be **0** for every folder:

   ```bash
   grep -c "Caught exception" results/*/client.log
   ```

   If any folder is non-zero, that run hit server errors — tell me which.

The simplest thing: just zip the `results/` folder (logs only, no videos)
and send it.

```bash
zip -r results_praneet.zip results -x '*/rollouts/*'
```

---

## Notes / gotchas (from the README)

- `--args.resize_size 256` is **required** — it matches the checkpoint's
  training images.
- Ignore the `Total success rate` line; it also counts the distractor. The
  numbers I need are the per-task `touches` / `successes` counts.
- If a run dies partway, resume in a **new** folder with
  `--args.starting_task_id N` and redo task N from its first trial.
- Each config is ~25–35 min at 10 trials (faster on your card). The early-k
  sweep is the long pole (4 configs).
