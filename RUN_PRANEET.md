# Overnight run — Praneet's guidance screens

Thanks for running these! Just run the commands top to bottom on the Linux +
NVIDIA box. ~4–5 h, mostly unattended. When done, send me the `results/`
folder (step 5).

If anything errors out, paste me the error and stop there.

---

## 1. Get the code

```bash
git fetch origin
git checkout praneet/per-step-guidance
```

## 2. Setup (one time)

```bash
scripts/setup_sim.sh
scripts/setup_policy.sh
```

## 3. Sanity test

Run this from the repo root:

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES="" $POLICY_PY -m pytest policy_server/tests -q
```

## 4. The screens

Five runs. Each one is a **server** command (Terminal 1) and an **evaluator**
command (Terminal 2). Run both terminals with `source scripts/env.sh` first.
After each run finishes, Ctrl-C the server before starting the next.

**Run 1 — baseline**

```bash
# Terminal 1
$POLICY_PY -m smolvla_cf.serve --port 8000 --null negative --scale 2.0
```
```bash
# Terminal 2
mkdir -p results/base && cd results/base
$SIM_PY $CF/eval/main_cf.py --args.task_suite_name libero_cf_object --args.port 8000 --args.resize_size 256 --args.num_trials_per_task 10
```

**Run 2 — early k=2**

```bash
# Terminal 1
$POLICY_PY -m smolvla_cf.serve --port 8000 --null negative --scale 2.0 --schedule early --early-k 2
```
```bash
# Terminal 2
mkdir -p results/early_k2 && cd results/early_k2
$SIM_PY $CF/eval/main_cf.py --args.task_suite_name libero_cf_object --args.port 8000 --args.resize_size 256 --args.num_trials_per_task 10
```

**Run 3 — early k=4**

```bash
# Terminal 1
$POLICY_PY -m smolvla_cf.serve --port 8000 --null negative --scale 2.0 --schedule early --early-k 4
```
```bash
# Terminal 2
mkdir -p results/early_k4 && cd results/early_k4
$SIM_PY $CF/eval/main_cf.py --args.task_suite_name libero_cf_object --args.port 8000 --args.resize_size 256 --args.num_trials_per_task 10
```

**Run 4 — early k=6**

```bash
# Terminal 1
$POLICY_PY -m smolvla_cf.serve --port 8000 --null negative --scale 2.0 --schedule early --early-k 6
```
```bash
# Terminal 2
mkdir -p results/early_k6 && cd results/early_k6
$SIM_PY $CF/eval/main_cf.py --args.task_suite_name libero_cf_object --args.port 8000 --args.resize_size 256 --args.num_trials_per_task 10
```

**Run 5 — early k=8**

```bash
# Terminal 1
$POLICY_PY -m smolvla_cf.serve --port 8000 --null negative --scale 2.0 --schedule early --early-k 8
```
```bash
# Terminal 2
mkdir -p results/early_k8 && cd results/early_k8
$SIM_PY $CF/eval/main_cf.py --args.task_suite_name libero_cf_object --args.port 8000 --args.resize_size 256 --args.num_trials_per_task 10
```

## 5. Send back

```bash
zip -r results_praneet.zip results -x '*/rollouts/*'
```

Send me `results_praneet.zip`. That's it — I'll take it from there.
