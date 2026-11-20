from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import modal

REPO = "/home/orio/GenAI"
CF = f"{REPO}/third_party/libero-cf"
OPENPI = f"{CF}/openpi-cf"
SIM = f"{REPO}/envs/sim/.venv/bin/python"
PI05 = [f"{OPENPI}/.venv/bin/python", f"{REPO}/policy_server/pi05_cf/serve.py"]
PI05_STEP = [f"{OPENPI}/.venv/bin/python", f"{REPO}/policy_server/pi05_cf/serve_step.py"]
VOL = "/vol"
GPU = os.environ.get("CF_GPU", "L4")
SKIP = ["**/.venv", "**/rollouts", "**/__pycache__", "**/*.pyc", "**/.git", ".git", "openvla-oft-cf", "libero/datasets"]

def clear_execstack():
    import struct

    for path in Path(f"{REPO}/envs/sim/.venv/lib").rglob("*.so*"):
        with open(path, "r+b") as f:
            head = f.read(64)
            if head[:4] != b"\x7fELF" or head[4] != 2:
                continue
            phoff, = struct.unpack_from("<Q", head, 32)
            phentsize, phnum = struct.unpack_from("<HH", head, 54)
            for i in range(phnum):
                f.seek(phoff + i * phentsize)
                p_type, p_flags = struct.unpack("<II", f.read(8))
                if p_type == 0x6474E551 and p_flags & 1:
                    f.seek(phoff + i * phentsize + 4)
                    f.write(struct.pack("<I", p_flags & ~1))
                    print(f"cleared execstack: {path}")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "build-essential", "libegl1", "libgl1", "libglib2.0-0", "libosmesa6",
                 "ffmpeg", "procps")
    .pip_install("uv==0.13.6")
    .env({"NVIDIA_DRIVER_CAPABILITIES": "all", "GIT_LFS_SKIP_SMUDGE": "1", "UV_LINK_MODE": "copy"})
    .add_local_dir(CF, CF, ignore=SKIP, copy=True)
    .add_local_dir(f"{REPO}/scripts", f"{REPO}/scripts", copy=True)
    .run_commands(f"bash {REPO}/scripts/setup_sim.sh")
    .run_function(clear_execstack)
    .run_commands(f"cd {OPENPI} && touch -a README.md packages/openpi-client/README.md && uv sync --frozen")
    .run_commands(
        f"{OPENPI}/.venv/bin/python -c \"from openpi.shared import download; "
        "download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero'); "
        "download.maybe_download('gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'})\""
    )
    .add_local_dir(f"{REPO}/policy_server", f"{REPO}/policy_server", ignore=["**/__pycache__", "**/.venv"])
)

app = modal.App("genai-cf")
vol = modal.Volume.from_name("genai-cf", create_if_missing=True)


def parse(config):
    if config == "baseline":
        return "none", "1"
    null, scale = config.rsplit("_w", 1)
    return null, scale


def finished(log):
    return log.exists() and "Final results for task" in log.read_text(errors="ignore")


def prepare():
    Path(VOL, "results").mkdir(parents=True, exist_ok=True)
    link = Path(REPO, "results")
    if not link.exists():
        link.symlink_to(Path(VOL, "results"))


def committer(stop):
    while not stop.wait(120):
        try:
            vol.commit()
        except Exception as e:
            print(f"volume commit failed: {e}", flush=True)


def nvidia_smi(field):
    return subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def kill(p):
    if p.poll() is None:
        os.killpg(p.pid, 15)
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9)
            p.wait()


@app.function(image=image, gpu=GPU, cpu=4, memory=16384, timeout=4 * 3600, volumes={VOL: vol},
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def cell(root: str, config: str, task: int, trials: int, suite: str = "libero_cf_object", resize: int = 256) -> dict:
    prepare()
    vol.reload()
    out = Path(REPO, "results", root, config, f"attempt_t{task:02d}")
    log = out / "client.log"
    if finished(log):
        return {"config": config, "task": task, "status": "already done"}
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    null, scale = parse(config)
    env = dict(os.environ, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", MUJOCO_GL="egl",
               PYTHONPATH=CF, LIBERO_CONFIG_PATH=f"{REPO}/.libero")
    gpu = nvidia_smi("name")
    (out / "cell.txt").write_text(f"{time.strftime('%F %T')} {gpu} config={config} null={null} scale={scale} "
                                  f"task={task} trials={trials} suite={suite} resize={resize}\n")
    server_cmd = PI05 if null in ("none", "mask", "negative") else PI05_STEP
    server = subprocess.Popen([*server_cmd, "--jobs", f"8000:{null}:{scale}"], stdout=open(out / "server.log", "w"),
                              stderr=subprocess.STDOUT, cwd=out, start_new_session=True,
                              env=dict(env, PYTHONPATH=f"{OPENPI}/src", XLA_PYTHON_CLIENT_PREALLOCATE="false"))
    client = subprocess.Popen([SIM, "-X", "faulthandler", f"{CF}/eval/main_cf.py", "--args.task_suite_name", suite,
                               "--args.port", "8000", "--args.resize_size", str(resize),
                               "--args.num_trials_per_task", str(trials), "--args.starting_task_id", str(task)],
                              stdout=open(log, "w"), stderr=subprocess.STDOUT, cwd=out, start_new_session=True, env=env)
    stop = threading.Event()
    threading.Thread(target=committer, args=(stop,), daemon=True).start()
    t0 = time.time()
    status = None
    while status is None:
        time.sleep(10)
        if finished(log):
            time.sleep(3)
            status = "ok"
        elif server.poll() is not None:
            status = f"server exited {server.returncode}"
        elif client.poll() is not None:
            status = f"client exited {client.returncode}"
        elif "Caught exception" in log.read_text(errors="ignore"):
            status = "caught exception"
    vram = nvidia_smi("memory.used")
    kill(client)
    kill(server)
    minutes = (time.time() - t0) / 60
    with open(out / "cell.txt", "a") as f:
        f.write(f"{time.strftime('%F %T')} {status} after {minutes:.1f} min, vram {vram}\n")
    stop.set()
    vol.commit()
    if status != "ok":
        raise RuntimeError(f"{config} task {task}: {status} (see results/{root}/{config}/attempt_t{task:02d})")
    return {"config": config, "task": task, "status": status, "minutes": round(minutes, 1), "gpu": gpu, "vram": vram}


@app.function(image=image, cpu=1, memory=2048, timeout=24 * 3600, volumes={VOL: vol})
def sweep_remote(root: str, configs: list[str], tasks: list[int], trials: int, suite: str, resize: int) -> None:
    prepare()
    report = Path(REPO, "results", root, "cells.log")
    report.parent.mkdir(parents=True, exist_ok=True)
    args = [(root, c, t, trials, suite, resize) for c in configs for t in tasks]
    with open(report, "a") as f:
        f.write(f"{time.strftime('%F %T')} START {len(args)} cells configs={configs} tasks={tasks} "
                f"trials={trials} suite={suite} resize={resize}\n")
    vol.commit()
    for (_, c, t, _, _, _), result in zip(args, cell.starmap(args, return_exceptions=True)):
        with open(report, "a") as f:
            f.write(f"{time.strftime('%F %T')} {c} t{t:02d} {result}\n")
        vol.commit()
    with open(report, "a") as f:
        f.write(f"{time.strftime('%F %T')} FINISHED\n")
    vol.commit()


@app.function(image=image, gpu=GPU, cpu=4, memory=32768, timeout=3600)
def test_remote(expr: str) -> int:
    env = dict(os.environ, PYTHONPATH=f"{OPENPI}/src:{REPO}/policy_server/pi05_cf",
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    return subprocess.run([f"{OPENPI}/.venv/bin/python", "-m", "pytest", "-s", "-v", "-p", "no:cacheprovider",
                           "tests/test_pi05.py", "-k", expr], cwd=f"{REPO}/policy_server", env=env).returncode


@app.local_entrypoint()
def test(expr: str = ""):
    code = test_remote.remote(expr)
    print(f"pytest exit {code}")
    if code:
        raise SystemExit(code)


@app.local_entrypoint()
def smoke(config: str = "baseline", task: int = 0, trials: int = 2, root: str = "modal_smoke",
          suite: str = "libero_cf_object"):
    print(cell.remote(root, config, task, trials, suite))


@app.local_entrypoint()
def sweep(root: str, configs: str, trials: int, tasks: str = "0-9", suite: str = "libero_cf_object", resize: int = 256):
    first, last = map(int, tasks.split("-"))
    names = configs.split(",")
    for name in names:
        parse(name)
    print(f"{root}: {names} tasks {first}-{last} x {trials} trials on {GPU}; "
          f"progress in Volume genai-cf:/results/{root}/cells.log")
    sweep_remote.remote(root, names, list(range(first, last + 1)), trials, suite, resize)
