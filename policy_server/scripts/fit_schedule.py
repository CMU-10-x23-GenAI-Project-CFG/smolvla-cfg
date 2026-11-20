"""Fit a per-step guidance schedule w*(tau) from saved velocity samples.

Reads a .npz with four arrays, each shaped (num_steps, samples, chunk, dim):
    targets   -- clean target action chunks a_star
    noises    -- the noise eps used to form x_tau at each step
    cond_vels -- conditional velocities v_cond
    null_vels -- null velocities v_null
and writes a JSON list of one w* per Euler step (proposal Eq. 2), ready for
``serve.py --schedule fitted --fitted-file ws.json``.

Usage:
    python scripts/fit_schedule.py samples.npz ws.json
"""

import json
import sys

import numpy as np
import torch

from smolvla_cf import guidance


def main(npz_path, out_path):
    data = np.load(npz_path)
    to_list = lambda key: [torch.tensor(x) for x in data[key]]  # per step
    ws = guidance.fit_schedule(
        to_list("targets"), to_list("noises"), to_list("cond_vels"), to_list("null_vels")
    )
    with open(out_path, "w") as f:
        json.dump(ws, f)
    print(f"wrote {len(ws)} per-step weights to {out_path}: "
          f"{[round(w, 3) for w in ws]}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python scripts/fit_schedule.py samples.npz ws.json")
    main(sys.argv[1], sys.argv[2])
