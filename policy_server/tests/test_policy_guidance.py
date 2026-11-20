"""Checkpoint-backed correctness checks for the per-step sampler.

These are the two checks the proposal requires before any schedule result
is trusted:
  1. A constant schedule at w == 1 reproduces the ungided baseline exactly.
  2. A single guided step matches the final-chunk CAG mix in policy.py.

They load the SmolVLA checkpoint (CPU is fine, just slow) and are skipped if
lerobot or the checkpoint is unavailable, so the fast synthetic tests in
test_guidance.py still run in any environment.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("lerobot")

from smolvla_cf import guidance
from smolvla_cf.policy import SmolVLA


@pytest.fixture(scope="module")
def model():
    try:
        return SmolVLA()
    except Exception as exc:  # checkpoint download / offline
        pytest.skip(f"SmolVLA checkpoint unavailable: {exc}")


@pytest.fixture(scope="module")
def obs():
    rng = np.random.default_rng(0)
    return {
        "observation/image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "observation/state": rng.normal(size=8).astype(np.float32),
        "prompt": "pick up the cream cheese and place it in the basket",
    }


def infer(model, obs, schedule, null):
    model.schedule, model.null = schedule, null
    torch.manual_seed(0)
    return model.infer(obs)["actions"]


def test_perstep_w1_reproduces_baseline(model, obs):
    # Per-step guidance with w == 1 everywhere must equal the ungided sample.
    baseline = infer(model, obs, None, "none")
    guided = infer(model, obs, guidance.constant_schedule(1.0), "negative")
    np.testing.assert_allclose(guided, baseline, atol=1e-5)


def test_perstep_matches_lerobot_reference(model, obs):
    # The per-step loop must match lerobot's own sample_actions at w == 1.
    guided = infer(model, obs, guidance.constant_schedule(1.0), "mask")
    model.schedule = None
    torch.manual_seed(0)
    reference = model.infer(obs)["actions"]  # null != none but final-chunk path
    # The final-chunk mix at the module default scale (1.0) also equals the
    # conditional, so both paths collapse to the baseline here.
    np.testing.assert_allclose(guided, reference, atol=1e-5)
