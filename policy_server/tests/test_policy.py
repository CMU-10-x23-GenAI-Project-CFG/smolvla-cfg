import json
import pathlib

import numpy as np
import pytest
import torch

from smolvla_cf.policy import NEGATIVES, SmolVLA

CONDITIONS = pathlib.Path(__file__).parents[2] / "third_party/libero-cf/libero/libero/conditions/libero_cf_object.json"


@pytest.fixture(scope="module")
def model():
    return SmolVLA()


@pytest.fixture(scope="module")
def obs():
    rng = np.random.default_rng(0)
    return {
        "observation/image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "observation/state": rng.normal(size=8).astype(np.float32),
        "prompt": "pick up the cream cheese and place it in the basket",
    }


def test_matches_lerobot(model, obs):
    torch.manual_seed(0)
    ours = model.infer(obs)["actions"]
    torch.manual_seed(0)
    reference = model.policy.predict_action_chunk(model.to_batch(obs), noise=torch.randn(1, 50, 32))
    assert ours.shape == (50, 7)
    np.testing.assert_allclose(ours, reference[0].numpy(), atol=1e-6)


def run(model, obs, scale, null):
    model.scale, model.null = scale, null
    torch.manual_seed(0)
    return model.infer(obs)["actions"]


@pytest.mark.parametrize("null", ["mask", "negative"])
def test_guidance_formula(model, obs, null):
    conditional = run(model, obs, 1.0, "none")
    unconditional = run(model, obs, 0.0, null)
    np.testing.assert_allclose(run(model, obs, 1.0, null), conditional, atol=1e-5)
    np.testing.assert_allclose(run(model, obs, 2.0, null), 2 * conditional - unconditional, atol=1e-5)
    assert np.abs(conditional - unconditional).max() > 1e-3


def test_negatives_name_the_benchmark_distractor():
    conditions = json.loads(CONDITIONS.read_text())
    assert len(conditions) == len(NEGATIVES) == 10
    for key, (_, distractor) in conditions.items():
        instruction = key.split("/")[1][3:-5].replace("_", " ")
        assert distractor.split()[1].removesuffix("_1").replace("_", " ") in NEGATIVES[instruction]
