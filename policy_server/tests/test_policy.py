import numpy as np
import pytest
import torch

from smolvla_cf.policy import SmolVLA


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
