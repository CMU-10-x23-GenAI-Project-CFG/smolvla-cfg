"""CPU-only tests for the per-step guidance schedules and A-CFG adaptation.

These exercise the math in guidance.py on synthetic tensors and do NOT load
the SmolVLA checkpoint, so they run fast anywhere (no GPU, no lerobot import
beyond the schedule helpers). The checkpoint-backed equivalence check
(w == 1 reproduces the baseline sampler end to end) lives in
test_policy_guidance.py and is skipped when the model cannot be loaded.
"""

import torch

from smolvla_cf import guidance


def test_constant_schedule_is_flat():
    s = guidance.constant_schedule(2.0)
    assert [s(t, i, 10) for i, t in enumerate([1.0, 0.5, 0.0])] == [2.0, 2.0, 2.0]


def test_early_only_guides_then_relaxes_to_one():
    s = guidance.early_only_schedule(3.0, k=4)
    weights = [s(1.0 - i / 10, i, 10) for i in range(10)]
    assert weights[:4] == [3.0, 3.0, 3.0, 3.0]
    assert weights[4:] == [1.0] * 6  # w == 1 means "baseline" past step k


def test_tabulated_schedule_reads_per_step_and_defaults_to_one():
    s = guidance.tabulated_schedule([1.2, 0.8])
    assert s(1.0, 0, 10) == 1.2
    assert s(0.9, 1, 10) == 0.8
    assert s(0.0, 5, 10) == 1.0  # out of range -> baseline


def test_fitted_w_recovers_known_weight():
    # Construct v_null, v_cond and a target u that is exactly v_null + w(v_cond
    # - v_null) for a known w; the closed form must return that w.
    torch.manual_seed(0)
    v_null = torch.randn(8, 50, 32)
    v_cond = torch.randn(8, 50, 32)
    w_true = 1.7
    u = v_null + w_true * (v_cond - v_null)
    assert abs(guidance.fitted_w(u, v_cond, v_null) - w_true) < 1e-5


def test_fitted_w_matches_bruteforce_argmin_under_noise():
    # With noise, the closed form must match a dense grid search of the MSE.
    torch.manual_seed(1)
    v_null = torch.randn(16, 50, 32)
    v_cond = torch.randn(16, 50, 32)
    u = v_null + 0.9 * (v_cond - v_null) + 0.05 * torch.randn(16, 50, 32)
    w_closed = guidance.fitted_w(u, v_cond, v_null)

    def mse(w):
        return torch.mean((v_null + w * (v_cond - v_null) - u) ** 2).item()

    grid = torch.linspace(w_closed - 0.5, w_closed + 0.5, 2001)
    w_grid = grid[min(range(len(grid)), key=lambda j: mse(grid[j].item()))].item()
    assert abs(w_closed - w_grid) < 2e-3


def test_fit_schedule_returns_one_weight_per_step():
    torch.manual_seed(2)
    steps = 10
    targets = [torch.randn(4, 50, 32) for _ in range(steps)]
    noises = [torch.randn(4, 50, 32) for _ in range(steps)]
    cond = [torch.randn(4, 50, 32) for _ in range(steps)]
    null = [torch.randn(4, 50, 32) for _ in range(steps)]
    ws = guidance.fit_schedule(targets, noises, cond, null)
    assert len(ws) == steps
    assert all(isinstance(w, float) for w in ws)


def test_acfg_renoise_keeps_shape_and_changes_only_some_steps():
    # A-CFG should re-noise roughly `frac` of the chunk steps and leave the
    # rest untouched. Use a velocity fn that makes a few chunk steps far more
    # uncertain than the others, and check those are the ones that change.
    torch.manual_seed(3)
    x_t = torch.randn(1, 50, 32)
    noisy_steps = {3, 17, 42}

    def model_sample_fn(x, tau):
        v = torch.zeros_like(x)
        for s in noisy_steps:
            v[:, s] = torch.randn(32)  # high variance across draws at these steps
        return v

    x_null = guidance.acfg_renoise(x_t, torch.tensor(0.5), model_sample_fn,
                                   k_draws=4, frac=0.1)
    assert x_null.shape == x_t.shape
    changed = (x_null != x_t).any(dim=-1)[0]  # (chunk,)
    assert changed.sum().item() == max(1, int(round(0.1 * 50)))
