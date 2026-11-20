"""Per-step classifier-free guidance for SmolVLA's flow-matching sampler.

The committed ``policy.py`` mixes only the final action chunk
(``a = a_null + w * (a_cond - a_null)``). This module lifts that to a
guided velocity applied at *every* Euler step,

    v_tilde(tau) = v_null + w(tau) * (v_cond - v_null),            (Eq. 1)

which is the dependency shared by all three strength schedules in the
proposal (early-only, closed-form fitted, and the A-CFG adaptation).

Language enters SmolVLA only through the prefix KV-cache, so the
conditional and null branches differ *only* in their prefix (tokens /
masks). We therefore build two prefix caches once, then at each Euler
step evaluate both velocities from the shared ``x_t`` and mix them.

Correctness contract (verified in tests, CPU-only, no checkpoint needed
for the schedule math):
  * ``w(tau) == 1`` for all tau reproduces the ungided baseline exactly.
  * A single mixing step matches the final-chunk CAG mix in ``policy.py``.
  * ``fitted_w`` matches a brute-force argmin of the per-step velocity MSE.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

# A strength schedule maps a scalar flow time tau in [0, 1] (and the step
# index / total steps) to a guidance weight w. Returning 1.0 everywhere is
# the baseline; a constant c is standard CFG at scale c.
Schedule = Callable[[float, int, int], float]


def constant_schedule(w: float) -> Schedule:
    """Standard CFG: the same weight at every step."""
    return lambda tau, step, num_steps: w


def early_only_schedule(w: float, k: int) -> Schedule:
    """Guide with weight ``w`` for the first ``k`` steps, then w = 1.

    Tests the hypothesis that guidance matters most in the early, noisy
    steps while the action chunk is still being committed. Steps are
    0-indexed; ``k`` counts from the first (highest-noise) step.
    """
    return lambda tau, step, num_steps: w if step < k else 1.0


def tabulated_schedule(ws: list[float]) -> Schedule:
    """A schedule given as one weight per Euler step (e.g. a fitted w*).

    ``ws[step]`` is used directly; falls back to 1.0 if the step index is
    out of range, so a shorter table degrades to the baseline.
    """
    return lambda tau, step, num_steps: ws[step] if step < len(ws) else 1.0


@torch.no_grad()
def guided_sample_actions(
    model,
    images,
    img_masks,
    cond_tokens,
    cond_masks,
    null_tokens,
    null_masks,
    state,
    noise,
    schedule: Schedule,
):
    """Run the SmolVLA Euler loop with per-step CFG.

    ``model`` is a ``SmolVLAFlowMatching`` (``policy.model``). We mirror its
    own ``sample_actions`` loop (prefix cache + ``denoise_step`` + Euler
    step, ``dt = -1/num_steps``) but evaluate two branches per step.

    When ``schedule`` returns 1.0 at every step this is identical to a
    single conditional ``sample_actions`` call: v_tilde = v_cond.
    """
    bsize = state.shape[0]
    device = state.device

    def build_prefix(tokens, masks):
        prefix_embs, prefix_pad, prefix_att = model.embed_prefix(
            images, img_masks, tokens, masks, state=state
        )
        att_2d = _make_att_2d(model, prefix_pad, prefix_att)
        pos_ids = torch.cumsum(prefix_pad, dim=1) - 1
        _, kv = model.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=model.config.use_cache,
            fill_kv_cache=True,
        )
        return prefix_pad, kv

    cond_pad, cond_kv = build_prefix(cond_tokens, cond_masks)
    null_pad, null_kv = build_prefix(null_tokens, null_masks)

    num_steps = model.config.num_steps
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    step = 0
    while time >= -dt / 2:
        expanded_time = time.expand(bsize)
        v_cond = model.denoise_step(cond_pad, cond_kv, x_t, expanded_time)
        w = schedule(float(time), step, num_steps)
        if w == 1.0:
            v = v_cond  # baseline: skip the null branch entirely
        else:
            v_null = model.denoise_step(null_pad, null_kv, x_t, expanded_time)
            v = v_null + w * (v_cond - v_null)
        x_t = x_t + dt * v
        time = time + dt
        step += 1
    return x_t


def _make_att_2d(model, pad_masks, att_masks):
    # Imported lazily so the schedule helpers above stay importable on a
    # machine without lerobot (e.g. for the pure-math unit tests).
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    return make_att_2d_masks(pad_masks, att_masks)


# ---------------------------------------------------------------------------
# Closed-form fitted schedule  w*(tau)   (proposal Eq. 2)
# ---------------------------------------------------------------------------
def fitted_w(u, v_cond, v_null):
    """Least-squares optimal guidance weight at one flow time.

    Eq. 1 is linear in w, so the w minimising ||v_null + w (v_cond - v_null)
    - u||^2 over a batch of samples is, with numerator and denominator each
    summed over samples *before* dividing:

        w* = <u - v_null, v_cond - v_null> / ||v_cond - v_null||^2 .

    ``u = eps - a_star`` is the target (ground-truth) velocity toward a clean
    target chunk ``a_star`` from noise ``eps``. All tensors share shape
    ``(samples, chunk, action_dim)``. Returns a Python float.
    """
    diff = v_cond - v_null
    num = torch.sum((u - v_null) * diff)
    den = torch.sum(diff * diff)
    return float(num / den)


def fit_schedule(targets, noises, cond_vels, null_vels):
    """Fit one w* per Euler step from held-out demonstrations.

    Each argument is a list indexed by step (length = num_steps); entry
    ``step`` is a batched tensor ``(samples, chunk, action_dim)``:
      * ``targets[step]``   -- clean target chunks a_star
      * ``noises[step]``    -- the noise eps used to form x_tau
      * ``cond_vels[step]`` -- conditional velocities v_cond at that step
      * ``null_vels[step]`` -- null velocities v_null at that step
    Returns a list of floats (the fitted w* per step), ready to wrap with
    ``tabulated_schedule``.
    """
    ws = []
    for a_star, eps, v_cond, v_null in zip(targets, noises, cond_vels, null_vels):
        u = eps - a_star
        ws.append(fitted_w(u, v_cond, v_null))
    return ws


# ---------------------------------------------------------------------------
# A-CFG adapted to continuous actions
# ---------------------------------------------------------------------------
def acfg_renoise(x_t, time, model_sample_fn, k_draws: int = 4, frac: float = 0.5):
    """Build an A-CFG null input by re-noising the least-confident chunk steps.

    Adaptive CFG was defined for masked diffusion *language* models, which
    re-mask the tokens the model is least sure about. We adapt it to a
    continuous action chunk: draw ``k_draws`` noise samples, decode each to
    a clean-action estimate ``a_hat = x_t - tau * v``, and measure per-
    chunk-step disagreement (variance across draws). The ``frac`` most
    disagreeing chunk steps are re-noised back to the *current* level
    ``tau`` (so the model never sees mixed noise levels); confident steps
    keep ``x_t``.

    ``model_sample_fn(x, tau)`` returns the velocity v for a candidate x at
    time tau. Returns the adapted null input ``x_null`` (same shape as x_t).
    This is a reference implementation for the midway writeup; it is
    exercised by a CPU-only unit test on synthetic tensors.
    """
    tau = float(time)
    bsize, chunk, dim = x_t.shape
    estimates = []
    for _ in range(k_draws):
        eps = torch.randn_like(x_t)
        x_perturbed = tau * eps + (1.0 - tau) * x_t  # re-noise to level tau
        v = model_sample_fn(x_perturbed, time)
        estimates.append(x_perturbed - tau * v)  # a_hat
    stacked = torch.stack(estimates, dim=0)  # (k, bsize, chunk, dim)
    # Disagreement per chunk step: variance across draws, averaged over dims.
    disagreement = stacked.var(dim=0).mean(dim=-1)  # (bsize, chunk)
    n_renoise = max(1, int(round(frac * chunk)))
    idx = disagreement.topk(n_renoise, dim=1).indices  # (bsize, n_renoise)
    x_null = x_t.clone()
    eps = torch.randn_like(x_t)
    renoised = tau * eps + (1.0 - tau) * x_t
    for b in range(bsize):
        x_null[b, idx[b]] = renoised[b, idx[b]]
    return x_null
