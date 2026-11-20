import argparse
import asyncio
import json
import pathlib
import types

import einops
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.shared import nnx_utils

from serve import Pi05, serve

CONTROLS = json.loads((pathlib.Path(__file__).parent / "controls.json").read_text())
NULLS = ["mask", "negative", "empty", "wrong", "paraphrase", "generic"]


def guided_sample_actions(self, rng, observation, noise, scale, guided_steps, num_steps=10):
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
        observation, mask_language_tokens=self.mask_language_tokens, mask_vision_tokens=self.mask_vision_tokens
    )
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def velocity(x_t, time):
        x_b = jnp.broadcast_to(x_t, (batch_size, *x_t.shape[1:]))
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_b, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    def step(carry):
        x_t, time, i = carry
        v = velocity(x_t, time)
        v_c, v_n = v[:1], v[1:]
        v_g = jnp.where(i < guided_steps, v_n + scale * (v_c - v_n), v_c)
        return x_t + dt * v_g, time + dt, i + 1

    def cond(carry):
        return carry[1] >= -dt / 2

    x_0, _, _ = jax.lax.while_loop(cond, step, (noise, 1.0, 0))
    return x_0


class Pi05Step(Pi05):
    def __init__(self):
        super().__init__()
        self.guided = nnx_utils.module_jit(types.MethodType(guided_sample_actions, self.policy._model))

    def null_inputs(self, obs, inputs, null):
        if null == "mask":
            return {**inputs, "tokenized_prompt": np.zeros_like(inputs["tokenized_prompt"]),
                    "tokenized_prompt_mask": np.zeros_like(inputs["tokenized_prompt_mask"])}
        return self.prepare(obs, CONTROLS[str(obs["prompt"])][null])

    def draw(self):
        model = self.policy._model
        self.rng, noise_rng, rng = jax.random.split(self.rng, 3)
        return jax.random.normal(noise_rng, (1, model.action_horizon, model.action_dim)), rng

    def infer_step(self, obs, null, scale, guided_steps):
        noise, rng = self.draw()
        inputs = self.prepare(obs)
        null_inputs = self.null_inputs(obs, inputs, null)
        pair = jax.tree.map(lambda c, n: jnp.stack([jnp.asarray(c), jnp.asarray(n)]), inputs, null_inputs)
        actions = self.guided(rng, _model.Observation.from_dict(pair), noise, jnp.float32(scale),
                              jnp.int32(guided_steps), **self.policy._sample_kwargs)
        return self.policy._output_transform({"state": np.asarray(inputs["state"]), "actions": np.asarray(actions[0])})

    def infer_final(self, obs, null, scale):
        noise, rng = self.draw()
        inputs = self.prepare(obs)
        actions = self.sample(inputs, noise, rng)
        null_actions = self.sample(self.null_inputs(obs, inputs, null), noise, rng)
        actions = null_actions + scale * (actions - null_actions)
        return self.policy._output_transform({"state": np.asarray(inputs["state"]), "actions": actions})

    def infer(self, obs, null, scale):
        base, step, k = null.partition("-step")
        if null == "none":
            return super().infer(obs, null, scale)
        if step:
            return self.infer_step(obs, base, scale, int(k) if k else self.policy._sample_kwargs.get("num_steps", 10))
        return self.infer_final(obs, null, scale)


def main():
    parser = argparse.ArgumentParser()
    nulls = "/".join(NULLS)
    parser.add_argument("--jobs", nargs="+", required=True,
                        help=f"port:null:scale, null in none or <n>, <n>-step, <n>-step<k> with <n> in {nulls}")
    args = parser.parse_args()
    jobs = [(int(port), null, float(scale)) for port, null, scale in (job.split(":") for job in args.jobs)]
    for _, null, _ in jobs:
        assert null == "none" or null.partition("-step")[0] in NULLS, null
    asyncio.run(serve(Pi05Step(), jobs))


if __name__ == "__main__":
    main()
