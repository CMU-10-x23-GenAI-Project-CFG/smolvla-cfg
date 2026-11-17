import argparse
import asyncio
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import websockets.asyncio.server as ws_server
from openpi.models import model as _model
from openpi.policies import policy_config
from openpi.training import config as _config
from openpi_client import msgpack_numpy

NEGATIVES = json.loads((pathlib.Path(__file__).parent.parent / "smolvla_cf" / "negatives.json").read_text())


class Pi05:
    def __init__(self):
        self.policy = policy_config.create_trained_policy(
            _config.get_config("pi05_libero"), "gs://openpi-assets/checkpoints/pi05_libero"
        )
        self.rng = jax.random.key(7)

    def prepare(self, obs, prompt=None):
        return self.policy._input_transform(dict(obs) if prompt is None else {**obs, "prompt": prompt})

    def sample(self, inputs, noise, rng):
        batch = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        actions = self.policy._sample_actions(rng, _model.Observation.from_dict(batch), **{**self.policy._sample_kwargs, "noise": noise})
        return np.asarray(actions[0])

    def infer(self, obs, null, scale):
        model = self.policy._model
        self.rng, noise_rng, rng = jax.random.split(self.rng, 3)
        noise = jax.random.normal(noise_rng, (1, model.action_horizon, model.action_dim))
        inputs = self.prepare(obs)
        actions = self.sample(inputs, noise, rng)
        if null != "none":
            if null == "negative":
                null_inputs = self.prepare(obs, NEGATIVES[str(obs["prompt"])])
            else:
                null_inputs = {**inputs, "tokenized_prompt": np.zeros_like(inputs["tokenized_prompt"]),
                               "tokenized_prompt_mask": np.zeros_like(inputs["tokenized_prompt_mask"])}
            null_actions = self.sample(null_inputs, noise, rng)
            actions = null_actions + scale * (actions - null_actions)
        return self.policy._output_transform({"state": np.asarray(inputs["state"]), "actions": actions})


async def serve(policy, jobs):
    def handler_for(null, scale):
        async def handler(websocket):
            packer = msgpack_numpy.Packer()
            await websocket.send(packer.pack({}))
            async for message in websocket:
                await websocket.send(packer.pack(policy.infer(msgpack_numpy.unpackb(message), null, scale)))

        return handler

    servers = [await ws_server.serve(handler_for(null, scale), "0.0.0.0", port, compression=None, max_size=None)
               for port, null, scale in jobs]
    await asyncio.gather(*(server.serve_forever() for server in servers))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", nargs="+", required=True, help="port:null:scale, null in none/mask/negative")
    args = parser.parse_args()
    jobs = [(int(port), null, float(scale)) for port, null, scale in (job.split(":") for job in args.jobs)]
    asyncio.run(serve(Pi05(), jobs))


if __name__ == "__main__":
    main()
