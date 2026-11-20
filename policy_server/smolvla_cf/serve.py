import argparse
import asyncio
import json
import pathlib

import websockets.asyncio.server as ws_server
from openpi_client import msgpack_numpy

from smolvla_cf import guidance
from smolvla_cf.policy import SmolVLA


async def serve(policy, port):
    async def handler(websocket):
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack({}))
        async for message in websocket:
            await websocket.send(packer.pack(policy.infer(msgpack_numpy.unpackb(message))))

    async with ws_server.serve(handler, "0.0.0.0", port, compression=None, max_size=None) as server:
        await server.serve_forever()


def build_schedule(args):
    """Translate CLI flags into a guidance.Schedule, or None for the
    final-chunk mix. ``--schedule constant`` with the default --scale is
    equivalent to the legacy behaviour but routed through the per-step
    sampler, which is what the correctness checks compare against."""
    if args.schedule == "none":
        return None
    if args.schedule == "constant":
        return guidance.constant_schedule(args.scale)
    if args.schedule == "early":
        return guidance.early_only_schedule(args.scale, args.early_k)
    if args.schedule == "fitted":
        ws = json.loads(pathlib.Path(args.fitted_file).read_text())
        return guidance.tabulated_schedule(ws)
    raise ValueError(args.schedule)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--null", default="none", choices=["none", "mask", "negative"])
    parser.add_argument("--schedule", default="none",
                        choices=["none", "constant", "early", "fitted"],
                        help="per-step strength schedule; 'none' uses the final-chunk mix")
    parser.add_argument("--early-k", type=int, default=4,
                        help="number of early Euler steps to guide (--schedule early)")
    parser.add_argument("--fitted-file", default=None,
                        help="JSON list of per-step w* weights (--schedule fitted)")
    args = parser.parse_args()
    asyncio.run(serve(SmolVLA(args.scale, args.null, build_schedule(args)), args.port))


if __name__ == "__main__":
    main()
