import argparse
import asyncio

import websockets.asyncio.server as ws_server
from openpi_client import msgpack_numpy

from smolvla_cf.policy import SmolVLA


async def serve(policy, port):
    async def handler(websocket):
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack({}))
        async for message in websocket:
            await websocket.send(packer.pack(policy.infer(msgpack_numpy.unpackb(message))))

    async with ws_server.serve(handler, "0.0.0.0", port, compression=None, max_size=None) as server:
        await server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--null", default="none", choices=["none", "mask"])
    args = parser.parse_args()
    asyncio.run(serve(SmolVLA(args.scale, args.null), args.port))


if __name__ == "__main__":
    main()
