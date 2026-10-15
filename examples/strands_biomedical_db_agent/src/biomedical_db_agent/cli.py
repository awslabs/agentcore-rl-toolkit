"""Run either database agent from a terminal."""

import argparse
import asyncio

from .config import ModelConfig
from .service import open_agents


async def run(name, query):
    config = ModelConfig.from_env(name)
    async with open_agents([name]) as agents:
        return await agents[name].answer(query, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent", choices=["opentargets", "gwas_catalog"])
    parser.add_argument("query")
    args = parser.parse_args()
    print(asyncio.run(run(args.agent, args.query)).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
