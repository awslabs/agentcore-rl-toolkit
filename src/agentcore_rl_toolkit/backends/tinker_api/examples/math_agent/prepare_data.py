"""Write GSM8K invocation payloads for examples/strands_math_agent."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    args = parser.parse_args()
    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    if args.split == "train":
        dataset = dataset.shuffle(seed=42)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for row in dataset:
            answer = row["answer"].rsplit("####", 1)[1].strip().replace(",", "")
            stream.write(json.dumps({"prompt": row["question"], "answer": answer}) + "\n")


if __name__ == "__main__":
    main()
