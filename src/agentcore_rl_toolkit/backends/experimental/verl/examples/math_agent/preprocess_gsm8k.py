"""Preprocess openai/gsm8k into the parquet format expected by fsdp_fft_sync_grpo.sh.

Output schema (per row) — a single column:
    payload: the exact ACR invoke payload, authored against the agent's own
             contract (rl_app.py expects "prompt" as the question string and
             "answer" as the ground truth). The agent loop forwards this dict
             verbatim, keeping the agent free of any trainer/dataset knowledge.

The chat-format ``prompt`` column verl's dataloader needs is synthesized at load
time by ``PayloadDataset`` (see ../../dataset.py) from ``payload["prompt"]`` —
dataset authors never write it.
"""

import argparse
import os
import re

import pandas as pd
from datasets import load_dataset

ANSWER_RE = re.compile(r"####\s*(.+?)\s*$")


def extract_final_answer(answer_field: str) -> str:
    match = ANSWER_RE.search(answer_field)
    if match is None:
        raise ValueError(f"No '#### <answer>' marker found in: {answer_field!r}")
    return match.group(1).replace(",", "").strip()


def build_split(split):
    return [
        {
            "payload": {
                "prompt": ex["question"],
                "answer": extract_final_answer(ex["answer"]),
            },
        }
        for ex in split
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to write gsm8k_agent_{train,test}.parquet into.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_dataset("openai/gsm8k", "main")

    for split_name, out_name in [("train", "gsm8k_agent_train.parquet"), ("test", "gsm8k_agent_test.parquet")]:
        df = pd.DataFrame(build_split(ds[split_name]), columns=["payload"])
        out_path = os.path.join(args.output_dir, out_name)
        df.to_parquet(out_path, index=False)
        print(f"wrote {out_path}: {len(df)} rows")


if __name__ == "__main__":
    main()
