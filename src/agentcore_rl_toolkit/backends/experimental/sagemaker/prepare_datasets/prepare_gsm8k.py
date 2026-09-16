"""Preprocess openai/gsm8k into the parquet format.
Require datasets be installed by `pip install datasets`.
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
