"""Preprocess openai/gsm8k into the parquet format expected by run_agentcore_grpo.sh.

Output schema (per row):
    prompt: str - the question text (consumed by rl_app.py via payload["prompt"])
    answer: str - the final numeric answer extracted from the "#### N" marker
                  in the gold solution (consumed by rl_app.py via payload["answer"]
                  and passed to GSM8KReward as ground_truth)
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from datasets import load_dataset

ANSWER_RE = re.compile(r"####\s*(.+?)\s*$")


def extract_final_answer(answer_field: str) -> str:
    match = ANSWER_RE.search(answer_field)
    if match is None:
        raise ValueError(f"No '#### <answer>' marker found in: {answer_field!r}")
    return match.group(1).replace(",", "").strip()


def build_split(split):
    return [{"prompt": ex["question"], "answer": extract_final_answer(ex["answer"])} for ex in split]


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

    str_dtype = pd.StringDtype(storage="pyarrow", na_value=np.nan)
    columns = ["prompt", "answer"]

    for split_name, out_name in [("train", "gsm8k_agent_train.parquet"), ("test", "gsm8k_agent_test.parquet")]:
        rows = build_split(ds[split_name])
        df = pd.DataFrame(rows, columns=columns)
        df = df.astype({col: str_dtype for col in df.columns})
        out_path = os.path.join(args.output_dir, out_name)
        df.to_parquet(out_path, index=False)
        print(f"wrote {out_path}: {len(df)} rows")


if __name__ == "__main__":
    main()
