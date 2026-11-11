"""Merge an exported LoRA adapter into the base Qwen3.5-9B snapshot, for serving with vLLM.

  .venv/bin/python scripts/merge_adapter.py exports/EXP-adapter exports/EXP --base BASE_MODEL_DIR

Adds (lora_alpha / r) * B @ A to each adapted weight, shard by shard, and copies every other tensor
and file unchanged, so the output has the base snapshot's exact layout. The merged weights are BF16, so
many small deltas round away; the published scores were measured on this merged model.
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "base_model.model."


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads((args.adapter / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    weight_map = json.loads((args.base / "model.safetensors.index.json").read_text())["weight_map"]
    lora = {}  # base weight name -> {"A": tensor, "B": tensor}
    with safe_open(str(args.adapter / "adapter_model.safetensors"), "pt") as f:
        for key in f.keys():  # noqa: SIM118 - safe_open is not iterable
            module, part = key.removeprefix(PREFIX).rsplit(".lora_", 1)  # part is "A.weight" or "B.weight"
            name = f"{module}.weight"
            if name not in weight_map:  # the adapter is text-only; the snapshot nests it under language_model
                name = name.replace("model.", "model.language_model.", 1)
            lora.setdefault(name, {})[part[0]] = f.get_tensor(key)
    assert all(name in weight_map and set(ab) == {"A", "B"} for name, ab in lora.items()), "unexpected adapter keys"

    args.out.mkdir(parents=True)
    for shard in sorted(set(weight_map.values())):
        tensors = {}
        with safe_open(str(args.base / shard), "pt") as f:
            for key in f.keys():  # noqa: SIM118
                weight = f.get_tensor(key)
                if key in lora:
                    delta = scale * (lora[key]["B"].float() @ lora[key]["A"].float())
                    weight = (weight.float() + delta).to(weight.dtype)
                tensors[key] = weight.contiguous()
        save_file(tensors, str(args.out / shard), metadata={"format": "pt"})
        print(f"wrote {shard}")
    for item in args.base.iterdir():  # config, tokenizer, chat template, processor and index files
        if not item.name.endswith(".safetensors"):
            shutil.copy(item, args.out / item.name)
    source = (args.adapter / "SOURCE.txt").read_text().strip()
    (args.out / "MERGED_FROM.json").write_text(json.dumps({"adapter": str(args.adapter), "checkpoint": source}) + "\n")
    print(f"merged {len(lora)} modules (scale {scale}) into {args.out}")


if __name__ == "__main__":
    main()
