"""Export a training run's final checkpoint as a Hugging Face PEFT LoRA adapter.

  .venv/bin/python scripts/export_adapter.py runs/EXP exports/EXP-adapter

SkyRL saves training checkpoints as Megatron shards, but a named sampler checkpoint as a PEFT adapter
archive in $SKYRL_STATE_DIR. So this loads the checkpoint into a new LoRA client on the running endpoint,
saves it as a sampler checkpoint and extracts the archive. Needs the SkyRL endpoint running.
"""

import json
import os
import sys
import tarfile
from pathlib import Path

import tinker
from dotenv import load_dotenv
from safetensors import safe_open


def main():
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    checkpoint = json.loads((run / "checkpoint.json").read_text())["path"]
    rank = json.loads((run / "config.json").read_text())["lora_rank"]
    state_dir = Path(os.environ.get("SKYRL_STATE_DIR", "~/skyrl-state")).expanduser()

    endpoint = f"http://127.0.0.1:{os.environ.get('TINKER_PORT', '18080')}"
    service = tinker.ServiceClient(base_url=endpoint, api_key="tml-dummy")
    base_model = service.get_server_capabilities().supported_models[0].model_name
    trainer = service.create_lora_training_client(base_model=base_model, rank=rank, seed=42)
    trainer.load_state(checkpoint).result()  # SkyRL only allows this as a new client's first request
    name = f"export-{out.name}"
    trainer.save_weights_for_sampler(name).result()
    archive = state_dir / "checkpoints" / trainer._guaranteed_model_id() / "sampler_weights" / f"{name}.tar.gz"

    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(out, filter="data")
    # A new adapter has all-zero lora_B weights; any non-zero one shows the checkpoint loaded.
    with safe_open(str(out / "adapter_model.safetensors"), "pt") as f:
        if not any(f.get_tensor(key).any() for key in f.keys() if "lora_B" in key):  # noqa: SIM118
            raise SystemExit(f"{out}: the adapter's lora_B weights are all zero, so {checkpoint} did not load")
    (out / "SOURCE.txt").write_text(checkpoint + "\n")
    print(f"exported {checkpoint} to {out}")


if __name__ == "__main__":
    main()
