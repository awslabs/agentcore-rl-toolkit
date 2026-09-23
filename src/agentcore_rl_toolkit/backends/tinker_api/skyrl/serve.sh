#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source environment.sh
mkdir -p "$TMPDIR" "$SKYRL_STATE_DIR"
# Prevent two launchers from starting servers against the same database.
exec 9>"$SKYRL_STATE_DIR/endpoint.lock"
flock -n 9 || { echo "An endpoint is already running on this instance." >&2; exit 1; }

BACKEND_CONFIG=$(python3 - <<'PY'
import json
import os

state = os.environ["SKYRL_STATE_DIR"]
print(json.dumps({
    "trainer.placement.colocate_all": True,
    "trainer.placement.policy_num_nodes": 1,
    "trainer.placement.policy_num_gpus_per_node": 8,
    "trainer.policy.language_model_only": True,
    "trainer.remove_microbatch_padding": False,
    "trainer.micro_forward_batch_size_per_gpu": 1,
    "trainer.micro_train_batch_size_per_gpu": 1,
    "trainer.max_prompt_length": 4096,
    "trainer.logger": "console",
    "trainer.log_path": f"{state}/logs",
    "trainer.policy.model.lora.lora_sync_path": f"{state}/lora-sync",
    "generator.sampling_params.max_generate_length": 1024,
    "generator.inference_engine.backend": "vllm",
    "generator.inference_engine.num_engines": 4,
    "generator.inference_engine.tensor_parallel_size": 2,
    "generator.inference_engine.language_model_only": True,
    "generator.inference_engine.distributed_executor_backend": "mp",
    "generator.inference_engine.run_engines_locally": True,
    "generator.inference_engine.weight_sync_backend": "nccl",
    "generator.inference_engine.gpu_memory_utilization": 0.7,
    "generator.inference_engine.engine_init_kwargs": {
        "max_model_len": 4096,
        "gdn_prefill_backend": "triton",
        "disable_custom_all_reduce": True,
    },
    "generator.batched": True,
}))
PY
)

cd "$SKYRL_DIR"
exec uv run --frozen --extra tinker --extra fsdp \
  -m skyrl.tinker.api \
  --base-model "$MODEL_DIR" --backend fsdp --host 0.0.0.0 \
  --port "${TINKER_PORT:-18080}" \
  --database-url "sqlite:///$SKYRL_STATE_DIR/tinker.db" \
  --checkpoints-base "$SKYRL_STATE_DIR/checkpoints" \
  --external-inference-lora-base "$SKYRL_STATE_DIR/external-lora" \
  --backend-config "$BACKEND_CONFIG"
