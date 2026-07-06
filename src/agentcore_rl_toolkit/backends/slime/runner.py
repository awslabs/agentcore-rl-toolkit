"""SlimeRunner — one Python entry point for slime-backed training.

Users instantiate ``SlimeRunner`` with a handful of per-experiment fields
and call ``.train()``; the runner reproduces what ``train.sh`` does today
(stop stale processes, start a Ray head, source the slime model script,
submit the slime training job) via subprocess.

``train.sh`` stays in the repo as the low-level escape hatch; this class
is the primary entry point.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Keys on SlimeRunner that flow to slime's `args` namespace via the
# --custom-config-path YAML (slime's `utils/arguments.py` setattrs each key
# onto `args` at parse time). Consumed by SlimeArtConfig.from_args().
_TOOLKIT_CONFIG_KEYS = (
    "agent_runtime_arn",
    "s3_bucket",
    "exp_id",
    "gateway_port",
    "acr_timeout",
    "model_id",
    "acr_tps_limit",
    "max_concurrent",
    "max_pool_connections",
    "reward_postprocessing",
    "cumulative_token_mode",
    "renderer_family",
    "gateway_log_level",
)


@dataclass
class SlimeRunner:
    """One Python entry point for slime-backed training.

    Example:
        >>> SlimeRunner(
        ...     exp_id="gsm8k-3b-smoke",
        ...     agent_runtime_arn="arn:aws:bedrock-agentcore:...",
        ...     s3_bucket="my-bucket",
        ...     model_dir="/path/to/Qwen2.5-3B-Instruct",
        ...     data_path="/path/to/gsm8k_tiny.jsonl",
        ...     model_type="qwen2.5-3B",
        ... ).train(num_rollout=1)
    """

    # --- Required: per-experiment ---
    exp_id: str
    agent_runtime_arn: str
    s3_bucket: str
    model_dir: str
    data_path: str
    model_type: str

    # --- Optional: cluster ---
    num_gpus: int = 8
    tp_size: int = 2
    rollout_gpus_per_engine: int = 2
    slime_dir: str = "/root/slime"
    megatron_dir: str = "/root/Megatron-LM"

    # --- Optional: CUDA toolchain pinning ---
    # When set, the runner pins the worker env to one CUDA toolkit (mirrors
    # train.sh): exports CUDA_HOME/PATH, prepends the venv-bundled nvidia/*/lib
    # dirs, and strips every other /usr/local/cuda-* from LD_LIBRARY_PATH so
    # TransformerEngine doesn't abort with "Multiple libcudart libraries found".
    # Leave None to inherit the ambient environment unchanged.
    cuda_home: str | None = None

    # --- Optional: ACR / toolkit (forwarded to slime via custom-config yaml) ---
    model_id: str = "default"
    acr_timeout: int = 900
    acr_tps_limit: int = 25
    max_concurrent: int = 100
    max_pool_connections: int = 100
    gateway_port: int = 9090
    reward_postprocessing: str = "grpo"
    # Gateway use_sglang parsers (must match the served model) + cumulative mode.
    sglang_tool_call_parser: str = "qwen"
    sglang_reasoning_parser: str | None = None
    cumulative_token_mode: bool = False
    renderer_family: str = "auto"
    gateway_log_level: str = "warning"

    # --- Optional: training hyperparameters ---
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 1024
    rollout_temperature: float = 1.0
    lr: float = 1e-6
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    weight_decay: float = 0.1
    adam_beta2: float = 0.98
    sglang_mem_fraction_static: float = 0.7
    sglang_context_length: int | None = None
    max_tokens_per_gpu: int = 9216

    # --- Wandb (opt-in; no defaults injected if unset) ---
    wandb_project: str | None = None
    wandb_group: str | None = None

    # --- Escape hatch ---
    extra_flags: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "SlimeRunner":
        """Load kwargs from a YAML file (convenience for config-file workflows)."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def train(self, num_rollout: int = 1) -> None:
        """Run the training job. Blocks until the slime job exits.

        Mirrors ``train.sh`` step-by-step: stop stale sglang/ray, start a Ray
        head, source the slime model script, submit the slime training job
        via ``ray job submit``. Streams stdout/stderr to the parent process.
        """
        self._stop_stale_processes()
        self._start_ray()
        model_args = self._source_model_script()
        runtime_env = self._build_runtime_env()
        with self._write_toolkit_config() as config_path:
            self._submit_ray_job(num_rollout, model_args, runtime_env, config_path)

    # ------------------------------------------------------------------
    # Internals — one per train.sh step, no magic
    # ------------------------------------------------------------------

    def _stop_stale_processes(self) -> None:
        subprocess.run(["pkill", "-9", "sglang"], check=False)
        subprocess.run(["ray", "stop", "--force"], check=False)
        subprocess.run(["sleep", "3"], check=True)

    def _start_ray(self) -> None:
        subprocess.run(
            ["ray", "start", "--head", "--num-gpus", str(self.num_gpus), "--disable-usage-stats"],
            check=True,
        )

    def _source_model_script(self) -> list[str]:
        """Source slime's scripts/models/<model_type>.sh and return MODEL_ARGS.

        slime ships per-model arg files (e.g. qwen2.5-3B.sh) that export
        MODEL_ARGS as a bash array. We invoke bash to source the script and
        print the array one element per null byte, then split in Python.
        """
        script = Path(self.slime_dir) / "scripts" / "models" / f"{self.model_type}.sh"
        if not script.exists():
            raise FileNotFoundError(
                f"slime model script not found: {script}. "
                f"Check slime_dir={self.slime_dir!r} and model_type={self.model_type!r}."
            )
        cmd = f'source "{script}"; printf "%s\\0" "${{MODEL_ARGS[@]}}"'
        out = subprocess.check_output(["bash", "-c", cmd])
        items = out.split(b"\0")
        return [x.decode() for x in items if x]

    def _cuda_ld_library_path(self) -> str:
        """LD_LIBRARY_PATH pinned to a single CUDA toolkit (mirrors train.sh).

        Prepends every venv-bundled ``nvidia/*/lib`` dir (runtime+cublas, cudnn,
        nccl, cusparselt, nvshmem) and ``$CUDA_HOME/lib64``, after stripping every
        other ``/usr/local/cuda-*`` from the ambient LD_LIBRARY_PATH so
        TransformerEngine sees only this CUDA runtime. Requires ``cuda_home``.
        """
        import glob
        import re
        import sysconfig

        nvidia_base = os.path.join(sysconfig.get_path("purelib"), "nvidia")
        nvidia_libs = sorted(glob.glob(os.path.join(nvidia_base, "*", "lib")))
        ambient = os.environ.get("LD_LIBRARY_PATH", "")
        kept = [p for p in ambient.split(":") if p and not re.match(r"^/usr/local/cuda(-[0-9.]+)?/", p)]
        parts = [*nvidia_libs, f"{self.cuda_home}/lib64", *kept]
        return ":".join(parts)

    def _build_runtime_env(self) -> dict:
        """Runtime env forwarded to every Ray worker.

        Mirrors train.sh's inline python snippet: PYTHONPATH for Megatron,
        CUDA_DEVICE_MAX_CONNECTIONS for Megatron TP, plus wandb keys when set in
        the parent environment. When ``cuda_home`` is set, also pins CUDA_HOME +
        LD_LIBRARY_PATH to one toolkit so the worker (and the train actors, via
        --train-env-vars) avoid the "Multiple libcudart libraries found" abort.
        """
        env_vars: dict[str, str] = {
            "PYTHONPATH": self.megatron_dir,
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }
        if self.cuda_home:
            env_vars["CUDA_HOME"] = self.cuda_home
            env_vars["LD_LIBRARY_PATH"] = self._cuda_ld_library_path()
        for key in ("WANDB_API_KEY", "WANDB_ENTITY"):
            val = os.environ.get(key)
            if val:
                env_vars[key] = val
        return {"env_vars": env_vars}

    def _write_toolkit_config(self):
        """Write a temp YAML of toolkit fields for slime --custom-config-path.

        slime's argparse loads this YAML and setattr's each key onto args,
        where our rollout integration reads them via SlimeArtConfig.from_args.
        Returned as a context manager so the temp file lives for the job's
        duration and is cleaned up after.
        """
        import contextlib

        import yaml

        data = {k: getattr(self, k) for k in _TOOLKIT_CONFIG_KEYS}

        @contextlib.contextmanager
        def _ctx():
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix="slime-runner-", delete=False) as f:
                yaml.safe_dump(data, f)
                path = f.name
            try:
                yield path
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        return _ctx()

    def _submit_ray_job(
        self,
        num_rollout: int,
        model_args: list[str],
        runtime_env: dict,
        config_path: str,
    ) -> None:
        flags = self._build_slime_flags(num_rollout, model_args, config_path)
        cmd = [
            "ray",
            "job",
            "submit",
            "--address=http://127.0.0.1:8265",
            f"--runtime-env-json={json.dumps(runtime_env)}",
            "--",
            "python3",
            str(Path(self.slime_dir) / "train.py"),
            *flags,
        ]
        subprocess.run(cmd, check=True)

    def _build_slime_flags(
        self,
        num_rollout: int,
        model_args: list[str],
        config_path: str,
    ) -> list[str]:
        """Flags passed to ``python3 slime/train.py`` — mirrors train.sh 1:1."""
        flags: list[str] = [
            *model_args,
            "--hf-checkpoint",
            self.model_dir,
            "--ref-load",
            self.model_dir,
            "--prompt-data",
            self.data_path,
            "--num-rollout",
            str(num_rollout),
            "--tensor-model-parallel-size",
            str(self.tp_size),
            "--rollout-num-gpus-per-engine",
            str(self.rollout_gpus_per_engine),
            "--input-key",
            "prompt",
            "--rollout-batch-size",
            str(self.rollout_batch_size),
            "--n-samples-per-prompt",
            str(self.n_samples_per_prompt),
            "--rollout-max-response-len",
            str(self.rollout_max_response_len),
            "--rollout-temperature",
            str(self.rollout_temperature),
            "--advantage-estimator",
            "grpo",
            "--use-kl-loss",
            "--kl-loss-type",
            "low_var_kl",
            "--eps-clip",
            str(self.eps_clip),
            "--eps-clip-high",
            str(self.eps_clip_high),
            "--lr",
            str(self.lr),
            "--lr-decay-style",
            "constant",
            "--weight-decay",
            str(self.weight_decay),
            "--adam-beta2",
            str(self.adam_beta2),
            "--optimizer-cpu-offload",
            "--overlap-cpu-optimizer-d2h-h2d",
            "--use-precision-aware-optimizer",
            "--sequence-parallel",
            "--sglang-mem-fraction-static",
            str(self.sglang_mem_fraction_static),
            "--sglang-cuda-graph-max-bs",
            "32",
            "--sglang-tool-call-parser",
            self.sglang_tool_call_parser,
            "--sglang-log-level",
            "warning",
            "--sglang-log-level-http",
            "warning",
            "--attention-dropout",
            "0.0",
            "--hidden-dropout",
            "0.0",
            "--accumulate-allreduce-grads-in-fp32",
            "--attention-softmax-in-fp32",
            "--attention-backend",
            "flash",
            "--actor-num-gpus-per-node",
            str(self.num_gpus),
            "--colocate",
            "--megatron-to-hf-mode",
            "bridge",
            "--rollout-function-path",
            "agentcore_rl_toolkit.backends.slime.integration.rollout.generate_rollout",
            "--custom-reward-post-process-path",
            "agentcore_rl_toolkit.backends.slime.integration.rewards.normalize_episode_rewards",
            "--custom-config-path",
            config_path,
            "--use-dynamic-global-batch-size",
            "--use-dynamic-batch-size",
            "--max-tokens-per-gpu",
            str(self.max_tokens_per_gpu),
        ]

        # Optional SGLang reasoning parser (split <think>...</think> in the gateway).
        if self.sglang_reasoning_parser:
            flags.extend(["--sglang-reasoning-parser", self.sglang_reasoning_parser])
        # Optional SGLang context length cap.
        if self.sglang_context_length is not None:
            flags.extend(["--sglang-context-length", str(self.sglang_context_length)])
        # CUDA pinning for the Megatron train actors: slime gives them their own
        # runtime_env that does NOT carry CUDA_HOME/LD_LIBRARY_PATH, so pass the
        # pinned paths through --train-env-vars too (mirrors train.sh). Only when
        # cuda_home is set.
        if self.cuda_home:
            train_env = {"CUDA_HOME": self.cuda_home, "LD_LIBRARY_PATH": self._cuda_ld_library_path()}
            flags.extend(["--train-env-vars", json.dumps(train_env)])

        # Wandb opt-in: only emit --use-wandb if the user (or env) supplied an API key.
        if os.environ.get("WANDB_API_KEY"):
            flags.append("--use-wandb")
            if self.wandb_project:
                flags.extend(["--wandb-project", self.wandb_project])
            if self.wandb_group:
                flags.extend(["--wandb-group", self.wandb_group])

        flags.extend(self.extra_flags)
        return flags
