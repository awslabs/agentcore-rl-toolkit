"""vLLM-Neuron plugins that make ``vllm serve`` SkyRL's external sampler, with LoRA refresh.

Start stock ``vllm serve`` with ``--middleware server.ModelResolver`` and
``--worker-extension-cls server.WeightRefreshWorker`` (this directory on ``PYTHONPATH``),
configured by the ``ART_WEIGHT_REFRESH_*`` environment variables (see ``refresh_settings``).

SkyRL's ``--external-inference-url`` client samples with ``model=<model_id>_<ckpt>``.
The first request naming a new version triggers a refresh: download SkyRL's sampler
archive from S3, merge it into the immutable base on CPU, then stage and copy weights
on every replica. Workers retain parameter objects and compiled graphs.
Failures after the request barrier leave serving unavailable until recovery.

This module is model-independent. Model-specific merging and the Neuron model class
live in sibling ``model_<type>.py`` modules, registered in ``MODEL_SUPPORT``.
"""

import asyncio
import hmac
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

if __package__:
    from . import model_qwen3_5
else:  # Imported by name with this directory on sys.path (PYTHONPATH).
    import model_qwen3_5

MODEL_SUPPORT = {module.MODEL_TYPE: module for module in (model_qwen3_5,)}
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
# Copied from the base into every merged checkpoint, and required to match the resident model.
BASE_METADATA = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
)
SKYRL_ID = re.compile(r"[A-Za-z0-9_-]+")  # SkyRL model and checkpoint IDs; also safe as a directory name.
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
SAMPLING_PATHS = ("/v1/completions", "/v1/chat/completions")


def refresh_settings(environ=os.environ):
    """Read and validate the refresh configuration; raises at ``vllm serve`` startup."""
    names = ("ART_WEIGHT_REFRESH_ROOT", "ART_WEIGHT_REFRESH_BASE", "ART_WEIGHT_REFRESH_CHECKPOINTS")
    missing = [name for name in names if not environ.get(name)]
    if missing:
        raise ValueError(f"Weight refresh requires {', '.join(missing)}")
    settings = SimpleNamespace(
        # Merged checkpoints are written here; the immutable local base is used for every merge.
        root=str(Path(environ["ART_WEIGHT_REFRESH_ROOT"]).resolve(strict=True)),
        base_path=str(Path(environ["ART_WEIGHT_REFRESH_BASE"]).resolve(strict=True)),
        # SkyRL's --checkpoints-base (s3://bucket/prefix) holding sampler archives, and its region.
        checkpoints_prefix=environ["ART_WEIGHT_REFRESH_CHECKPOINTS"],
        s3_region=environ.get("ART_WEIGHT_REFRESH_S3_REGION") or None,
    )
    s3_location(settings.checkpoints_prefix)
    model_support(settings.base_path)
    return settings


def model_support(checkpoint):
    """Return the model-support module for a checkpoint's ``model_type``."""
    model_type = json.loads((Path(checkpoint) / "config.json").read_text()).get("model_type")
    if model_type not in MODEL_SUPPORT:
        raise ValueError(
            f"Weight refresh does not support model_type {model_type!r}; "
            f"supported: {', '.join(sorted(MODEL_SUPPORT))}"
        )
    return MODEL_SUPPORT[model_type]


# --- Checkpoint storage: SkyRL sampler archives in S3 ---------------------------------


def s3_location(uri):
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or ":" in parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Checkpoint storage must be an s3://bucket/prefix URI")
    key = parsed.path.lstrip("/").rstrip("/")
    if not key or any(part in ("", ".", "..") for part in key.split("/")):
        raise ValueError("S3 checkpoint URI requires a nonempty prefix without relative components")
    return parsed.netloc, key


def s3_client(region=None):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name=region,
        config=Config(connect_timeout=10, read_timeout=60, retries={"mode": "standard", "max_attempts": 2}),
    )


def sampler_checkpoint_key(version, checkpoints_prefix, client):
    """Find SkyRL's ``<model_id>/sampler_weights/<ckpt>.tar.gz`` for ``<model_id>_<ckpt>``.

    Both IDs may contain ``_``. Every split is checked; exactly one archive must exist.
    """
    from botocore.exceptions import ClientError

    bucket, prefix = s3_location(checkpoints_prefix)
    found = []
    for index, char in enumerate(version):
        model_id, checkpoint_id = version[:index], version[index + 1 :]
        if char != "_" or not SKYRL_ID.fullmatch(model_id) or not SKYRL_ID.fullmatch(checkpoint_id):
            continue
        key = f"{prefix}/{model_id}/sampler_weights/{checkpoint_id}.tar.gz"
        try:
            client.head_object(Bucket=bucket, Key=key)
        except ClientError as error:
            # Without s3:ListBucket, S3 reports a missing key as 403 rather than 404.
            if error.response.get("Error", {}).get("Code") not in ("403", "404", "NoSuchKey", "NotFound"):
                raise
            continue
        found.append(key)
    if len(found) != 1:
        raise LookupError(
            f"No unique readable SkyRL sampler checkpoint for model {version!r} "
            f"under {checkpoints_prefix} (missing, ambiguous or access denied)"
        )
    return bucket, found[0]


def prepare_adapter(version, *, base_path, output_root, checkpoints_prefix, region=None, client=None):
    """Download SkyRL's sampler archive for ``version`` and merge it into the original base."""
    if not isinstance(version, str) or not SKYRL_ID.fullmatch(version):
        raise LookupError(f"{version!r} is not a SkyRL <model_id>_<ckpt> name")
    root = Path(output_root).resolve(strict=True)
    target = root / version
    if target.exists():
        # merge_checkpoint publishes atomically, so an existing directory is complete.
        return target
    client = client if client is not None else s3_client(region)
    bucket, key = sampler_checkpoint_key(version, checkpoints_prefix, client)
    with tempfile.TemporaryDirectory(dir=root, prefix=".adapter-") as temporary:
        archive, adapter = Path(temporary) / "adapter.tar.gz", Path(temporary) / "adapter"
        response = client.get_object(Bucket=bucket, Key=key)
        if not 0 < response["ContentLength"] <= MAX_ARCHIVE_BYTES:
            response["Body"].close()
            raise ValueError("Sampler archive must be between 1 byte and 4 GiB")
        with response["Body"] as stream, archive.open("wb") as output:
            shutil.copyfileobj(stream, output, 8 * 1024 * 1024)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(adapter, filter="data")
        for name in ADAPTER_FILES:
            path = adapter / name
            if path.is_symlink() or not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Sampler archive is missing {name}")
        if json.loads((adapter / "adapter_config.json").read_text()).get("peft_type") != "LORA":
            raise ValueError("Only PEFT LoRA adapter checkpoints are supported")
        return merge_checkpoint(base_path, adapter, target, version)


# --- Merge and validation ---------------------------------------------------------


def merge_checkpoint(base_path, adapter_path, output_path, version):
    import torch

    # Model loading may initialize temporary CPU tensors; preserve the serving process RNG.
    with torch.random.fork_rng(devices=[]):
        base_path, adapter_path, output_path = map(Path, (base_path, adapter_path, output_path))
        if not base_path.is_dir():
            raise ValueError("Use a local, pinned base checkpoint")
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint {output_path}")
        # Merge on a new CPU copy while the resident serving model stays available.
        merged = model_support(base_path).load_merged(base_path, adapter_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_path.parent, prefix=".merge-") as temporary:
            staged = Path(temporary) / "checkpoint"
            # Export the model's native keys. HF's reverse-conversion mappings
            # differ across versions and can misname the frozen vision backbone.
            merged.save_pretrained(staged, safe_serialization=True, save_original_format=False)
            # Keep the adapter snapshot with the artifact for reproducibility.
            shutil.copytree(adapter_path, staged / "adapter")
            # LoRA changes weights, not architecture/tokenization. Preserve the
            # pinned base bytes instead of reserializing with the local HF version.
            for name in BASE_METADATA:
                if (base_path / name).is_file():
                    shutil.copy2(base_path / name, staged / name)
                elif (staged / name).exists():
                    (staged / name).unlink()
            # Publish atomically: an existing output directory is always complete.
            staged.rename(output_path)
        return output_path


def validate_checkpoint(root, current, checkpoint, version):
    """Check a merged checkpoint before activation.

    ``merge_checkpoint`` publishes ``<root>/<version>`` with an atomic rename, so the
    directory is complete. Only weights may change: the resident config and tokenizer
    must remain valid for every compiled shape.
    """
    root, current, checkpoint = (Path(p).resolve(strict=True) for p in (root, current, checkpoint))
    if checkpoint.parent != root or checkpoint.name != version:
        raise ValueError("Checkpoint must be this version's directory inside the configured refresh root")
    if not any(checkpoint.glob("*.safetensors")):
        raise ValueError("Checkpoint has no weight files")
    for name in BASE_METADATA:
        old, new = current / name, checkpoint / name
        if old.exists() != new.exists() or (old.exists() and old.read_bytes() != new.read_bytes()):
            raise ValueError(f"Refresh cannot change {name}; restart required")
    return str(checkpoint)


# --- vLLM worker extension (runs in every worker process) ---------------------------


def encoder_config(runner):
    return getattr(runner, "encoder_neuron_config", getattr(runner, "vision_neuron_config", None))


class WeightRefreshWorker:
    """vLLM worker extension; no changes to vLLM or Neuron worker classes."""

    def reset_encoder_cache(self):
        # vLLM's pause(clear_cache=True) unconditionally calls this method.
        # The deployed Neuron worker omits it; text-only serving has no encoder
        # cache to invalidate. Reject an encoder rather than silently retain it.
        if encoder_config(self.model_runner) is not None:
            raise ValueError("Persistent refresh does not support a vision encoder")

    def stage_checkpoint(self, checkpoint, version):
        import torch

        support = model_support(checkpoint)
        runner = self.model_runner
        model = getattr(runner.model, "_orig_mod", runner.model)
        if type(model) is not support.neuron_model_class():
            raise ValueError(f"Persistent refresh for {support.MODEL_TYPE} requires its validated Neuron model")
        if runner.drafter is not None or encoder_config(runner) is not None:
            raise ValueError("Persistent refresh requires text-only, non-speculative serving")
        started = time.monotonic()
        with torch.device("meta"):
            staged = type(model)(model.config)
        staged.load_weights(checkpoint, torch.device("cpu"), None)
        source = dict(staged.named_parameters())
        target = dict(model.named_parameters())
        if source.keys() != target.keys():
            raise ValueError("Checkpoint parameter names differ")
        for name, tensor in source.items():
            old = target[name]
            if tensor.shape != old.shape or tensor.dtype != old.dtype:
                raise ValueError(f"Checkpoint shape/dtype differs: {name}")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"Nonfinite checkpoint parameter: {name}")
        self._refresh_staged = (version, source)
        return {
            "pid": os.getpid(),
            "rank": self.rank,
            "parameters": len(source),
            "stage_seconds": time.monotonic() - started,
            "version": version,
        }

    def apply_checkpoint(self, version):
        import torch

        started = time.monotonic()
        staged_version, source = self._refresh_staged
        if version != staged_version:
            raise ValueError("No staged checkpoint for requested version")
        model = getattr(self.model_runner.model, "_orig_mod", self.model_runner.model)
        target = dict(model.named_parameters())
        changed = 0
        with torch.no_grad():
            for name, tensor in source.items():
                old = target[name]
                # Keep the tensor/storage identity used by compiled graphs.
                before = old.cpu()
                changed += int(not torch.equal(before, tensor))
                old.copy_(tensor)
                if not torch.equal(old.cpu(), tensor):
                    raise RuntimeError(f"Device weight verification failed: {name}")
        self._refresh_staged = None
        self._refresh_version = version
        return {
            "pid": os.getpid(),
            "rank": self.rank,
            "version": version,
            "changed_parameters": changed,
            "parameters": len(source),
            "apply_seconds": time.monotonic() - started,
        }


# --- Request routing and refresh orchestration (API server process) -----------------


async def all_workers(engine, method, *args):
    # vLLM's public DP collective waits for all replicas but returns only the
    # first replica's result. Keep all acknowledgements for activation checks.
    core = engine.engine_core
    if hasattr(core, "core_engines"):
        groups = await asyncio.gather(
            *[
                core._call_utility_async("collective_rpc", method, 180, args, {}, engine=replica)
                for replica in core.core_engines
            ]
        )
        return [worker for group in groups for worker in group]
    return await engine.collective_rpc(method, timeout=180, args=args)


def check_acknowledgements(results, expected, version, step):
    pids = {r["pid"] for r in results}
    if len(results) != expected or len(pids) != expected:
        raise RuntimeError(f"Missing {step} acknowledgements")
    if any(r["version"] != version for r in results):
        raise RuntimeError(f"{step.capitalize()} version mismatch")
    return pids


async def read_body(receive):
    chunks, more = [], True
    while more:
        message = await receive()
        if message["type"] == "http.disconnect":
            raise ConnectionError("Client disconnected")
        chunks.append(message.get("body", b""))
        more = message.get("more_body", False)
    return b"".join(chunks)


def prepare_request(path, body):
    """Return (model, body). Rewrites completion logprobs N>0 to 0 for NKI sampling."""
    try:
        payload = json.loads(body)
    except ValueError:
        return None, body
    if not isinstance(payload, dict):
        return None, body
    # NKI sampling rejects logprobs other than 0. SkyRL's external client sends
    # logprobs=True but reads only each sampled token's logprob, which 0 returns.
    if path == "/v1/completions" and payload.get("logprobs") not in (None, False, 0):
        payload["logprobs"] = 0
        body = json.dumps(payload).encode()
    model = payload.get("model")
    return (model if isinstance(model, str) else None), body


class RefreshController:
    """Activates ``<model_id>_<ckpt>`` the first time a sampling request names it.

    This mirrors vLLM's filesystem LoRA resolver for SkyRL's ``--external-inference-url``
    client: requests for a new version wait for its merge and refresh instead of failing,
    and versions that were replaced are rejected rather than reloaded.
    """

    def __init__(self, app, settings):
        # vLLM stores its parsed CLI arguments on app.state; the middleware cannot see them earlier.
        args = getattr(app.state, "args", None)
        # The same API keys vLLM's own authentication accepts (--api-key, else VLLM_API_KEY).
        self.tokens = [key for key in (getattr(args, "api_key", None) or [os.environ.get("VLLM_API_KEY")]) if key]
        if args is not None:
            if args.host not in ("127.0.0.1", "::1") and not self.tokens:
                raise ValueError("Weight refresh on a non-loopback --host requires --api-key or VLLM_API_KEY")
            if not str(args.worker_extension_cls or "").endswith(".WeightRefreshWorker"):
                raise ValueError("Weight refresh requires --worker-extension-cls <module>.WeightRefreshWorker")
            # vllm serve defaults to one API server per data-parallel rank; each would hold its own refresh state.
            if (getattr(args, "api_server_count", None) or 1) != 1:
                raise ValueError("Weight refresh requires vllm serve --api-server-count 1")
        self.app, self.root, self.base_path = app, settings.root, settings.base_path
        self.checkpoints_prefix, self.s3_region = settings.checkpoints_prefix, settings.s3_region
        self.busy = self.failed = False  # busy: admission blocked; failed: resident model unknown
        self.task = None  # (version, asyncio.Task) while a refresh runs
        self.active, self.retired = 0, set()
        self.drained = asyncio.Event()
        self.drained.set()

    def authorized(self, scope):
        """vLLM checks API keys after this middleware, so a new model name must not start a refresh first."""
        if not self.tokens:
            return True
        header = dict(scope["headers"]).get(b"authorization", b"")
        return any(hmac.compare_digest(header, f"Bearer {token}".encode()) for token in self.tokens)

    def models(self):
        # vLLM sets this in init_app_state, after build_app returns; read it per call.
        return self.app.state.openai_serving_models

    def served(self):
        return self.models().base_model_paths[0].name

    async def resolve(self, model):
        """Return None when the request may proceed, otherwise (status, body)."""
        while True:
            if self.failed:
                return 503, {"error": "Checkpoint refresh failed; restart required", "restart_required": True}
            if self.task is not None:
                version, task = self.task
                if model == self.served() and not self.busy:
                    return None  # The old version stays available while the new one is prepared.
                try:
                    await asyncio.shield(task)
                except Exception as exc:
                    if model == version:
                        status = 404 if isinstance(exc, LookupError) else 500
                        return status, {
                            "error": f"Activation of {version} failed: {exc}",
                            "restart_required": self.failed,
                        }
                continue
            if model is None or model == self.served():
                return None
            if model in self.retired:
                return 409, {"error": f"Model {model} was replaced by {self.served()}"}
            self.task = (model, asyncio.create_task(self.guarded_refresh(model)))

    async def track(self, app, scope, receive, send):
        # Counts request parsing and the entire streaming response,
        # not just requests already submitted to the engine.
        self.active += 1
        self.drained.clear()
        try:
            await app(scope, receive, send)
        finally:
            self.active -= 1
            if self.active == 0:
                self.drained.set()

    async def refresh(self, version):
        from vllm.entrypoints.openai.models.protocol import BaseModelPath

        started = time.monotonic()
        engine = self.app.state.engine_client
        current = self.models().base_model_paths[0].model_path
        checkpoint = await asyncio.to_thread(
            prepare_adapter,
            version,
            base_path=self.base_path,
            output_root=self.root,
            checkpoints_prefix=self.checkpoints_prefix,
            region=self.s3_region,
        )
        checkpoint = await asyncio.to_thread(validate_checkpoint, self.root, current, checkpoint, version)
        prepared = time.monotonic()
        # Download and CPU merge do not block requests to the old version.
        self.busy = True
        await asyncio.wait_for(self.drained.wait(), timeout=1800)
        await engine.pause_generation(mode="wait", clear_cache=True)
        expected = engine.vllm_config.parallel_config.tensor_parallel_size * len(
            getattr(engine.engine_core, "core_engines", [None])
        )
        staged = await all_workers(engine, "stage_checkpoint", checkpoint, version)
        staged_pids = check_acknowledgements(staged, expected, version, "staging")
        applied = await all_workers(engine, "apply_checkpoint", version)
        if check_acknowledgements(applied, expected, version, "worker") != staged_pids:
            raise RuntimeError("Workers changed during refresh")
        previous = self.models().base_model_paths[0]
        # All request handlers share this list through the model registry.
        self.models().base_model_paths[:] = [BaseModelPath(name=version, model_path=checkpoint)]
        self.retired.add(previous.name)
        await engine.resume_generation()
        # Device weights are copied; the replaced merged checkpoint (never the base) is no longer read.
        old = Path(previous.model_path).resolve()
        if old != Path(checkpoint).resolve() and old.is_relative_to(Path(self.root).resolve()) and old.is_dir():
            await asyncio.to_thread(shutil.rmtree, old, True)
        result = {
            "version": version,
            "seconds": time.monotonic() - started,
            "prepare_seconds": prepared - started,
            "refresh_seconds": time.monotonic() - prepared,
            "staged": staged,
            "applied": applied,
        }
        print(json.dumps({"weight_refresh": result}), flush=True)
        return result

    async def guarded_refresh(self, version):
        try:
            return await self.refresh(version)
        except BaseException:
            # A failed download/merge has not touched the resident model.
            self.failed = self.busy
            raise
        finally:
            self.busy = False
            self.task = None


class ModelResolver:
    """ASGI middleware, installed through vLLM's ``--middleware``: resolves sampling
    requests' model names and tracks in-flight requests."""

    def __init__(self, app, settings=None):
        # vLLM's build_app constructs this, so invalid settings stop vllm serve from starting.
        self.app, self.settings = app, settings or refresh_settings()

    async def __call__(self, scope, receive, send):
        from fastapi.responses import JSONResponse

        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        state = scope["app"].state  # Starlette passes the application in every scope.
        controller = getattr(state, "weight_refresh", None)
        if controller is None:
            controller = state.weight_refresh = RefreshController(scope["app"], self.settings)
        if scope["path"] not in SAMPLING_PATHS or scope["method"] != "POST":
            if controller.busy or controller.failed:
                response = JSONResponse({"error": "Checkpoint refresh in progress or failed"}, status_code=503)
                return await response(scope, receive, send)
            return await controller.track(self.app, scope, receive, send)
        if not controller.authorized(scope):
            return await JSONResponse({"error": "Unauthorized"}, status_code=401)(scope, receive, send)
        model, body = prepare_request(scope["path"], await read_body(receive))
        rejected = await controller.resolve(model)
        if rejected is not None:
            return await JSONResponse(rejected[1], status_code=rejected[0])(scope, receive, send)
        replayed = False

        async def receive_body():
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        scope = {
            **scope,
            "headers": [
                (k, str(len(body)).encode() if k.lower() == b"content-length" else v) for k, v in scope["headers"]
            ],
        }
        return await controller.track(self.app, scope, receive_body, send)
