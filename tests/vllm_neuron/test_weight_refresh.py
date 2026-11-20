"""Checkpoint integrity and all-replica acknowledgements are activation invariants."""

import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

SOURCE = Path(
    os.environ.get(
        "ART_WEIGHT_REFRESH_SOURCE",
        str(
            Path(__file__).resolve().parents[2]
            / "src/agentcore_rl_toolkit/backends/tinker_api/skyrl/serving/vllm_neuron/server.py"
        ),
    )
)
# As in script execution, the server's directory provides its model-support modules.
sys.path.insert(0, str(SOURCE.parent))
spec = importlib.util.spec_from_file_location("vllm_neuron_serve", SOURCE)
refresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refresh)


@pytest.mark.parametrize("field", ["vision_neuron_config", "encoder_neuron_config"])
def test_encoder_cache_reset_rejects_encoder_in_either_runner_layout(field):
    from types import SimpleNamespace

    worker = refresh.WeightRefreshWorker()
    worker.model_runner = SimpleNamespace(**{field: None})
    worker.reset_encoder_cache()
    setattr(worker.model_runner, field, object())
    with pytest.raises(ValueError, match="vision encoder"):
        worker.reset_encoder_cache()


PACKAGE_NAME = "agentcore_rl_toolkit.backends.tinker_api.skyrl.serving.vllm_neuron.server"


@pytest.mark.parametrize("name", ["server", PACKAGE_NAME])
def test_vllm_imports_plugins_by_name_without_starting_anything(name):
    """vLLM imports --middleware in the API server and --worker-extension-cls in each worker."""
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib, sys; "
            "module = importlib.import_module(sys.argv[1]); "
            "assert callable(module.ModelResolver) and callable(module.WeightRefreshWorker); "
            'assert "vllm" not in sys.modules and "torch" not in sys.modules',
            name,
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [str(SOURCE.parent), str(Path(__file__).resolve().parents[2] / "src"), *sys.path]
            ),
        },
        check=True,
        capture_output=True,
        text=True,
    )


def test_refresh_settings_fail_startup_when_incomplete_or_invalid(monkeypatch, tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    env = {
        "ART_WEIGHT_REFRESH_ROOT": str(tmp_path),
        "ART_WEIGHT_REFRESH_BASE": str(base),
        "ART_WEIGHT_REFRESH_CHECKPOINTS": "s3://bucket/skyrl-checkpoints",
        "ART_WEIGHT_REFRESH_S3_REGION": "us-east-2",
    }
    settings = refresh.refresh_settings(env)
    assert (settings.root, settings.base_path) == (str(tmp_path.resolve()), str(base.resolve()))
    assert settings.s3_region == "us-east-2"
    with pytest.raises(ValueError, match="ART_WEIGHT_REFRESH_CHECKPOINTS"):
        refresh.refresh_settings({**env, "ART_WEIGHT_REFRESH_CHECKPOINTS": ""})
    with pytest.raises(ValueError, match="s3://"):
        refresh.refresh_settings({**env, "ART_WEIGHT_REFRESH_CHECKPOINTS": "/local/path"})
    with pytest.raises(FileNotFoundError):
        refresh.refresh_settings({**env, "ART_WEIGHT_REFRESH_ROOT": str(tmp_path / "missing")})
    (base / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with pytest.raises(ValueError, match="model_type"):
        refresh.refresh_settings(env)
    # vLLM's build_app constructs the middleware from the environment, so vllm serve stops here.
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="model_type"):
        refresh.ModelResolver(app=None)


def test_controller_requires_loopback_or_api_key_and_the_worker_extension(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    settings = SimpleNamespace(root=str(tmp_path), base_path="/base", checkpoints_prefix="s3://b/p", s3_region=None)

    def app(**args):
        defaults = {"host": "127.0.0.1", "worker_extension_cls": "server.WeightRefreshWorker"}
        return SimpleNamespace(state=SimpleNamespace(args=SimpleNamespace(**{**defaults, **args})))

    assert refresh.RefreshController(app(), settings).root == str(tmp_path)
    with pytest.raises(ValueError, match="--host"):
        refresh.RefreshController(app(host=None), settings)
    with pytest.raises(ValueError, match="--api-key or VLLM_API_KEY"):
        refresh.RefreshController(app(host="10.0.1.5"), settings)
    # A private address is allowed once vLLM requires an API key.
    assert refresh.RefreshController(app(host="10.0.1.5", api_key=["secret"]), settings).tokens == ["secret"]
    with pytest.raises(ValueError, match="worker-extension-cls"):
        refresh.RefreshController(app(worker_extension_cls=""), settings)
    with pytest.raises(ValueError, match="--api-server-count 1"):
        refresh.RefreshController(app(api_server_count=8), settings)
    assert refresh.RefreshController(app(api_server_count=1), settings).root == str(tmp_path)


def checkpoint(root, name):
    path = root / name
    path.mkdir()
    (path / "config.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")
    (path / "model.safetensors").write_bytes(name.encode())
    return path


def test_refresh_requires_weights_and_unchanged_model_contract(tmp_path):
    old = checkpoint(tmp_path, "old")
    new = checkpoint(tmp_path, "new")
    assert refresh.validate_checkpoint(tmp_path, old, new, "new") == str(new)
    with pytest.raises(ValueError, match="version"):
        refresh.validate_checkpoint(tmp_path, old, new, "wrong")
    (new / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="no weight files"):
        refresh.validate_checkpoint(tmp_path, old, new, "new")
    changed = checkpoint(tmp_path, "changed")
    (old / "tokenizer.json").write_text('{"different":true}')
    with pytest.raises(ValueError, match="tokenizer.json"):
        refresh.validate_checkpoint(tmp_path, old, changed, "changed")
    outside = checkpoint(tmp_path, "outside")
    with pytest.raises(ValueError, match="inside"):
        refresh.validate_checkpoint(new, old, outside, "outside")


def test_collective_checks_every_replica_and_propagates_remote_failure():
    class Core:
        core_engines = [0, 1, 2]
        fail = False

        async def _call_utility_async(self, *args, engine):
            if engine == 2 and self.fail:
                raise RuntimeError("replica 2 failed")
            return [{"pid": engine * 4 + rank} for rank in range(4)]

    class Engine:
        engine_core = Core()

    engine = Engine()
    results = asyncio.run(refresh.all_workers(engine, "apply_checkpoint", "new"))
    assert len({r["pid"] for r in results}) == 12
    engine.engine_core.fail = True
    with pytest.raises(RuntimeError, match="replica 2"):
        asyncio.run(refresh.all_workers(engine, "apply_checkpoint", "new"))


def test_text_worker_supports_pause_encoder_reset_but_rejects_vision():
    from types import SimpleNamespace

    worker = refresh.WeightRefreshWorker()
    worker.model_runner = SimpleNamespace(vision_neuron_config=None)
    worker.reset_encoder_cache()
    worker.model_runner.vision_neuron_config = object()
    with pytest.raises(ValueError, match="vision encoder"):
        worker.reset_encoder_cache()


def serving_app(monkeypatch, workers=None):
    """FastAPI stand-in for vLLM's app: /v1/completions echoes the served model."""
    import sys
    import types
    from types import SimpleNamespace

    fastapi = pytest.importorskip("fastapi")
    protocol = types.ModuleType("vllm.entrypoints.openai.models.protocol")
    protocol.BaseModelPath = SimpleNamespace
    monkeypatch.setitem(sys.modules, protocol.__name__, protocol)
    monkeypatch.setattr(refresh, "validate_checkpoint", lambda root, current, checkpoint, version: checkpoint)
    calls = []

    class Engine:
        vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=4))
        engine_core = SimpleNamespace(core_engines=[0, 1, 2])

        async def pause_generation(self, **kwargs):
            calls.append(("pause", kwargs))

        async def resume_generation(self):
            calls.append(("resume", {}))

    async def all_ok(engine, method, *args):
        calls.append((method, args))
        return [{"pid": i, "version": args[-1]} for i in range(12)]

    monkeypatch.setattr(refresh, "all_workers", workers or all_ok)
    app = fastapi.FastAPI()
    models = SimpleNamespace(base_model_paths=[SimpleNamespace(name="base", model_path="/base")])
    gate = {"entered": asyncio.Event(), "finish": asyncio.Event(), "slow": False}

    def init_state():
        # vLLM populates these in init_app_state, after build_app returns.
        app.state.engine_client = Engine()
        app.state.openai_serving_models = models

    app.init_state = init_state

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/v1/completions")
    async def completions(body: dict):
        if gate["slow"]:
            gate["entered"].set()
            await gate["finish"].wait()
        return {"model": models.base_model_paths[0].name, "requested": body["model"], "logprobs": body.get("logprobs")}

    return app, models, calls, gate


def install_routes(app, root, base_path, checkpoints_prefix, s3_region=None):
    """Attach the middleware as vLLM's build_app does for ``--middleware``, with explicit settings."""
    from types import SimpleNamespace

    settings = SimpleNamespace(
        root=root, base_path=base_path, checkpoints_prefix=checkpoints_prefix, s3_region=s3_region
    )
    app.add_middleware(refresh.ModelResolver, settings=settings)


def sample(client, model):
    return client.post("/v1/completions", json={"model": model, "prompt": [1, 2]})


@pytest.mark.parametrize("fail_apply", [False, True])
def test_new_model_name_triggers_one_refresh_and_waits_instead_of_failing(monkeypatch, tmp_path, fail_apply):
    httpx = pytest.importorskip("httpx")
    entered, proceed = asyncio.Event(), asyncio.Event()
    prepared = []

    def prepare(version, **kwargs):
        prepared.append((version, kwargs["checkpoints_prefix"]))
        path = tmp_path / version
        path.mkdir()
        return str(path)

    monkeypatch.setattr(refresh, "prepare_adapter", prepare)

    async def workers(engine, method, *args):
        calls.append((method, args))
        if method == "stage_checkpoint":
            entered.set()
            await proceed.wait()
        if method == "apply_checkpoint" and fail_apply:
            raise RuntimeError("one replica failed after partial copy")
        return [{"pid": i, "version": args[-1]} for i in range(12)]

    app, models, calls, gate = serving_app(monkeypatch, workers)
    install_routes(app, str(tmp_path), "/base", "s3://bucket/skyrl-checkpoints")
    app.init_state()

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
            gate["slow"] = True
            old_request = asyncio.create_task(sample(client, "base"))
            await gate["entered"].wait()
            gate["slow"] = False
            first = asyncio.create_task(sample(client, "model_ab12cd34_step-1"))
            second = asyncio.create_task(sample(client, "model_ab12cd34_step-1"))
            for _ in range(200):
                if (await client.get("/health")).status_code == 503:
                    break
                await asyncio.sleep(0.001)
            else:
                pytest.fail("Refresh did not block admission")
            assert not any(name == "pause" for name, _ in calls), "Weights changed under an in-flight request"
            gate["finish"].set()
            assert (await old_request).json()["model"] == "base"
            await asyncio.wait_for(entered.wait(), 5)
            assert models.base_model_paths[0].name == "base"
            proceed.set()
            results = await asyncio.gather(first, second)
            assert prepared == [("model_ab12cd34_step-1", "s3://bucket/skyrl-checkpoints")]
            if fail_apply:
                assert [r.status_code for r in results] == [500, 500]
                assert results[0].json()["restart_required"] is True
                assert not any(name == "resume" for name, _ in calls)
                assert (await client.get("/health")).status_code == 503
                assert (await sample(client, "base")).status_code == 503
            else:
                assert [r.json()["model"] for r in results] == ["model_ab12cd34_step-1"] * 2
                assert calls[0] == ("pause", {"mode": "wait", "clear_cache": True})
                assert calls[-1][0] == "resume"
                stale = await sample(client, "base")
                assert stale.status_code == 409 and "replaced" in stale.json()["error"]

    asyncio.run(scenario())


def test_failed_preparation_keeps_old_version_and_allows_retry(monkeypatch, tmp_path):
    import threading

    httpx = pytest.importorskip("httpx")
    entered, release = threading.Event(), threading.Event()
    attempts = []

    def prepare(version, **kwargs):
        attempts.append(version)
        if len(attempts) == 1:
            entered.set()
            assert release.wait(5)
            raise ValueError("corrupt adapter")
        path = tmp_path / version
        path.mkdir()
        return str(path)

    monkeypatch.setattr(refresh, "prepare_adapter", prepare)
    app, models, calls, _ = serving_app(monkeypatch)
    install_routes(app, str(tmp_path), "/base", "s3://bucket/skyrl-checkpoints")
    app.init_state()

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
            pending = asyncio.create_task(sample(client, "model_ab12cd34_step-1"))
            assert await asyncio.to_thread(entered.wait, 5)
            try:
                # The old version keeps serving while the new one downloads and merges.
                assert (await sample(client, "base")).json()["model"] == "base"
                assert (await client.get("/health")).status_code == 200
            finally:
                release.set()
            response = await pending
            assert response.status_code == 500 and response.json()["restart_required"] is False
            assert models.base_model_paths[0].name == "base" and not calls
            retry = await sample(client, "model_ab12cd34_step-1")
            assert retry.json()["model"] == "model_ab12cd34_step-1"

    asyncio.run(scenario())


def test_successful_refresh_removes_replaced_merged_checkpoint_but_never_the_base(monkeypatch, tmp_path):
    httpx = pytest.importorskip("httpx")

    def prepare(version, **kwargs):
        path = tmp_path / version
        path.mkdir()
        return str(path)

    monkeypatch.setattr(refresh, "prepare_adapter", prepare)
    app, models, _, _ = serving_app(monkeypatch)
    base = tmp_path.parent / "immutable-base"
    base.mkdir(exist_ok=True)
    models.base_model_paths[0].model_path = str(base)
    install_routes(app, str(tmp_path), str(base), "s3://bucket/skyrl-checkpoints")
    app.init_state()

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
            assert (await sample(client, "model_ab12cd34_step-1")).status_code == 200
            assert base.is_dir() and (tmp_path / "model_ab12cd34_step-1").is_dir()
            assert (await sample(client, "model_ab12cd34_step-2")).status_code == 200
            assert not (tmp_path / "model_ab12cd34_step-1").exists()
            assert (tmp_path / "model_ab12cd34_step-2").is_dir() and base.is_dir()

    asyncio.run(scenario())


class FakeS3:
    def __init__(self, objects, missing_code="404"):
        self.objects, self.gets, self.missing_code = objects, [], missing_code

    def head_object(self, Bucket, Key):
        from botocore.exceptions import ClientError

        if (Bucket, Key) not in self.objects:
            # Without s3:ListBucket, S3 reports a missing key as 403.
            raise ClientError({"Error": {"Code": self.missing_code}}, "HeadObject")
        return {}

    def get_object(self, Bucket, Key):
        import io

        self.gets.append(Key)
        data = self.objects[(Bucket, Key)]
        return {"ContentLength": len(data), "Body": io.BytesIO(data)}


def sampler_archive(files):
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=0) as tar:
        for name, data in files.items():
            info = tarfile.TarInfo("./" + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.mark.parametrize("missing_code", ["404", "403"])
def test_model_name_resolves_to_exactly_one_skyrl_sampler_archive(missing_code):
    pytest.importorskip("botocore")
    key = "skyrl/model_ab12cd34/sampler_weights/run_1-step-2.tar.gz"
    client = FakeS3({("bucket", key): b"x"}, missing_code)
    assert refresh.sampler_checkpoint_key("model_ab12cd34_run_1-step-2", "s3://bucket/skyrl", client) == ("bucket", key)
    with pytest.raises(LookupError):
        refresh.sampler_checkpoint_key("model_ab12cd34_missing", "s3://bucket/skyrl", client)
    ambiguous = FakeS3(
        {
            ("bucket", "skyrl/a/sampler_weights/b_c.tar.gz"): b"x",
            ("bucket", "skyrl/a_b/sampler_weights/c.tar.gz"): b"x",
        }
    )
    with pytest.raises(LookupError, match="ambiguous"):
        refresh.sampler_checkpoint_key("a_b_c", "s3://bucket/skyrl", ambiguous)


def test_prepare_adapter_extracts_skyrl_archive_and_reuses_completed_merge(monkeypatch, tmp_path):
    pytest.importorskip("botocore")
    adapter = {
        "adapter_config.json": b'{"peft_type": "LORA"}',
        "adapter_model.safetensors": b"weights",
        "tokenizer.json": b"{}",
    }
    key = "skyrl/model_ab12cd34/sampler_weights/step-1.tar.gz"
    client = FakeS3({("bucket", key): sampler_archive(adapter)})
    merged = []

    def merge(base_path, adapter_path, output_path, version):
        merged.append({p.name: p.read_bytes() for p in Path(adapter_path).iterdir()})
        output_path.mkdir()
        return output_path

    monkeypatch.setattr(refresh, "merge_checkpoint", merge)
    options = dict(base_path="/base", output_root=tmp_path, checkpoints_prefix="s3://bucket/skyrl", client=client)
    assert refresh.prepare_adapter("model_ab12cd34_step-1", **options) == tmp_path / "model_ab12cd34_step-1"
    assert merged == [adapter]
    assert refresh.prepare_adapter("model_ab12cd34_step-1", **options) == tmp_path / "model_ab12cd34_step-1"
    assert len(merged) == 1 and client.gets == [key]
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".adapter-")]

    broken = FakeS3(
        {
            ("bucket", "skyrl/model_ab12cd34/sampler_weights/step-2.tar.gz"): sampler_archive(
                {"adapter_config.json": b'{"peft_type": "LORA"}'}
            )
        }
    )
    with pytest.raises(ValueError, match="adapter_model.safetensors"):
        refresh.prepare_adapter("model_ab12cd34_step-2", **{**options, "client": broken})


def test_unknown_model_name_is_404_and_leaves_serving_unchanged(monkeypatch, tmp_path):
    httpx = pytest.importorskip("httpx")

    def prepare(version, **kwargs):
        raise LookupError(f"No unique readable SkyRL sampler checkpoint for model {version!r}")

    monkeypatch.setattr(refresh, "prepare_adapter", prepare)
    app, models, calls, _ = serving_app(monkeypatch)
    install_routes(app, str(tmp_path), "/base", "s3://bucket/skyrl-checkpoints")
    app.init_state()

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
            response = await sample(client, "model_ab12cd34_missing")
            assert response.status_code == 404 and response.json()["restart_required"] is False
            assert (await sample(client, "base")).json()["model"] == "base" and not calls

    asyncio.run(scenario())


@pytest.mark.parametrize("requested,forwarded", [(True, 0), (1, 0), (5, 0), (0, 0), (None, None)])
def test_completion_logprobs_are_limited_to_the_sampled_token(monkeypatch, tmp_path, requested, forwarded):
    httpx = pytest.importorskip("httpx")
    app, _, _, _ = serving_app(monkeypatch)
    install_routes(app, str(tmp_path), "/base", "s3://bucket/skyrl-checkpoints")
    app.init_state()

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
            body = {"model": "base", "prompt": [1, 2]}
            if requested is not None:
                body["logprobs"] = requested
            response = await client.post("/v1/completions", json=body)
            assert response.status_code == 200
            assert response.json()["logprobs"] == forwarded

    asyncio.run(scenario())


def test_model_support_is_selected_by_checkpoint_model_type(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    assert refresh.model_support(tmp_path).MODEL_TYPE == "qwen3_5"
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with pytest.raises(ValueError, match="does not support model_type 'llama'"):
        refresh.model_support(tmp_path)


def test_worker_rejects_a_model_class_other_than_the_supported_one(monkeypatch, tmp_path):
    from types import SimpleNamespace

    pytest.importorskip("torch")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))

    class Supported:
        pass

    monkeypatch.setattr(refresh.MODEL_SUPPORT["qwen3_5"], "neuron_model_class", lambda: Supported)
    worker = refresh.WeightRefreshWorker()
    worker.model_runner = SimpleNamespace(model=object(), drafter=None, vision_neuron_config=None)
    with pytest.raises(ValueError, match="qwen3_5 requires its validated Neuron model"):
        worker.stage_checkpoint(str(tmp_path), "v1")


def test_api_key_is_checked_before_a_new_model_name_starts_a_refresh(monkeypatch, tmp_path):
    httpx = pytest.importorskip("httpx")
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    prepared = []

    def prepare(version, **kwargs):
        prepared.append(version)
        path = tmp_path / version
        path.mkdir()
        return str(path)

    monkeypatch.setattr(refresh, "prepare_adapter", prepare)
    app, models, calls, gate = serving_app(monkeypatch)
    install_routes(app, str(tmp_path), "/base", "s3://bucket/skyrl-checkpoints")
    app.init_state()

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            body = {"model": "model_ab_ckpt", "prompt": [1, 2]}
            missing = await client.post("/v1/completions", json=body)
            wrong = await client.post("/v1/completions", json=body, headers={"Authorization": "Bearer nope"})
            assert missing.status_code == wrong.status_code == 401
            assert prepared == []  # Rejected before resolution: no refresh started.
            ok = await client.post("/v1/completions", json=body, headers={"Authorization": "Bearer secret"})
            assert ok.status_code == 200 and ok.json()["model"] == "model_ab_ckpt"
            assert prepared == ["model_ab_ckpt"]

    asyncio.run(run())
