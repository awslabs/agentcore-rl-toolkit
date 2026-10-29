"""Real SDK composition tests and an opt-in SageMaker Job Runtime integration test.

Run: uv run --extra server --group sagemaker-integration pytest tests/runtime/test_sagemaker_rft.py
"""

import asyncio
import json
import os
import threading

import httpx
import pytest

from agentcore_rl_toolkit import AgentCoreRuntimeApp

rft = pytest.importorskip("sagemaker.train.rft")

from openai import OpenAI  # noqa: E402
from sagemaker.core.token_generator import generate_token  # noqa: E402
from sagemaker.train.rft.context import get_inference_params  # noqa: E402
from sagemaker.train.rft.headers import get_inference_headers  # noqa: E402


async def completed(client, invocation_id, timeout=5):
    async with asyncio.timeout(timeout):
        while True:
            response = await client.post(
                "/invocations",
                json={"_agentcore_runtime": {"version": 1, "operation": "get", "invocation_id": invocation_id}},
            )
            response.raise_for_status()
            if response.json()["status"] != "in_progress":
                return response.json()
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_handler", [False, True])
@pytest.mark.parametrize("fail", [False, True])
async def test_rft_feedback_follows_background_handler(tmp_path, monkeypatch, async_handler, fail):
    """Use the real SDK decorator, recording feedback HTTP calls without sending them."""
    app = AgentCoreRuntimeApp(background=True, state_dir=tmp_path)
    release = threading.Event()
    feedback = []

    def record_feedback(self, path, body):
        feedback.append((path, json.loads(body)))

    monkeypatch.setattr(rft.RolloutFeedbackClient, "_bearer_post", record_feedback)
    metadata = {"jobArn": "test-job", "rolloutId": "test-rollout"}

    def run(payload):
        assert get_inference_headers() == {
            "X-Amzn-SageMaker-Job-Arn": metadata["jobArn"],
            "X-Amzn-SageMaker-Trajectory-Id": metadata["rolloutId"],
        }
        assert get_inference_params() == payload["inferenceParams"]
        assert release.wait(5)
        if fail:
            raise ValueError("agent failed")
        return {"reward": 0.5}

    async def run_async(payload):
        return await asyncio.to_thread(run, payload)

    app.entrypoint(rft.sagemaker_rft_handler(run_async if async_handler else run))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://agent") as client:
        try:
            response = await client.post(
                "/invocations",
                json={"prompt": "test", "metadata": metadata, "inferenceParams": {"temperature": 0.5}},
            )
            assert response.status_code == 200
            assert response.json()["status"] == "in_progress"
            assert feedback == []
        finally:
            release.set()
        result = await completed(client, response.json()["invocation_id"])

    if fail:
        assert result["error"] == "agent failed"
    else:
        assert result["result"] == {"reward": 0.5}
    identity = {"JobArn": metadata["jobArn"], "TrajectoryId": metadata["rolloutId"]}
    assert feedback == [
        ("/complete-rollout", {**identity, "Status": "failed" if fail else "ready"}),
        ("/update-reward", {**identity, "Rewards": [0.0 if fail else 0.5]}),
    ]


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("SAGEMAKER_RFT_TEST_ENDPOINT"),
    reason="set SAGEMAKER_RFT_TEST_ENDPOINT, SAGEMAKER_RFT_TEST_JOB_ARN and SAGEMAKER_RFT_TEST_ROLLOUT_ID",
)
async def test_live_rft_background_feedback(tmp_path, monkeypatch):
    """Run the app locally against real SageMaker inference and feedback APIs.

    Set SAGEMAKER_RFT_TEST_ENDPOINT to the service base URL (without /v1),
    SAGEMAKER_RFT_TEST_JOB_ARN to an active job with an available model session,
    and SAGEMAKER_RFT_TEST_ROLLOUT_ID to a dedicated pending rollout. Requires
    AWS credentials authorized for that job; completes the rollout with reward 1.0.
    """
    endpoint = os.environ["SAGEMAKER_RFT_TEST_ENDPOINT"].rstrip("/")
    job_arn = os.environ["SAGEMAKER_RFT_TEST_JOB_ARN"]
    rollout_id = os.environ["SAGEMAKER_RFT_TEST_ROLLOUT_ID"]
    region = job_arn.split(":")[3]
    app = AgentCoreRuntimeApp(background=True, state_dir=tmp_path)
    release = threading.Event()
    accepted_feedback = []
    bearer_post = rft.RolloutFeedbackClient._bearer_post

    def record_accepted_feedback(self, path, body):
        # Observe successful HTTP calls; the decorator can swallow feedback errors.
        bearer_post(self, path, body)
        accepted_feedback.append((path, json.loads(body)))

    monkeypatch.setattr(rft.RolloutFeedbackClient, "_bearer_post", record_accepted_feedback)

    @app.entrypoint
    @rft.sagemaker_rft_handler
    def handler(payload):
        assert release.wait(5)
        with OpenAI(
            api_key=generate_token(region=region),
            base_url=endpoint + "/v1",
            default_headers=get_inference_headers(),
            timeout=60,
            max_retries=0,
        ) as model:
            response = model.chat.completions.create(
                model="default",
                messages=[{"role": "user", "content": payload["prompt"]}],
                max_tokens=16,
            )
        assert response.choices
        return {"reward": 1.0}

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://agent") as client:
        try:
            response = await client.post(
                "/invocations",
                json={
                    "prompt": "Reply with OK.",
                    "metadata": {"jobArn": job_arn, "rolloutId": rollout_id, "endpoint": endpoint, "region": region},
                },
            )
            assert response.status_code == 200
            assert response.json()["status"] == "in_progress"
            assert accepted_feedback == []
        finally:
            release.set()
        result = await completed(client, response.json()["invocation_id"], timeout=360)

    assert result["result"] == {"reward": 1.0}
    identity = {"JobArn": job_arn, "TrajectoryId": rollout_id}
    assert accepted_feedback == [
        ("/complete-rollout", {**identity, "Status": "ready"}),
        ("/update-reward", {**identity, "Rewards": [1.0]}),
    ]


ROW = {"task_id": "t-1", "question": "2 + 2?"}
TOKEN = "sagemaker-api-key-test"
# The loopback stand-in is not a SageMaker endpoint, so tests trust it deliberately.
LOOPBACK = r"http://127\.0\.0\.1:\d+"
LOCAL = "http://127.0.0.1:9"


@pytest.fixture
def sagemaker(monkeypatch):
    """Record SageMaker feedback, and stand in a loopback inference endpoint the proxy may call."""
    import sagemaker.core.token_generator as token_generator
    from aiohttp import web

    calls = {"feedback": [], "inference": []}
    monkeypatch.setattr(
        rft.RolloutFeedbackClient,
        "_bearer_post",
        lambda self, path, body: calls["feedback"].append((path, json.loads(body))),
    )
    monkeypatch.setattr(token_generator, "generate_token", lambda region=None, **_: TOKEN)

    async def completions(request):
        body = await request.json()
        calls["inference"].append((request.headers.copy(), body))
        if not body.get("stream"):
            return web.json_response(
                {
                    "id": "c-1",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body["model"],
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "4"}, "finish_reason": "stop"}
                    ],
                }
            )
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for content in ("4", "!"):
            chunk = {
                "id": "c-1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def start():
        server = web.Application()
        server.router.add_post("/v1/chat/completions", completions)
        runner = web.AppRunner(server)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        endpoint = f"http://127.0.0.1:{runner.addresses[0][1]}"
        calls["endpoint"] = endpoint
        return runner

    calls["start"] = start
    return calls


def rft_request(endpoint, **overrides):
    return {
        "prompt": json.dumps(ROW),
        "metadata": {"jobArn": "test-job", "rolloutId": "test-rollout", "endpoint": endpoint, "region": "us-west-2"},
        "inferenceParams": {"temperature": 0.5, "maxTokens": 64},
        **overrides,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("async_handler", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_sagemaker_mtrl_serves_a_config_entrypoint(tmp_path, sagemaker, async_handler, stream):
    """A SageMaker request reaches a `(payload, context)` handler as the `_config` contract."""
    from openai import AsyncOpenAI

    from agentcore_rl_toolkit import sagemaker_mtrl

    runner = await sagemaker["start"]()
    app = AgentCoreRuntimeApp(background=True, state_dir=tmp_path)
    seen = []

    def complete(payload):
        config = payload["_config"]
        with OpenAI(base_url=config["base_url"], api_key=config["api_key"], max_retries=0) as model:
            request = dict(model=config["model_id"], messages=[{"role": "user", "content": "q"}], temperature=1.0)
            if not stream:
                return model.chat.completions.create(**request).choices[0].message.content
            return "".join(
                c.choices[0].delta.content or "" for c in model.chat.completions.create(**request, stream=True)
            )

    async def complete_async(payload):
        config = payload["_config"]
        async with AsyncOpenAI(base_url=config["base_url"], api_key=config["api_key"], max_retries=0) as model:
            request = dict(model=config["model_id"], messages=[{"role": "user", "content": "q"}], temperature=1.0)
            if not stream:
                return (await model.chat.completions.create(**request)).choices[0].message.content
            chunks = await model.chat.completions.create(**request, stream=True)
            return "".join([c.choices[0].delta.content or "" async for c in chunks])

    if async_handler:

        @app.entrypoint
        @sagemaker_mtrl(trusted_endpoint=LOOPBACK)
        async def handler(payload, context):
            seen.append((payload, context))
            return {"reward": 0.5, "answer": await complete_async(payload)}

    else:

        @app.entrypoint
        @sagemaker_mtrl(trusted_endpoint=LOOPBACK)
        def handler(payload, context):
            seen.append((payload, context))
            return {"reward": 0.5, "answer": complete(payload)}

    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://agent") as client:
            response = await client.post("/invocations", json=rft_request(sagemaker["endpoint"]))
            assert response.json()["status"] == "in_progress"
            result = await completed(client, response.json()["invocation_id"])
    finally:
        await runner.cleanup()

    assert result["result"] == {"reward": 0.5, "answer": "4!" if stream else "4"}
    ((payload, context),) = seen
    assert context is not None
    config = payload.pop("_config")
    assert payload == ROW
    assert config["base_url"].startswith("http://127.0.0.1:") and config["model_id"] == "default"
    ((headers, body),) = sagemaker["inference"]
    assert headers["Authorization"] == f"Bearer {TOKEN}"
    assert headers["X-Amzn-SageMaker-Job-Arn"] == "test-job"
    assert headers["X-Amzn-SageMaker-Trajectory-Id"] == "test-rollout"
    assert (body["temperature"], body["max_tokens"]) == (0.5, 64)
    identity = {"JobArn": "test-job", "TrajectoryId": "test-rollout"}
    assert sagemaker["feedback"] == [
        ("/complete-rollout", {**identity, "Status": "ready"}),
        ("/update-reward", {**identity, "Rewards": [0.5]}),
    ]
    # The rollout's key stops routing once the rollout ends.
    async with httpx.AsyncClient() as client:
        response = await client.post(
            config["base_url"] + "/chat/completions", headers={"Authorization": f"Bearer {config['api_key']}"}, json={}
        )
    assert response.status_code == 401


@pytest.mark.parametrize(
    ("result", "error"),
    [
        ({"answer": "4"}, "entrypoint returned no finite scalar reward (reward=None)"),
        ({"reward": float("nan")}, "entrypoint returned no finite scalar reward (reward=nan)"),
        ({"reward": True}, "entrypoint returned no finite scalar reward (reward=True)"),
        *(
            ({"reward": 0.5, "turn_rewards": turn_rewards}, f"entrypoint returned invalid {detail}")
            for turn_rewards, detail in [
                ([], "turn_rewards=[]; expected a non-empty list of finite numbers"),
                ([0.5, float("nan")], "turn_rewards=[0.5, nan]; expected a non-empty list of finite numbers"),
                ([True], "turn_rewards=[True]; expected a non-empty list of finite numbers"),
                (0.5, "turn_rewards=0.5; expected a non-empty list of finite numbers"),
            ]
        ),
    ],
)
def test_sagemaker_mtrl_fails_a_rollout_without_a_reward(sagemaker, result, error):
    from agentcore_rl_toolkit import sagemaker_mtrl

    returned = sagemaker_mtrl(trusted_endpoint=LOOPBACK)(lambda payload: result)(rft_request(LOCAL))

    assert returned == {**result, "status": "error", "error": error}
    identity = {"JobArn": "test-job", "TrajectoryId": "test-rollout"}
    assert sagemaker["feedback"] == [
        ("/complete-rollout", {**identity, "Status": "failed"}),
        ("/update-reward", {**identity, "Rewards": [0.0]}),
    ]


@pytest.mark.parametrize(
    "metadata",
    [
        {"jobArn": "test-job", "rolloutId": "test-rollout"},
        # The SDK's RolloutMetadata field names; the SDK resolves either spelling.
        {"job_arn": "test-job", "trajectory_id": "test-rollout"},
    ],
)
def test_sagemaker_mtrl_reports_turn_rewards(sagemaker, metadata):
    """`turn_rewards` reaches SageMaker as per-turn rewards; the record keeps the scalar for ART's backends."""
    from agentcore_rl_toolkit import sagemaker_mtrl

    result = {"reward": 0.5, "turn_rewards": [0.0, 0.5, 1]}
    request = rft_request(LOCAL, metadata={**metadata, "endpoint": LOCAL, "region": "us-west-2"})
    returned = sagemaker_mtrl(trusted_endpoint=LOOPBACK)(lambda payload: result)(request)

    assert returned == result
    identity = {"JobArn": "test-job", "TrajectoryId": "test-rollout"}
    assert sagemaker["feedback"] == [
        ("/complete-rollout", {**identity, "Status": "ready"}),
        ("/update-reward", {**identity, "Rewards": [0.0, 0.5, 1]}),
    ]


@pytest.mark.parametrize(
    ("request_overrides", "error"),
    [
        ({"prompt": json.dumps({**ROW, "_config": {}})}, "reserved field '_config'"),
        ({"prompt": "[1]"}, "must be a JSON object"),
        ({"metadata": {"jobArn": "test-job", "endpoint": LOCAL}}, "must name both the job and the trajectory"),
    ],
)
def test_sagemaker_mtrl_fails_an_unusable_rollout(sagemaker, request_overrides, error):
    from agentcore_rl_toolkit import sagemaker_mtrl

    handler = sagemaker_mtrl(trusted_endpoint=LOOPBACK)(lambda payload: pytest.fail("the handler must not run"))
    with pytest.raises((TypeError, ValueError), match=error):
        handler(rft_request(LOCAL, **request_overrides))

    # The SDK reports a failure only for a trajectory it can name.
    if "rolloutId" in rft_request(LOCAL, **request_overrides)["metadata"]:
        assert [body["Status"] for path, body in sagemaker["feedback"] if path == "/complete-rollout"] == ["failed"]
    else:
        assert sagemaker["feedback"] == []


@pytest.mark.parametrize(
    "endpoint",
    [
        None,
        "",
        "https://attacker.example",
        "http://job-runtime.sagemaker.us-west-2.api.aws",
        "https://job-runtime.sagemaker.us-west-2.api.aws.example",
        "https://job-runtime.sagemaker.us-west-2.api.aws/path",
        "https://job-runtime.sagemaker.us-west-2.api.aws:8443",
    ],
)
def test_sagemaker_mtrl_refuses_an_endpoint_that_is_not_sagemakers(sagemaker, monkeypatch, endpoint):
    """A refused endpoint gets no inference call and no feedback: both would carry the Runtime's token."""
    from agentcore_rl_toolkit import sagemaker_mtrl

    # The SDK would fall back to this for a missing endpoint; ART does not.
    monkeypatch.setenv("RFT_RUNTIME_ENDPOINT", "https://job-runtime.sagemaker.us-west-2.api.aws")
    handler = sagemaker_mtrl(lambda payload: pytest.fail("the handler must not run"))
    with pytest.raises(ValueError, match="names no endpoint|does not match the trusted endpoint pattern"):
        handler(rft_request(endpoint))
    assert sagemaker["feedback"] == []


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://job-runtime.sagemaker.us-west-2.api.aws",
        "https://job-runtime.beta.sagemaker.us-east-1.api.aws/",
    ],
)
def test_sagemaker_mtrl_trusts_sagemaker_endpoints(sagemaker, endpoint):
    from agentcore_rl_toolkit import sagemaker_mtrl

    assert sagemaker_mtrl(lambda payload: {"reward": 1.0})(rft_request(endpoint)) == {"reward": 1.0}
    assert [body["Status"] for path, body in sagemaker["feedback"] if path == "/complete-rollout"] == ["ready"]


def test_sagemaker_mtrl_trusts_a_configured_endpoint_pattern(sagemaker):
    from agentcore_rl_toolkit import sagemaker_mtrl

    handler = sagemaker_mtrl(trusted_endpoint=r"https://rft\.example\.com")(lambda payload: {"reward": 1.0})
    assert handler(rft_request("https://rft.example.com")) == {"reward": 1.0}

    with pytest.raises(ValueError, match="does not match the trusted endpoint pattern"):
        handler(rft_request("https://job-runtime.sagemaker.us-west-2.api.aws"))


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "q", "_config": {"base_url": "http://gateway/v1", "model_id": "m", "api_key": "sid"}},
        # A slime row's own `prompt` and `metadata` do not select SageMaker while `_config` is present.
        {"prompt": "{}", "metadata": {"jobArn": "row-provenance"}, "_config": {"base_url": "http://gateway/v1"}},
        {"metadata": {"source": "gsm8k"}, "question": "q"},
    ],
)
def test_sagemaker_mtrl_passes_other_callers_through(sagemaker, payload):
    from agentcore_rl_toolkit import sagemaker_mtrl

    def handler(payload, context):
        return {"payload": payload, "context": context}

    assert sagemaker_mtrl(handler)(payload, "ctx") == {"payload": payload, "context": "ctx"}
    assert sagemaker["feedback"] == []


def test_sagemaker_mtrl_rejects_streaming_handlers():
    from agentcore_rl_toolkit import sagemaker_mtrl

    def handler(payload):
        yield {}

    with pytest.raises(TypeError, match="not a generator"):
        sagemaker_mtrl(handler)


def test_sagemaker_mtrl_fails_a_rollout_whose_status_is_error(sagemaker):
    from agentcore_rl_toolkit import sagemaker_mtrl

    result = {"status": "error", "error": "agent failed", "reward": 0.0}

    assert sagemaker_mtrl(trusted_endpoint=LOOPBACK)(lambda payload: result)(rft_request(LOCAL)) == result
    identity = {"JobArn": "test-job", "TrajectoryId": "test-rollout"}
    assert sagemaker["feedback"] == [
        ("/complete-rollout", {**identity, "Status": "failed"}),
        ("/update-reward", {**identity, "Rewards": [0.0]}),
    ]


def test_sagemaker_mtrl_surfaces_feedback_failures(monkeypatch):
    """A failed success report reaches the invocation record; a failed failure report keeps the cause."""
    import sagemaker.core.token_generator as token_generator

    from agentcore_rl_toolkit import sagemaker_mtrl

    def unreachable(self, path, body):
        raise RuntimeError("feedback unreachable")

    monkeypatch.setattr(rft.RolloutFeedbackClient, "_bearer_post", unreachable)
    monkeypatch.setattr(token_generator, "generate_token", lambda region=None, **_: TOKEN)
    with pytest.raises(RuntimeError, match="feedback unreachable"):
        sagemaker_mtrl(trusted_endpoint=LOOPBACK)(lambda payload: {"reward": 1.0})(rft_request(LOCAL))

    def broken(payload):
        raise ValueError("agent failed")

    with pytest.raises(ValueError, match="agent failed"):
        sagemaker_mtrl(trusted_endpoint=LOOPBACK)(broken)(rft_request(LOCAL))


# Contract checks against the installed SDK. ART leaves metadata spellings, headers, tokens, and
# feedback to the SDK; these pin the two SDK behaviors it relies on, so an SDK upgrade that changes
# either fails here rather than in a training job.


@pytest.mark.parametrize(
    "env",
    [{}, {"RFT_RUNTIME_ENDPOINT": "https://job-runtime.gamma.sagemaker.us-east-1.api.aws"}, {"RFT_STAGE": "beta"}],
)
def test_sdk_feedback_uses_the_metadata_endpoint(monkeypatch, env):
    """Inference goes to `metadata.endpoint`; feedback must go there too."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    endpoint = "https://job-runtime.sagemaker.us-west-2.api.aws"
    posted = []
    monkeypatch.setattr(rft.RolloutFeedbackClient, "_bearer_post", lambda self, path, body: posted.append(self))
    client = rft.RolloutFeedbackClient({"jobArn": "j", "rolloutId": "r", "endpoint": endpoint + "/"})
    client.complete_rollout("ready")

    assert posted == [client] and client._endpoint == endpoint


@pytest.mark.parametrize(
    "params",
    [{"temperature": 0.5}, {"maxTokens": 64}, {"topP": 0.9, "temperature": None}, {"stream": True}],
)
def test_sampling_params_match_the_sdk_strands_adapter(params):
    """The one mapping ART implements itself: `inferenceParams` fields to OpenAI request fields."""
    from sagemaker.train.rft.adapters.strands import wrap_model

    from agentcore_rl_toolkit.runtime.sagemaker_mtrl import sampling_params

    class Model:
        def __init__(self):
            self.config = {"params": {}}
            self.client_args = {}

        def update_config(self, **config):
            self.config.update(config)

        def stream(self, *args, **kwargs):
            return None

    model = Model()
    rft.set_rollout_context({"jobArn": "j", "rolloutId": "r"}, params)
    try:
        wrap_model(model).stream()
    finally:
        rft.clear_rollout_context()

    assert sampling_params(params) == model.config["params"]
