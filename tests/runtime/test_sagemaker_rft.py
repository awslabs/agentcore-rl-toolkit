"""Real SDK composition tests and an opt-in SageMaker Job Runtime integration test.

Run: uv run --group sagemaker-integration pytest tests/runtime/test_sagemaker_rft.py
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
