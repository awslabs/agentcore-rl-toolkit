"""AgentCore loop manager for verl integration.

Manages the dispatch of rollout requests to AgentCore Runtime via RolloutClient,
collects rewards from S3 and token traces from the model gateway, then
post-processes them into segment-level DataProto for training.
"""

import asyncio
import logging
import time
import uuid
from collections import Counter

import numpy as np
import torch
from omegaconf import DictConfig
from rllm_model_gateway import AsyncGatewayClient
from tensordict import TensorDict
from verl.protocol import DataProto
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

from agentcore_rl_toolkit import RolloutClient

logger = logging.getLogger(__name__)


def _pad_sequence_to_length(tensor, target_length, pad_value, left_pad=False):
    """Pad a 2D tensor along dim=1 to reach target_length."""
    current_length = tensor.shape[1]
    if current_length >= target_length:
        return tensor
    pad_size = target_length - current_length
    padding = torch.full(
        (tensor.shape[0], pad_size),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    if left_pad:
        return torch.cat([padding, tensor], dim=1)
    else:
        return torch.cat([tensor, padding], dim=1)


class AgentCoreLoopManager:
    """Manager that dispatches rollout requests to AgentCore Runtime and collects results.

    Implements the same interface as verl's AgentLoopManager:
        generate_sequences(batch: DataProto) -> DataProto
    """

    def __init__(self, config: DictConfig, gateway_url: str):
        """Initialize AgentCore loop manager.

        Args:
            config: Global verl config.
            gateway_url: URL of the rllm-model-gateway.
        """
        self.config = config
        self._gateway_url = gateway_url

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

        self.exp_id = self._generate_exp_id()
        self._initialize_rollout_client()

    def _generate_exp_id(self) -> str:
        agent_runtime_arn = self.config.actor_rollout_ref.rollout.agentcore.agent_runtime_arn
        agent_id = agent_runtime_arn.split("/")[-1]
        timestamp = time.strftime("%Y%m%d%H%M%S")
        return f"{agent_id}_{timestamp}"

    def _initialize_rollout_client(self):
        """Initialize RolloutClient for dispatching requests and collecting results."""
        agentcore_config = self.config.actor_rollout_ref.rollout.agentcore

        self.rollout_client = RolloutClient(
            agent_runtime_arn=agentcore_config.agent_runtime_arn,
            s3_bucket=agentcore_config.s3_bucket,
            exp_id=self.exp_id,
            model_id=self.config.actor_rollout_ref.model.path,
            tps_limit=agentcore_config.reqs_per_sec,
            max_pool_connections=agentcore_config.max_pool_connections,
        )

    # -- Dispatch and collect -------------------------------------------------

    async def _process_single_task(self, async_client, payload, uid, sampling_params, max_time):
        """Process one task: create gateway session -> dispatch -> collect reward -> fetch traces -> cleanup.

        Returns:
            Dict with rollout data, or dict with ``_error`` key on failure.
        """
        session_id = str(uuid.uuid4())

        # Create gateway session
        await async_client.create_session(session_id=session_id)
        session_url = async_client.get_session_url(session_id)

        # Dispatch to AgentCore with gateway URL as inference endpoint
        reward = 0.0
        success = False
        dispatch_error = None
        try:
            future = await self.rollout_client.invoke_async(
                payload,
                session_id=session_id,
                input_id=uid,
                base_url=session_url,
                sampling_params=sampling_params,
            )
            raw = await future.result_async(timeout=max_time)
            reward, success = self._extract_reward_from_s3(raw)
        except Exception as e:
            dispatch_error = str(e)
            logger.warning(f"Task {uid} (session {session_id[:8]}...) failed: {e}")
            reward = 0.0
            success = False

        # Fetch traces from gateway
        traces = []
        try:
            await async_client.flush()
            traces = await async_client.get_session_traces(session_id)
        except Exception as e:
            logger.warning(f"Failed to fetch gateway traces for {uid} (session {session_id[:8]}...): {e}")

        # Cleanup gateway session
        try:
            await async_client.delete_session(session_id)
        except Exception:
            pass

        # Convert traces to step format
        steps = self._traces_to_steps(traces)
        if not steps:
            reason = dispatch_error or "no traces captured (agent never called LLM)"
            return {"_error": reason, "uid": uid, "session_id": session_id}

        return {
            "rollout": steps,
            "reward": reward,
            "uid": uid,
            "success": success,
            "session_id": session_id,
        }

    async def _dispatch_and_collect(
        self, rollout_batch_input: DataProto, rollout_sampling_params: dict = None
    ) -> list[dict]:
        """Dispatch rollout requests and collect results.

        Creates AsyncGatewayClient fresh inside this coroutine so the TCP
        connections live in the same event loop.
        """
        max_time = self.config.actor_rollout_ref.rollout.agentcore.max_rollout_time
        target_size = len(rollout_batch_input)
        max_concurrent = self.config.actor_rollout_ref.rollout.agentcore.max_pool_connections
        semaphore = asyncio.Semaphore(max_concurrent)

        async_client = AsyncGatewayClient(self._gateway_url)

        start_time = time.time()

        async def _run_with_semaphore(payload, uid):
            async with semaphore:
                return await self._process_single_task(
                    async_client, payload, uid, rollout_sampling_params or {}, max_time
                )

        tasks = []
        for i in range(target_size):
            non_tensor_item = rollout_batch_input[i].non_tensor_batch
            payload = {k: v for k, v in non_tensor_item.items()}

            if "uid" not in payload:
                raise ValueError(
                    "Missing 'uid' in input data. UIDs are required to group outputs by input prompts "
                    "during advantage computation."
                )

            uid = payload.pop("uid")
            tasks.append(asyncio.create_task(_run_with_semaphore(payload, uid)))

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await async_client.close()

        rollout_data_batch = []
        failure_reasons = []
        for r in results:
            if isinstance(r, Exception):
                failure_reasons.append(str(r))
            elif isinstance(r, dict) and "_error" in r:
                failure_reasons.append(f"uid={r.get('uid', '?')}: {r['_error']}")
            elif r is not None:
                rollout_data_batch.append(r)
            else:
                failure_reasons.append("unknown (returned None)")

        elapsed = time.time() - start_time
        n_failures = len(failure_reasons)
        logger.warning(
            f"{elapsed:.1f}s elapsed. {len(rollout_data_batch)} rollouts collected out of "
            f"{target_size} launched ({n_failures} failures)."
        )

        if n_failures > 0:
            reason_counts = Counter(failure_reasons)
            lines = [f"Failure reasons ({n_failures} total):"]
            for reason, count in reason_counts.most_common():
                lines.append(f"  [{count}x] {reason}")
            logger.warning("\n".join(lines))

        if len(rollout_data_batch) == 0:
            raise RuntimeError("No rollout data received. Check previous logs for dispatch or agent failures.")

        if len(rollout_data_batch) < target_size:
            logger.warning(f"{target_size - len(rollout_data_batch)} rollout traces are missing.")

        return rollout_data_batch

    # -- Data processing helpers -----------------------------------------------

    def _extract_reward_from_s3(self, s3_data: dict) -> tuple[float, bool]:
        """Extract reward from S3 result."""
        status_code = s3_data.get("status_code")
        success = status_code == 200

        if not success and status_code:
            stop_reason = s3_data.get("stop_reason", "unknown")
            input_id = s3_data.get("input_id", "?")
            logger.warning(
                f"Rollout failed for input_id={input_id}: status_code={status_code}, stop_reason={stop_reason}"
            )

        rewards = s3_data.get("rewards")
        if rewards is None:
            return 0.0, False

        if isinstance(rewards, list):
            reward = float(rewards[-1]) if rewards else 0.0
        else:
            reward = float(rewards)

        return reward, success

    @staticmethod
    def _traces_to_steps(traces) -> list[dict]:
        """Convert gateway TraceRecord objects to step dicts.

        Maps:
            TraceRecord.prompt_token_ids     -> step["prompt_ids"]
            TraceRecord.completion_token_ids -> step["response_ids"]
            TraceRecord.logprobs             -> step["response_logprobs"]
        """
        steps = []
        for i, trace in enumerate(traces):
            if not trace.prompt_token_ids and not trace.completion_token_ids:
                continue
            steps.append(
                {
                    "step_index": i,
                    "prompt_ids": trace.prompt_token_ids,
                    "response_ids": trace.completion_token_ids,
                    "response_logprobs": trace.logprobs if trace.logprobs else [],
                }
            )
        return steps

    # -- Post-processing -------------------------------------------------------

    @staticmethod
    def _merge_steps_to_segments(steps: list[dict]) -> list[dict]:
        """Merge a trajectory's steps into prefix-cumulative segments.

        Consecutive steps whose prompt is a prefix-extension of the previous
        step's (prompt+response) are merged into one row:

            row.prompt   = step[0].prompt_ids
            row.response = step[0].resp + delta_obs_1 + step[1].resp + ...
            row.mask     = [1*len(step[0].resp), 0*len(delta_obs_1), 1*len(step[1].resp), ...]
            row.logprobs = [lp_0, 0.0 * len(delta_obs_1), lp_1, ...]

        When a step's prompt is NOT a prefix-extension (e.g., context
        compression), we close the current segment and start a new one.
        """
        if not steps:
            return []

        def _new_segment(step):
            prompt = list(step["prompt_ids"])
            action = list(step["response_ids"])
            action_lp = list(step.get("response_logprobs") or [])
            if len(action_lp) < len(action):
                action_lp = action_lp + [0.0] * (len(action) - len(action_lp))
            elif len(action_lp) > len(action):
                action_lp = action_lp[: len(action)]
            return {
                "prompt_ids": prompt,
                "response_ids": list(action),
                "response_mask": [1] * len(action),
                "response_logprobs": list(action_lp),
                "full_seq": list(prompt) + list(action),
            }

        segments: list[dict] = []
        seg = _new_segment(steps[0])

        for step in steps[1:]:
            prompt_ids = list(step["prompt_ids"])
            full_seq = seg["full_seq"]
            is_prefix_extension = len(prompt_ids) >= len(full_seq) and prompt_ids[: len(full_seq)] == full_seq
            if is_prefix_extension:
                delta_obs = prompt_ids[len(full_seq) :]
                action = list(step["response_ids"])
                action_lp = list(step.get("response_logprobs") or [])
                if len(action_lp) < len(action):
                    action_lp = action_lp + [0.0] * (len(action) - len(action_lp))
                elif len(action_lp) > len(action):
                    action_lp = action_lp[: len(action)]

                seg["response_ids"].extend(delta_obs)
                seg["response_ids"].extend(action)
                seg["response_mask"].extend([0] * len(delta_obs))
                seg["response_mask"].extend([1] * len(action))
                seg["response_logprobs"].extend([0.0] * len(delta_obs))
                seg["response_logprobs"].extend(action_lp)
                seg["full_seq"].extend(delta_obs)
                seg["full_seq"].extend(action)
            else:
                segments.append(seg)
                seg = _new_segment(step)

        segments.append(seg)
        for s in segments:
            s.pop("full_seq", None)
        return segments

    def _post_process_rollout_data(self, rollout_data_batch: list[dict]) -> DataProto:
        """Post-process raw rollout data to segment-level DataProto format.

        One row = one merged segment. All model responses are used as training
        data (no turn filtering).
        """
        max_prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        max_response_length = self.config.actor_rollout_ref.rollout.response_length

        prompts: list[torch.Tensor] = []
        responses: list[torch.Tensor] = []
        response_masks: list[torch.Tensor] = []
        prompt_lengths: list[int] = []
        response_lengths: list[int] = []
        rollout_logprobs: list[torch.Tensor] = []
        reward_scalars: list[float] = []
        uids: list[str] = []
        trajectory_ids: list[str] = []
        is_last_steps: list[bool] = []

        for rollout_data in rollout_data_batch:
            if not rollout_data or not rollout_data.get("rollout"):
                continue
            steps = rollout_data["rollout"]
            outcome_reward = rollout_data["reward"]
            uid = rollout_data["uid"]
            trajectory_id = rollout_data["session_id"]

            segments = self._merge_steps_to_segments(steps)
            if not segments:
                continue

            # Drop segments whose prompt exceeds max_prompt_length
            valid_segments = []
            for seg_idx, seg in enumerate(segments):
                if len(seg["prompt_ids"]) > max_prompt_length:
                    logger.warning(
                        f"Dropping segment {seg_idx} of trajectory {trajectory_id}: "
                        f"prompt length {len(seg['prompt_ids'])} > max_prompt_length {max_prompt_length}"
                    )
                    continue
                valid_segments.append(seg)

            if not valid_segments:
                continue

            num_segments = len(valid_segments)
            for seg_idx, seg in enumerate(valid_segments):
                raw_prompt_ids = seg["prompt_ids"]
                raw_response_ids = seg["response_ids"]
                raw_mask = seg["response_mask"]
                raw_logprobs = seg["response_logprobs"]

                is_last = seg_idx == num_segments - 1

                prompts.append(torch.tensor(raw_prompt_ids, dtype=torch.long))
                responses.append(torch.tensor(raw_response_ids, dtype=torch.long))
                response_masks.append(torch.tensor(raw_mask, dtype=torch.long))
                prompt_lengths.append(len(raw_prompt_ids))
                response_lengths.append(len(raw_response_ids))
                rollout_logprobs.append(torch.tensor(raw_logprobs, dtype=torch.float32))

                # Outcome reward only on last segment of trajectory
                reward_scalars.append(outcome_reward if is_last else 0.0)
                uids.append(uid)
                trajectory_ids.append(trajectory_id)
                is_last_steps.append(is_last)

        if len(prompts) == 0:
            raise RuntimeError(
                "No valid segments after filtering overlong prompts. "
                f"All segments had initial prompt length > max_prompt_length ({max_prompt_length}). "
                "Consider increasing actor_rollout_ref.rollout.prompt_length."
            )

        observed_max_response = max(response_lengths) if response_lengths else 0
        response_dim = max(observed_max_response, max_response_length)

        # Prompt: left-pad, then ensure min length = max_prompt_length
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(p, dims=[0]) for p in prompts],
            batch_first=True,
            padding_value=self.pad_token_id,
        ).flip(dims=[1])
        prompts_batch = _pad_sequence_to_length(prompts_batch, max_prompt_length, self.pad_token_id, left_pad=True)
        prompts_batch = prompts_batch[:, -max_prompt_length:]

        # Response: right-pad to the dynamic dim
        responses_batch = torch.nn.utils.rnn.pad_sequence(
            responses,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        responses_batch = _pad_sequence_to_length(responses_batch, response_dim, self.pad_token_id, left_pad=False)

        # Action-vs-obs mask: right-pad with 0
        response_mask_batch = torch.nn.utils.rnn.pad_sequence(
            response_masks,
            batch_first=True,
            padding_value=0,
        )
        response_mask_batch = _pad_sequence_to_length(response_mask_batch, response_dim, 0, left_pad=False)

        # Rollout log probs: right-pad with -100.0 (sentinel for padded positions)
        logprobs_batch = torch.nn.utils.rnn.pad_sequence(
            rollout_logprobs,
            batch_first=True,
            padding_value=-100.0,
        )
        logprobs_batch = _pad_sequence_to_length(logprobs_batch, response_dim, -100.0, left_pad=False)

        # Build attention mask
        N = len(prompts)
        prompt_len_t = torch.as_tensor(prompt_lengths).clamp_(max=max_prompt_length)
        response_len_t = torch.as_tensor(response_lengths).clamp_(max=response_dim)

        prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
        prompt_attention_mask = prompt_pos >= (max_prompt_length - prompt_len_t.unsqueeze(1))

        resp_pos = torch.arange(response_dim).unsqueeze(0)
        response_attention_mask = resp_pos < response_len_t.unsqueeze(1)

        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1).long()
        input_ids = torch.cat([prompts_batch, responses_batch], dim=1)
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Reward tensor: place at last action token of the segment
        reward_tensor = torch.zeros(N, response_dim, dtype=torch.float32)
        for i in range(N):
            if reward_scalars[i] == 0.0:
                continue
            resp_len = min(response_lengths[i], response_dim)
            if resp_len <= 0:
                continue
            mask_row = response_mask_batch[i, :resp_len]
            action_positions = torch.nonzero(mask_row, as_tuple=False).flatten()
            if action_positions.numel() > 0:
                last_action_idx = int(action_positions[-1].item())
                reward_tensor[i, last_action_idx] = reward_scalars[i]
            else:
                reward_tensor[i, resp_len - 1] = reward_scalars[i]

        tensor_batch = TensorDict(
            {
                "prompts": prompts_batch,
                "responses": responses_batch,
                "response_mask": response_mask_batch.long(),
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "reward_tensor": reward_tensor,
                "rollout_log_probs": logprobs_batch,
            },
            batch_size=N,
        )

        batch = DataProto(
            batch=tensor_batch,
            non_tensor_batch={
                "uid": np.array(uids, dtype=object),
                "trajectory_id": np.array(trajectory_ids, dtype=object),
                "is_last_step": np.array(is_last_steps, dtype=bool),
            },
            meta_info={"timing": {}},
        )

        return batch

    # -- Public interface (matches AgentLoopManager) ---------------------------

    def generate_sequences(self, rollout_batch_input: DataProto) -> DataProto:
        """The rollout function called by the training loop.

        Args:
            rollout_batch_input: Input batch with payloads and UIDs.

        Returns:
            DataProto: Segment-level output batch.
        """
        config = self.config.actor_rollout_ref.rollout

        # OpenAI-compatible sampling parameters
        rollout_sampling_params = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.response_length,
        }
        extra_body = {
            "top_k": config.top_k,
            "repetition_penalty": getattr(config, "repetition_penalty", 1.0),
        }

        stop = getattr(config, "stop", None)
        if stop:
            rollout_sampling_params["stop"] = list(stop)
            extra_body["include_stop_str_in_output"] = bool(getattr(config, "include_stop_str_in_output", False))

        if rollout_batch_input.meta_info.get("validate", False):
            rollout_sampling_params["temperature"] = config.val_kwargs.temperature
            rollout_sampling_params["top_p"] = config.val_kwargs.top_p
            extra_body["top_k"] = config.val_kwargs.top_k

        if not rollout_batch_input.meta_info.get("do_sample", True):
            rollout_sampling_params["temperature"] = 0.0
            rollout_sampling_params["top_p"] = 1.0
            extra_body["top_k"] = -1

        if extra_body:
            rollout_sampling_params["extra_body"] = extra_body

        rollout_data_batch = asyncio.run(self._dispatch_and_collect(rollout_batch_input, rollout_sampling_params))

        return self._post_process_rollout_data(rollout_data_batch)
