# External sampling hardware

SkyRL trains on GPU while sampling runs on a separate server on other hardware.
This page covers the parts that do not depend on that hardware: the design, versions,
the GPU-side launch, network and permissions, and the contract a sampling server must
meet. Implementations:

| Server | Hardware | Guide |
|---|---|---|
| [`vllm_neuron/serve.sh`](vllm_neuron/serve.sh) (stock `vllm serve` with [`server.py`](vllm_neuron/server.py) plugins) | AWS Trainium (example profile: Qwen3.6-27B on `trn2.48xlarge`) | [vLLM-Neuron sampling server](vllm_neuron/README.md) |

ART's training loop is unchanged: run `agentcore_rl_toolkit.backends.tinker_api.train`
with `endpoint` set to the SkyRL API, as for any Tinker endpoint. SkyRL forwards each
`sample()` to the external server. The ART config has no serving settings.

## Design and ownership

| Component | Responsibility | Host |
|---|---|---|
| SkyRL `--runtime-role trainer` | Training only; `save_weights_for_sampler` writes the PEFT export to `<checkpoints-base>/<model_id>/sampler_weights/<ckpt>.tar.gz` | GPU |
| SkyRL `--external-inference-url` | The API sends each sample to `/v1/completions` with `model=<model_id>_<ckpt>` | GPU |
| Sampling server | Serves the version a request names, loading new versions from the checkpoints base (see the contract below) | Sampling host |

## SkyRL version and patches

External sampling uses SkyRL `a6c07705` rather than the endpoint's default pin, with two
patches from [`../patches/`](../patches/):

| Patch | Change |
|---|---|
| [`external-sampling.patch`](../patches/external-sampling.patch) | With `--runtime-role trainer`, `save_weights_for_sampler` exports the PEFT adapter as SkyRL's sampler archive; training and sampler archives are written and read through cloudpathlib, so `s3://` checkpoint bases work (the local cache copy is dropped after each upload, so the external client in the same process reads the object rather than racing S3's whole-second timestamps); LoRA `save_state` keeps only trainable tensors and optimizer state |
| [`external-inference-extract-race.patch`](../patches/external-inference-extract-race.patch) | Concurrent first samples for a new checkpoint no longer fail when another request finished extracting it (`OSError` ENOTEMPTY). Independent of external sampling hardware; also present on SkyRL main |

```bash
git clone https://github.com/NovaSky-AI/SkyRL.git "$SKYRL_DIR"
git -C "$SKYRL_DIR" checkout --detach a6c0770527e08626d25c84a99d3d3e69965936f0
for patch in "$ART_REPO"/src/agentcore_rl_toolkit/backends/tinker_api/skyrl/patches/*.patch; do
  git -C "$SKYRL_DIR" apply "$patch"
done
```

That revision accepts `tinker>=0.25.0,<0.28.0` clients, so install the `tinker_skyrl`
extra and then a tinker version in that range (tested: 0.27.2). No ART code runs in the
SkyRL process.

## Launch the trainer

On the GPU host, use the patched SkyRL checkout (`SKYRL_DIR`) and your trainer-only FSDP
configuration:

```bash
export ART_REPO=/path/to/agentcore-rl-toolkit
export SKYRL_BACKEND_CONFIG=/path/to/trainer-only-fsdp.json
export SKYRL_CHECKPOINTS_BASE=s3://BUCKET/skyrl-checkpoints
export SKYRL_EXTERNAL_INFERENCE_URL=http://SAMPLER_PRIVATE_IP:8000
export SKYRL_EXTERNAL_INFERENCE_API_KEY=SHARED_KEY   # the sampler's VLLM_API_KEY
bash "$ART_REPO/src/agentcore_rl_toolkit/backends/tinker_api/skyrl/serve.sh"
```

Start the sampling server as described in its guide.

## Network and permissions

| From | To | Port (default) | How |
|---|---|---|---|
| AgentCore Runtime | ART client's rollout gateway | `gateway_port` in the ART config | Runtime deployed in the client's VPC; security group rule |
| ART client | SkyRL API (GPU host) | `TINKER_PORT` (18080) | Security group rule |
| SkyRL API (GPU host) | Sampling server | 8000 | Security group rule, or a port forward (below) |
| GPU host, sampling host | S3 checkpoints base | HTTPS | IAM below |

**Private network (default).** When the GPU host can reach the sampling host privately,
for example in the same VPC, bind the sampler to its private address with an API key
(`VLLM_NEURON_BIND_HOST=PRIVATE_IP`, `VLLM_API_KEY=SHARED_KEY`) and give the trainer the
same key as `SKYRL_EXTERNAL_INFERENCE_API_KEY`. Allow inbound TCP 8000 on the sampler only
from the GPU host's security group. Without an API key, the sampler accepts only a
loopback address.

**No private route (fallback).** Across regions or VPCs, keep the sampler on loopback and
forward a port from the GPU host (SSH `-L`, or an SSM port-forwarding session, which
needs `ssm:StartSession` on the sampler instance and the `AWS-StartPortForwardingSession`
document, and `ssm:TerminateSession`). SSM forwards can stall without closing (in one
e2e run, responses were lost for 7.5 minutes), so probe `/health` through the forward
and restart it when a probe times out.

**IAM.** Each identity needs only the following, scoped to one checkpoints prefix and one
runtime. Set `AWS_REGION` to the bucket's region on both hosts.

| Identity | Actions |
|---|---|
| GPU trainer | `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject` on the prefix; `s3:ListBucket` on the bucket, limited to the prefix (cloudpathlib lists before every new write) |
| Sampling host | `s3:GetObject` on the prefix (add `s3:ListBucket` to get 404 rather than 403 for missing checkpoints) |
| ART client | `bedrock-agentcore:InvokeAgentRuntime` and `bedrock-agentcore:StopRuntimeSession` on the runtime |
| Runtime execution role | What the agent reads, such as task data; see the agent's example |

**Simplest: one role for the three hosts.** For a dedicated training job, give the GPU,
sampling and client hosts one EC2 instance role with the union of their actions (replace
BUCKET, PREFIX, REGION, ACCOUNT and RUNTIME_ID). Keep the runtime execution role separate:
AgentCore assumes it, and the agent's sandbox should not hold training permissions.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
     "Resource": "arn:aws:s3:::BUCKET/PREFIX/*"},
    {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": "arn:aws:s3:::BUCKET",
     "Condition": {"StringLike": {"s3:prefix": "PREFIX/*"}}},
    {"Effect": "Allow",
     "Action": ["bedrock-agentcore:InvokeAgentRuntime", "bedrock-agentcore:StopRuntimeSession"],
     "Resource": ["arn:aws:bedrock-agentcore:REGION:ACCOUNT:runtime/RUNTIME_ID",
                  "arn:aws:bedrock-agentcore:REGION:ACCOUNT:runtime/RUNTIME_ID/*"]}
  ]
}
```

Everything stays scoped to this job's prefix and runtime, but every host can do what any
host can: the sampling host could overwrite or delete checkpoints, including the final
training checkpoint, and the GPU and sampling hosts could invoke the runtime. If the
role is shared with unrelated workloads, or you want each host limited to its own
actions, use per-host roles. For the SSM port-forward fallback, also allow
`ssm:StartSession` and `ssm:TerminateSession` (see above).

<details>
<summary>Stricter: per-host policies</summary>

GPU trainer:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
     "Resource": "arn:aws:s3:::BUCKET/PREFIX/*"},
    {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": "arn:aws:s3:::BUCKET",
     "Condition": {"StringLike": {"s3:prefix": "PREFIX/*"}}}
  ]
}
```

Sampling host:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": "s3:GetObject", "Resource": "arn:aws:s3:::BUCKET/PREFIX/*"}
  ]
}
```

ART client:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow",
     "Action": ["bedrock-agentcore:InvokeAgentRuntime", "bedrock-agentcore:StopRuntimeSession"],
     "Resource": ["arn:aws:bedrock-agentcore:REGION:ACCOUNT:runtime/RUNTIME_ID",
                  "arn:aws:bedrock-agentcore:REGION:ACCOUNT:runtime/RUNTIME_ID/*"]}
  ]
}
```

Attach each to that host's own role. If several hosts must share a role that you cannot
narrow, scope bucket policy statements with `aws:userid` (`ROLE_ID:INSTANCE_ID`) instead.

</details>

Both hosts use one S3 location, SkyRL's checkpoints base: `SKYRL_CHECKPOINTS_BASE` on the
GPU host and `ART_WEIGHT_REFRESH_CHECKPOINTS` on the sampler. SkyRL also stores training
checkpoints (`save_state`) under it.

## Sampling server contract

- **Name resolution:** `<model_id>_<ckpt>` is split at each `_`; exactly one
  `<model_id>/sampler_weights/<ckpt>.tar.gz` must exist under the checkpoints base,
  otherwise the request fails with 404. Each archive contains `adapter_config.json`
  (`peft_type: LORA`) and `adapter_model.safetensors`, and may be at most 4 GiB.
- **Immutable versions:** a sampling client created before an update cannot silently
  receive newer weights. The server holds one resident copy: it serves the newest
  version and rejects requests naming a version it replaced (HTTP 409).
- **Waiting, not failing:** requests for a version being activated wait for it,
  because SkyRL's client does not retry. SkyRL's external client times out a sample
  request after 300 s without a response, so loading a new version plus the first
  reply must fit in that window (about 140 s in the Qwen3.6-27B example on Trn2, more with a
  cold page cache).
