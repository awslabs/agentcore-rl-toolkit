# Example: SkyRL Tinker-compatible endpoint on EC2

This example deploys **Qwen3.5-4B on one `p4d.24xlarge`** (8 A100 40 GB GPUs) with SkyPilot.
A separate CPU client uses the endpoint for LoRA training and sampling.
The supplied configuration uses Megatron with a 16,384-token context and one
active training client per endpoint.

Use these files as a starting point for your own single-node deployment:

- `endpoint.yaml`: set the model ID and revision, SkyRL revision, and EC2 instance type.
- `backend_config.json`: set the training GPU count, inference replicas, parallelism,
  batch sizes, and context limits to match your model and hardware.

Adjust the two configs together and validate your chosen configuration.

## Prerequisites

- Linux, Python 3.12 via [uv](https://docs.astral.sh/uv/), Git, SSH, and rsync.
- AWS credentials with EC2 provisioning permissions and the IAM permissions below.
- Quota and available capacity for your instance type (P4d in this example).
- An existing private subnet with a unique `Name` tag in the region and outbound
  HTTPS access for packages and model weights.

The launcher needs SSH access (TCP 22) and endpoint access (TCP 18080); the training
client needs TCP 18080. Routing and firewalls must permit these connections.
`--client-cidr` allows a source IPv4 range on both ports. For one host in the same
VPC, use its private IP plus `/32` (find it in EC2 details or `ip -4 addr`).
For separate hosts, use a suitable private subnet CIDR or a preconfigured security group.

## Deploy

If IAM permissions are missing (for example with `PowerUserAccess`), first run
the command below with `--plan`. It makes no AWS changes and prints the path to
`deployer-iam-policy.json`. Have an administrator attach that JSON as a customer
managed policy, scoped to creating/reading `skypilot-v1`, binding its role and
instance profile, and passing the role to EC2. Then rerun without `--plan`.

From this directory, using your AWS profile:

```bash
export AWS_PROFILE=my-profile

uv run --frozen python deploy.py \
  --region us-west-2 \
  --cluster skyrl-tinker-endpoint \
  --subnet-id subnet-YOUR_PRIVATE_SUBNET \
  --client-cidr 10.0.1.10/32
```

The launcher prepares IAM and the security group, provisions EC2, installs SkyRL
and model weights, and waits for HTTP readiness. It prints the private endpoint
URL and SDK `base_model` path.

| Option | Purpose |
| --- | --- |
| `--region REGION` | Default `us-west-2`; must match the subnet. Use a new cluster name for another region. |
| `--reservation cr-...` | Use a matching capacity reservation in the subnet's AZ. |
| `--security-group-id sg-...` | Reuse a group with the required ingress; existing rules are checked without editing. |
| `--image-id ami-...` | Override the default DLAMI in the selected region. |
| `--state-dir PATH` | Override the local deployment-state directory. |

By default, the launcher resolves the regional AMI ID for AWS Deep Learning Base
OSS Nvidia Driver GPU AMI (Ubuntu 22.04), release **20260922**.

<details>
<summary>Custom AMI requirements</summary>

Provide Ubuntu 22.04 with user `ubuntu`, CUDA 13.0 at `/usr/local/cuda-13.0`,
an NVIDIA driver supporting CUDA 13 (580 or newer), and the instance-store mount
at `/opt/dlami/nvme`. Validate these assumptions when overriding the image.

</details>

Deployment state is saved under the temporary directory in `skyrl-endpoints/CLUSTER/`.
Keep it for repeat deployments, or use `--state-dir` to choose a persistent location.

## Connect

Use **`tinker==0.24.1`** in your client environment for this SkyRL version.
For the toolkit's training loop, follow the [training guide](../README.md)
using the `gateway,tinker_skyrl` extras.

```python
import tinker

service = tinker.ServiceClient(
    base_url="http://GPU_PRIVATE_IP:18080",
    api_key="tml-dummy",
)
trainer = service.create_lora_training_client(
    base_model="/home/ubuntu/skyrl/models/Qwen3.5-4B",
    rank=8,
)
```

`tml-dummy` is a placeholder: **the endpoint has no authentication or TLS; keep it
private**. HTTP readiness does not initialize GPU workers; the first training
client and sampler can take several minutes to initialize.

## Manage and resume

```bash
uv run --frozen sky queue skyrl-tinker-endpoint
uv run --frozen sky logs skyrl-tinker-endpoint
uv run --frozen sky stop skyrl-tinker-endpoint   # preserve EBS
# Deletes EC2 and root EBS. Back up checkpoints first.
uv run --frozen sky down skyrl-tinker-endpoint
```

| Remote path | Storage | Contents |
| --- | --- | --- |
| `/home/ubuntu/skyrl/` | Root EBS | Source, environment, model weights and reusable caches |
| `/home/ubuntu/skyrl-state/` | Root EBS | Metadata, checkpoints, adapters and logs |
| `/opt/dlami/nvme/skyrl/tmp/` | Instance store | Temporary files and Ray runtime scratch |

Repeat the deployment command to reuse a running endpoint or resume after `stop`.
Stop/start preserves EBS and clears instance store; continuing training requires
reconnecting and explicitly loading a saved checkpoint. Configuration changes
require stopping the cluster and selecting a new `--state-dir`.

Failures leave resources in place. Check queue/logs; if an active job blocks a
retry, inspect it and explicitly cancel it with `uv run --frozen sky cancel CLUSTER JOB_ID`.
The endpoint stays running while idle, so stop it when finished. EBS and any
capacity reservation retain their own charges.

`down` deletes the root EBS data; backups are your responsibility. IAM resources
and security groups remain. Existing ingress rules are preserved during
deployment; remove obsolete client access explicitly.
