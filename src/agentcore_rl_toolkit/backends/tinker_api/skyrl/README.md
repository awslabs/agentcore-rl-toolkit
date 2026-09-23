# SkyRL Tinker-compatible endpoint on EC2

Deploy a private, persistent training and sampling endpoint with one command.
SkyPilot manages the EC2 lifecycle and runs SkyRL's Tinker API server. A separate
CPU program connects through the Tinker SDK. This example has no dependency on
ART, AgentCore, or the toolkit's training loop.

The verified configuration is **Qwen3.5-4B on one `p4d.24xlarge`** in `us-west-2`
(8 A100 40 GB GPUs). It uses LoRA with FSDP training and four vLLM replicas, each
with tensor parallelism 2. Training and sampling alternate on the same GPUs.
Use one active training client per endpoint. Context length is 4,096 tokens.
This first example deliberately fixes the model and GPU layout in `endpoint.yaml`
and `serve.sh`; it does not implement a general resource scheduler.

## Prerequisites

On the machine running the launcher:

- Linux, Python 3.12 via [uv](https://docs.astral.sh/uv/), Git, SSH, and rsync.
- AWS credentials with SkyPilot's EC2 provisioning permissions and the scoped IAM
  permissions below. Set `AWS_PROFILE` if using a named profile.
- A network route to an existing private subnet in the selected AWS region. The subnet must
  have a unique `Name` tag in the region, because SkyPilot selects it by name.
- Enough P4d quota and capacity, or a matching accessible capacity reservation.
  The subnet needs outbound HTTPS access for public packages and model weights.

The launcher and training client do not need local GPUs. They can run on the
same machine or on different hosts with access to the endpoint. `--client-cidr`
is the **source IPv4 range that must reach TCP 22 and 18080**. For one client in
the same VPC, use that host's private IP followed by `/32`; find it under EC2
instance details, or check `ip -4 addr` on the client. If launcher and training
client differ, use a suitable private subnet CIDR or preconfigure a security
group with both clients' required access. VPC routing, NACLs and firewalls must
also permit the traffic; this example does not create peering or VPN connections.

## Deploy

From this directory:

```bash
export AWS_PROFILE=my-profile

./deploy.sh \
  --region us-west-2 \
  --cluster skyrl-tinker-endpoint \
  --subnet-id subnet-YOUR_PRIVATE_SUBNET \
  --client-cidr 10.0.1.10/32
```

`--region` defaults to `us-west-2`; the subnet and optional
`--reservation cr-YOUR_RESERVATION` must belong to that region, with the
reservation in the subnet's AZ.
Use a new cluster name when deploying in another region; an existing EC2
instance and its EBS volume are not moved between regions by this command.

The launcher finds the official AWS **Deep Learning Base OSS Nvidia Driver GPU
AMI (Ubuntu 22.04), release 20260922** in the selected region. AMI IDs are regional:
the same release has a different ID in each region. The release is fixed for
reproducibility; users do not need to look up or supply its regional ID.

Optionally pass `--image-id ami-YOUR_IMAGE` to override it. A compatible image must
provide Ubuntu 22.04 with user `ubuntu`, CUDA 13.0 at `/usr/local/cuda-13.0`, an
NVIDIA driver supporting CUDA 13 (580 or newer), and DLAMI's instance-store mount
at `/opt/dlami/nvme`. Choosing another image requires validating these assumptions.
The full endpoint has been exercised in `us-west-2`; matching AMI resolution has
also been checked in `us-east-1`.

The command:

1. Resolves the subnet and regional AMI, and writes an account-scoped IAM policy
   for the deployer.
2. Creates the EC2 role and instance profile `skypilot-v1` if missing, then binds
   them. The new role trusts EC2 and has **no AWS resource-access policies**.
3. Creates a dedicated security group allowing TCP 22 and 18080 from the supplied
   CIDR, traffic within the group, and default outbound access. To reuse a group,
   pass `--security-group-id sg-...`; required ingress is checked without editing it.
4. Writes a deployment-specific SkyPilot configuration, provisions EC2, installs
   pinned SkyRL dependencies, downloads pinned model weights, and starts the server.
5. Waits for the server job to run and its HTTP health check to pass, then prints
   the private endpoint URL and SDK `base_model` path. GPU workers initialize when
   a client first requests training or sampling.

Existing security-group rules are preserved. Revoke obsolete client access
explicitly if you change the allowed CIDR for an example-created group.

The launcher automatically selects local scratch space: existing `TMPDIR`, then
`TMP` or `TEMP`, otherwise `/tmp`. Set `TMPDIR` only if you want to choose a
different location for the client environment and caches, for example a larger
local disk. This does not affect storage paths on the GPU instance.

The result prints the private endpoint URL and SDK `base_model` path. Generated
configuration and `deployment.json` live in `skyrl-endpoints/CLUSTER/` under the
selected scratch directory; use `--state-dir PATH` to select another local
location. Keep this directory for repeat deployments. Your global SkyPilot
configuration and local AWS credential files are not modified or uploaded to EC2.

Re-running the same command reuses a running endpoint without creating another
server. Re-running after `sky stop` starts the instance, runs setup, and checks
HTTP readiness for a new server job. Active jobs with
missing or incomplete deployment state require explicit inspection/cancellation
before retrying. Configuration changes require stopping the cluster and selecting
a new `--state-dir`; the script never silently replaces an active service.

### One-time IAM permissions

A `PowerUserAccess` identity generally cannot create IAM roles or instance
profiles. Preview the deployment without AWS writes:

```bash
./deploy.sh --subnet-id subnet-YOUR_PRIVATE_SUBNET \
  --region us-west-2 --client-cidr 10.0.1.10/32 --plan
```

This writes `deployer-iam-policy.json`. Have an administrator create a **customer
managed policy** from that JSON and attach it to the deploying identity. In
addition to EC2 provisioning permissions, it grants only:

- `GetRole` and `CreateRole` for `role/skypilot-v1`.
- `GetInstanceProfile`, `CreateInstanceProfile` and `AddRoleToInstanceProfile`
  for `instance-profile/skypilot-v1`.
- `PassRole` for `role/skypilot-v1`, limited to EC2.

It does not grant `PutRolePolicy`, `AttachRolePolicy`, or `IAMFullAccess`. The
launcher does not grant itself permissions. An administrator can instead create
and populate the profile once, and grant the deployer only the required reads
and `PassRole`.

Prepopulating the profile is intentional: SkyPilot's default bootstrap attaches
broader EC2/S3 policies when it creates its own role. SkyPilot reuses our populated
profile without those writes. If `skypilot-v1` already exists, this example checks
the role name and EC2 trust; it does **not** replace its trust or inspect/change
its existing permission policies. IAM resources are shared account resources and
are not deleted when an endpoint is stopped or terminated.

## Storage and restart behavior

| Remote path | Storage | Contents |
| --- | --- | --- |
| `/home/ubuntu/skyrl/` | Root EBS | SkyRL source, Python environment, model weights, uv/HF and GPU compilation caches |
| `/home/ubuntu/skyrl-state/` | Root EBS | SQLite metadata, checkpoints, adapter files and SkyRL logs |
| `/opt/dlami/nvme/skyrl/tmp/` | Instance store | Temporary files and SkyRL Ray runtime scratch |

The root EBS volume is 200 GB. EC2 **stop/start preserves EBS**, while instance
store contents are lost. Setup recreates scratch directories and reuses the
persistent source, environment and model files. Initialization still loads GPU
workers; some compilation can recur. This is not transparent resumption of an
in-flight client or unsaved optimizer state: reconnect and explicitly load saved
checkpoints when continuing training.

`sky down` terminates EC2 and deletes its root EBS volume. Back up needed
checkpoints elsewhere first. This example does not create an additional data
volume or implement S3 backups. EBS storage remains billable while EC2 is stopped;
a separately purchased capacity reservation has its own billing lifecycle.

## Connect and manage

The API server is a long-running SkyPilot job, not the training loop. HTTP
`/api/v1/healthz` indicates liveness. GPU initialization happens when the first
training client and sampler are created and can take several minutes.

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

`tml-dummy` satisfies the SDK's API-key requirement; **this deployment does not
provide authentication or TLS**. Keep it private. This is a single-user service
example, not a managed multi-tenant platform or a claim of complete Tinker API parity.

The launcher also exposes the pinned SkyPilot CLI:

```bash
./deploy.sh sky queue skyrl-tinker-endpoint
./deploy.sh sky logs skyrl-tinker-endpoint
./deploy.sh sky stop skyrl-tinker-endpoint   # preserve EBS; resume with deployment command
# Destructive: deletes EC2 and root EBS. Back up checkpoints first.
./deploy.sh sky down skyrl-tinker-endpoint
```

The long-running server remains a running job even while no client is training;
stop it explicitly when finished. A deployment failure leaves resources in place
and prints inspection/stop commands. Inspect `sky logs` before retrying. To
restart the service on the same instance, explicitly cancel its server job with
`./deploy.sh sky cancel CLUSTER JOB_ID`, then repeat the deployment command.
Customer-provided network resources and IAM resources are never deleted; the
example-created security group also remains for reuse and can be deleted after
all attached instances are terminated.

SkyRL and model revisions are pinned in `endpoint.yaml`, and the default DLAMI
release name is in `aws_setup.py`. The standalone deployment client depends only
on SkyPilot and is locked in `uv.lock`.
