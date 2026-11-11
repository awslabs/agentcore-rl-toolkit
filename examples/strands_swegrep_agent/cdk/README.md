# CDK: deploy the agent to AgentCore

Deploys `../agent` to an AWS Bedrock AgentCore Runtime in VPC mode, so the rollout container can reach
the rollout gateway on the GPU host's private IP, and creates the S3 bucket of repository snapshots.

## What the stack creates

- **AgentCore Runtime** from the ARM64 image built out of `../agent/Dockerfile`. On an x86 host the
  build needs QEMU, registered once per boot: `docker run --privileged --rm tonistiigi/binfmt --install arm64`.
- **VPC networking**: runtime ENIs in the GPU host's VPC, security-group rules opening the gateway port
  (9090) and the benchmark vLLM port (8399) on the host, and VPC endpoints (ecr.api, ecr.dkr, logs; S3
  gateway), since the runtime has no internet access. A VPC allows one S3 gateway route per route table
  and one private-DNS interface endpoint per service: if the VPC already has them, set
  `createVpcEndpoints: false` and `endpointSecurityGroupId`.
- **Corpus bucket** (retained): repo snapshots, read-only for the runtime role. Fill it after deploying
  with `../scripts/prepare_corpus.py`.

## Deploy

Prerequisites: Docker, AWS credentials, `cdk bootstrap` run once for the account and region, and
[uv](https://docs.astral.sh/uv/).

```bash
cd cdk
cp config.example.yaml config.yaml   # set account, region, vpcId, gatewaySecurityGroupId
bash deploy.sh                       # writes AGENTCORE_AGENT_ARN, SWEGREP_CORPUS_BUCKET and AWS_REGION to ../.env
```

Every key in `config.yaml` can be overridden with `-c key=value` (`bash deploy.sh -c corpusBucket=...`).
`recipe` (default `recipes/final.json`) sets the reward's beta on the runtime from that recipe; deploy with
`-c recipe=recipes/ablations/beta1.json` to train an ablation.
If a deploy fails, the stack ends in `ROLLBACK_COMPLETE` and must be deleted
(`aws cloudformation delete-stack --stack-name SweGrepAgentCoreStack`) before running `deploy.sh` again.
AgentCore keeps the runtime ENIs for about 8 hours after deletion, so `retainRuntimeSg: true` keeps the
runtime security group instead of blocking `cdk destroy` on it.
