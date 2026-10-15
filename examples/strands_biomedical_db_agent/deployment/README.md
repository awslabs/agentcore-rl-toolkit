# Deploy on AgentCore Runtime

Run the commands below from the example directory. For AgentCore setup, see the
[Runtime documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html)
and [IAM permissions](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-permissions.html).
The Open Targets service has a separate [deployment recipe](opentargets/README.md).

## Publish the GWAS snapshot

```bash
uv run biomedical-gwas-data \
  --directory /path/to/persistent/gwas/2026-03-17 \
  --s3-prefix s3://YOUR_BUCKET/gwas/2026-03-17
```

The command verifies the pinned files and uploads missing objects. Existing
objects must match. Set `GWAS_CATALOG_SQL_DATA_ROOT` to this S3 prefix and grant
the Runtime role read access to its three objects.

## Build an image

| Target | Entry point | AgentCore protocol |
| --- | --- | --- |
| `mcp` | Both specialist tools at `/mcp`, port 8000 | MCP |
| `opentargets` | Open Targets training entrypoint, port 8080 | HTTP |
| `gwas-catalog` | GWAS Catalog training entrypoint, port 8080 | HTTP |
| `codex` | Codex main agent with four tools, port 8080 | HTTP |

```bash
export AWS_REGION=us-west-2
export IMAGE_URI=ACCOUNT.dkr.ecr.REGION.amazonaws.com/REPOSITORY:TAG
docker buildx build --platform linux/arm64 --target mcp \
  -f deployment/Dockerfile -t "$IMAGE_URI" --push .
```

The `mcp` target serves both specialists using model environment variables.
The `opentargets` and `gwas-catalog` targets are separate training Runtimes:
each gets model configuration from its invocation's `_config` and needs only
its own database configuration.

## Configure and deploy

Create an external JSON file, such as `$TMPDIR/biomedical-runtime.json`:

```json
{
  "AWS_REGION": "us-west-2",
  "OPEN_TARGETS_GRAPHQL_URL": "http://DATABASE_HOST:8081/api/v4/graphql",
  "OPEN_TARGETS_EXPECTED_RELEASE": "26.06",
  "GWAS_CATALOG_SQL_DATA_ROOT": "s3://YOUR_BUCKET/gwas/2026-03-17",
  "OPENTARGETS_MODEL_BASE_URL": "http://MODEL_HOST:8000/v1",
  "OPENTARGETS_MODEL_ID": "your-opentargets-model",
  "GWAS_MODEL_BASE_URL": "http://MODEL_HOST:8001/v1",
  "GWAS_MODEL_ID": "your-gwas-model"
}
```

API keys and sampling settings use the names in [`.env.example`](../.env.example).
All values in the deployment JSON must be strings.

```bash
uv run python deployment/deploy.py \
  --name biomedical_agents --image "$IMAGE_URI" \
  --role arn:aws:iam::ACCOUNT:role/YOUR_RUNTIME_ROLE \
  --env "$TMPDIR/biomedical-runtime.json" \
  --subnet subnet-YOUR_SUBNET --security-group sg-YOUR_SECURITY_GROUP \
  --protocol MCP --region "$AWS_REGION"
```

The script selects platform V2, waits for `READY`, and prints the ARN.
Use `--protocol HTTP` for the HTTP targets. To update a Runtime, replace `--name`
with `--runtime-id` and supply the complete configuration again.
Repeat the VPC flags for multiple subnets/security groups, or omit both for public
networking. For the `codex` target and remote specialist connection, see the
[Codex integration](../integrations/codex/README.md).
