#!/usr/bin/env bash
# Deploy the agent to AgentCore Runtime and write its ARN, region and corpus bucket to ../.env.
#   bash cdk/deploy.sh [-c key=value ...]
# Needs Docker, AWS credentials and `cdk bootstrap` run once for the account and region.
set -euo pipefail
cd "$(dirname "$0")"
STACK=SweGrepAgentCoreStack
[ -f config.yaml ] || { echo "FATAL: copy config.example.yaml to config.yaml and fill it in" >&2; exit 1; }
CDK=$(command -v cdk || echo "npx --yes aws-cdk@2.1147.0")
$CDK deploy "$STACK" --require-approval never --outputs-file cdk-outputs.json "$@"

ARN=$(jq -er ".$STACK.AgentRuntimeArn" cdk-outputs.json)
CORPUS=$(jq -er ".$STACK.CorpusBucketName" cdk-outputs.json)
[ -f ../.env ] || cp ../.env.example ../.env
set_env() { { grep -v "^$1=" ../.env || true; echo "$1=$2"; } >../.env.tmp && mv ../.env.tmp ../.env; }
set_env AGENTCORE_AGENT_ARN "$ARN"
set_env SWEGREP_CORPUS_BUCKET "$CORPUS"
set_env AWS_REGION "$(cut -d: -f4 <<<"$ARN")"
echo "Wrote AGENTCORE_AGENT_ARN, SWEGREP_CORPUS_BUCKET and AWS_REGION to .env"
