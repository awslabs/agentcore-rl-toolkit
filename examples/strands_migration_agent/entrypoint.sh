#!/bin/bash
# Container entrypoint: obtain a CodeArtifact auth token, then exec the agent.

set -e

DOMAIN="${CODEARTIFACT_DOMAIN:-migration-aws-maven-mirror}"
DOMAIN_OWNER="${CODEARTIFACT_DOMAIN_OWNER:-521102048267}"
REGION="${AWS_REGION:-us-west-2}"

echo "$(date -Iseconds) [entrypoint] fetching CodeArtifact auth token (domain=${DOMAIN}, region=${REGION})" >&2

# Fetch token. If this fails, Maven will get 401/403 from CodeArtifact and
# all builds in the session will fail. A hard exit here surfaces the problem
# early rather than causing confusing Maven errors later.
if ! CODEARTIFACT_AUTH_TOKEN=$(aws codeartifact get-authorization-token \
        --domain "${DOMAIN}" \
        --domain-owner "${DOMAIN_OWNER}" \
        --region "${REGION}" \
        --query authorizationToken \
        --output text); then
    echo "$(date -Iseconds) [entrypoint] ERROR: failed to obtain CodeArtifact token" >&2
    echo "$(date -Iseconds) [entrypoint] Check that the runtime IAM role has codeartifact:GetAuthorizationToken permission" >&2
    echo "$(date -Iseconds) [entrypoint] and that CodeArtifact domain '${DOMAIN}' exists in account ${DOMAIN_OWNER}" >&2
    exit 1
fi
export CODEARTIFACT_AUTH_TOKEN

echo "$(date -Iseconds) [entrypoint] token obtained, launching agent" >&2
exec opentelemetry-instrument python -m rl_app
