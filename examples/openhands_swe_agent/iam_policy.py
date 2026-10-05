"""The execution role this recipe's AgentCore sessions assume, written out in full.

Scoped by account and repository, but deliberately not by region, runtime name or image
tag: one role serves every runtime this recipe deploys, in every region. Each deploy
reconciles the whole role, so a grant pinned to the deploying region would cut off the
runtimes in every other region. Four non-obvious points:

* AgentCore pulls the *agent* image with this role, not with anything belonging to the
  pool, so the read grant on that repository cannot be dropped.
* Populating the task images' pull through cache on first pull needs
  ``ecr:BatchImportUpstreamImage`` and ``ecr:CreateRepository`` (both scoped to the cache
  prefix). Without them, pulls work until the first image nobody has cached yet.
* The trust policy also lets the account assume the role, which local docker runs use to
  hand a container the same permissions an AgentCore session gets.
* There is no Bedrock grant: the agent's LLM credential is a bearer token in the rollout
  payload, so model access never rides on this role.
"""

# Also the key deciding which of the role's inline policies is ours and which are strays.
POLICY_NAME = "Policy"

DESCRIPTION = "minimal permissions for AgentCore runtime with swe_agent"


def trust_policy(account_id: str) -> dict:
    """Who may assume the execution role: AgentCore, and this account itself.

    The service statement is conditioned on the source account so the role cannot be used
    as a confused deputy by another account's runtime.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AssumeRolePolicy",
                "Effect": "Allow",
                "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {"aws:SourceAccount": account_id},
                    "ArnLike": {"aws:SourceArn": f"arn:aws:bedrock-agentcore:*:{account_id}:*"},
                },
            },
            {
                "Sid": "AssumeFromThisAccount",
                "Effect": "Allow",
                "Principal": {"AWS": f"arn:aws:iam::{account_id}:root"},
                "Action": "sts:AssumeRole",
            },
        ],
    }


def permissions_policy(account_id: str, cache_prefix: str, agent_repository: str) -> dict:
    """What a rollout container may do, once it is running.

    ``cache_prefix`` (task images) and ``agent_repository`` (the agent image) are the only
    two repositories this role can read. Every grant is region wildcarded on purpose: a
    session may run in any region deployed into, and a pull through cache is regional.
    """
    log_group = f"arn:aws:logs:*:{account_id}:log-group:/aws/bedrock-agentcore/runtimes"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ECRImageAccess",
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                ],
                "Resource": [
                    f"arn:aws:ecr:*:{account_id}:repository/{agent_repository}",
                    f"arn:aws:ecr:*:{account_id}:repository/{cache_prefix}/*",
                ],
            },
            {
                # Lets a task image be pulled for the first time.
                "Sid": "ECRPullThroughCache",
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchImportUpstreamImage",
                    "ecr:CreateRepository",
                ],
                "Resource": [f"arn:aws:ecr:*:{account_id}:repository/{cache_prefix}/*"],
            },
            {
                "Sid": "ECRTokenAccess",
                "Effect": "Allow",
                "Action": ["ecr:GetAuthorizationToken"],
                "Resource": "*",
            },
            {
                "Sid": "CloudWatchLogGroups",
                "Effect": "Allow",
                "Action": [
                    "logs:DescribeLogStreams",
                    "logs:CreateLogGroup",
                ],
                "Resource": [f"{log_group}/*"],
            },
            {
                "Sid": "CloudWatchLogStreams",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": [f"{log_group}/*:log-stream:*"],
            },
            {
                "Sid": "CloudWatchLogResourcePolicy",
                "Effect": "Allow",
                "Action": ["logs:PutResourcePolicy"],
                "Resource": [f"{log_group}/*"],
            },
            {
                # Unscopeable: DescribeLogGroups does not accept a narrower resource.
                "Sid": "CloudWatchDescribeLogGroups",
                "Effect": "Allow",
                "Action": ["logs:DescribeLogGroups"],
                "Resource": [f"arn:aws:logs:*:{account_id}:log-group:*"],
            },
            {
                "Sid": "CloudWatchMetrics",
                "Effect": "Allow",
                "Action": "cloudwatch:PutMetricData",
                "Resource": "*",
                "Condition": {"StringEquals": {"cloudwatch:namespace": "bedrock-agentcore"}},
            },
            {
                "Sid": "XRayTracing",
                "Effect": "Allow",
                "Action": [
                    "xray:PutTraceSegments",
                    "xray:PutTelemetryRecords",
                    "xray:GetSamplingRules",
                    "xray:GetSamplingTargets",
                ],
                "Resource": ["*"],
            },
            {
                "Sid": "GetAgentAccessToken",
                "Effect": "Allow",
                "Action": [
                    "bedrock-agentcore:GetWorkloadAccessToken",
                    "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
                    "bedrock-agentcore:GetWorkloadAccessTokenForUserId",
                ],
                "Resource": [
                    f"arn:aws:bedrock-agentcore:*:{account_id}:workload-identity-directory/default",
                    f"arn:aws:bedrock-agentcore:*:{account_id}:workload-identity-directory/default/workload-identity/*",
                ],
            },
        ],
    }
