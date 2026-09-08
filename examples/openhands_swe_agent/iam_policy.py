"""The execution role this recipe's AgentCore sessions assume, written out in full.

This is the role the agent's own code runs as, and the only one of the deployment's
roles whose permissions are a statement about *this recipe* -- which is why it is the
only one here. The capacity provider's operator and instance roles are a property of
the AgentCore feature rather than of the agent, so they live with the rest of the
pool's provisioning instead.

Three things the role needs that are easy to get wrong:

**Which repositories it may read.** Two, and no others: the repository holding the
*agent* image, and everything under the task images' pull through cache prefix. The
first is there because AgentCore pulls the agent image with *this* role rather than
with anything belonging to the pool -- the capacity provider's instance role grants
only ``bedrock-agentcore:PutSystemLogEvents``, and an image pull shows up in
CloudTrail as ``BatchGetImage`` by this role's session, called by ``ecr.amazonaws.com``
on its behalf. So narrowing the read grant to the cache alone does not tighten a
rollout's reach; it stops the next cold start from finding the agent image.

**Pull through cache.** The container pulls its *task* image -- one SWE-bench or
SWE-Gym environment -- from an ECR pull through cache of Docker Hub, via
``skopeo`` in ``image/swe_unpack.sh``. Reading a cache repository that already
holds the image needs no more than the ordinary pull permissions, so a role with
only those works right up until it meets a task whose image nobody has pulled
before, and then fails with ``name unknown: The repository with name ... does not
exist in the registry``. Populating the cache on first pull needs
``ecr:BatchImportUpstreamImage``, plus ``ecr:CreateRepository`` because the cache
repository itself does not exist yet either. Both are scoped to the cache prefix,
so this is not a general grant to create repositories -- only to have ECR
materialise the mirror of an upstream image under that one namespace.

**Self-assumption.** The trust policy lets the account assume the role in addition to
``bedrock-agentcore.amazonaws.com``, which the AgentCore session does not need but the
docker session does: it calls ``sts:AssumeRole`` on this same role and passes the
credentials into the local container as environment variables, so that a rollout run
locally has the same permissions as one run on AgentCore.

The resource ARNs are parameterised by account, region and repository, but
deliberately *not* by runtime name or image tag: one role serves every runtime the
recipe deploys -- the training one and whatever test runtime ``config.toml``
currently points at -- so a grant narrowed to one name would silently stop the
others from working the next time anyone deployed. They stay scoped to this
account's AgentCore runtimes and to the two repositories above, both of which every
runtime of this recipe shares.
"""

# The inline policy's name on the role. Kept as a constant because it is also the
# key that decides which of the role's existing inline policies is *this* one and
# which are strays to be removed.
POLICY_NAME = "Policy"

DESCRIPTION = "minimal permissions for AgentCore runtime with swe_agent"


def trust_policy(account_id: str, region: str) -> dict:
    """Who may assume the execution role: AgentCore, and this account itself.

    The service statement is conditioned on the source account and on an ARN in
    this region's AgentCore namespace, so the role cannot be used as a confused
    deputy by another account's runtime. See the module docstring for why the
    second statement is here.
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
                    "ArnLike": {"aws:SourceArn": f"arn:aws:bedrock-agentcore:{region}:{account_id}:*"},
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


def permissions_policy(account_id: str, region: str, cache_prefix: str, agent_repository: str) -> dict:
    """What a rollout container may do, once it is running.

    ``cache_prefix`` is the ECR pull through cache namespace the *task* images come from
    and ``agent_repository`` is the repository the agent image itself is in -- the two,
    and only the two, this role can read images from. Both are passed in, since which
    they are is the recipe's config rather than this policy's business.

    The ECR grants are region wildcarded on purpose: a pull through cache is
    regional, so the same prefix names a different repository in every region this
    recipe is deployed into, and the agent image is pushed to one region and pulled
    from wherever a session runs. Narrowing that would mean deciding here which
    regions a recipe may be deployed into, which is a worse thing to be wrong about
    than a repository name that is already this account's.
    """
    log_group = f"arn:aws:logs:{region}:{account_id}:log-group:/aws/bedrock-agentcore/runtimes"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                # Read on exactly two repositories: the agent image AgentCore starts
                # a session from, and the task images under the cache. See the
                # module docstring on why the first cannot be dropped.
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
                # The grant that lets a task image be pulled for the first time.
                # Without it the pull fails only for images nobody has pulled
                # before, which reads as a broken dataset rather than a missing
                # permission -- see the module docstring.
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
                # Unscopeable: DescribeLogGroups filters across the account and
                # does not accept a narrower resource.
                "Sid": "CloudWatchDescribeLogGroups",
                "Effect": "Allow",
                "Action": ["logs:DescribeLogGroups"],
                "Resource": [f"arn:aws:logs:{region}:{account_id}:log-group:*"],
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
                    f"arn:aws:bedrock-agentcore:{region}:{account_id}:workload-identity-directory/default",
                    f"arn:aws:bedrock-agentcore:{region}:{account_id}:workload-identity-directory/default/workload-identity/*",
                ],
            },
            {
                "Sid": "BedrockModelInvocation",
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                "Resource": [
                    "arn:aws:bedrock:*::foundation-model/*",
                    f"arn:aws:bedrock:{region}:{account_id}:*",
                ],
            },
        ],
    }
