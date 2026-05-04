# Dockerfile for the Migration agent, configured to use an AWS CodeArtifact
# mirror of Maven Central instead of hitting repo.maven.apache.org directly.
# This eliminates the HTTP 429 rate-limit failures seen under high-concurrency
# RL training.
#
# Prerequisites (run once outside this Dockerfile):
#   aws codeartifact create-domain --domain migration-aws-maven-mirror --region us-west-2
#   aws codeartifact create-repository --domain migration-aws-maven-mirror --repository maven-central-cache --region us-west-2
#   aws codeartifact associate-external-connection \
#       --domain migration-aws-maven-mirror --repository maven-central-cache \
#       --external-connection public:maven-central --region us-west-2
#   # Grant the AgentCoreRuntime IAM role codeartifact:GetAuthorizationToken
#   # and codeartifact:ReadFromRepository:
#   aws iam put-role-policy \
#       --role-name AgentCoreRuntime \
#       --policy-name CodeArtifactReadAccess \
#       --policy-document '{
#           "Version": "2012-10-17",
#           "Statement": [
#               {
#                   "Sid": "CodeArtifactRead",
#                   "Effect": "Allow",
#                   "Action": [
#                       "codeartifact:GetAuthorizationToken",
#                       "codeartifact:ReadFromRepository",
#                       "codeartifact:GetRepositoryEndpoint"
#                   ],
#                   "Resource": "*"
#               },
#               {
#                   "Sid": "STSServiceBearerToken",
#                   "Effect": "Allow",
#                   "Action": "sts:GetServiceBearerToken",
#                   "Resource": "*",
#                   "Condition": {
#                       "StringEquals": {
#                           "sts:AWSServiceName": "codeartifact.amazonaws.com"
#                       }
#                   }
#               }
#           ]
#       }'

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app

# All environment variables in one layer
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER_CONTAINER=1 \
    AWS_REGION=us-west-2 \
    AWS_DEFAULT_REGION=us-west-2 \
    CODEARTIFACT_DOMAIN=migration-aws-maven-mirror \
    CODEARTIFACT_DOMAIN_OWNER={your-aws-account-number}

# ---------- Java code migration specific requirements ----------
# 1. Install Java 17 as root user
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Verify Java installation
RUN echo "=== Java Installation Verification ===" && \
    java --version && \
    which java

# 2. Install maven
# install tools needed for Maven first
RUN apt-get update && \
    apt-get install -y curl unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://archive.apache.org/dist/maven/maven-3/3.9.6/binaries/apache-maven-3.9.6-bin.zip && \
    unzip apache-maven-3.9.6-bin.zip -d /opt/ && \
    rm apache-maven-3.9.6-bin.zip && \
    ln -s /opt/apache-maven-3.9.6 /opt/maven

# Set Maven environment variables
ENV MAVEN_HOME=/opt/maven
ENV PATH=$MAVEN_HOME/bin:$PATH

RUN echo "=== Maven Installation Verification ===" && mvn --version

# 3. Install Node.js (prevents some frontend plugins from downloading it
#    at runtime, which incurs latency & introduces spammy logs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 4. Install git
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    git config --system --add safe.directory '*' && \
    git config --system user.email "no-reply@amazon.com" && \
    git config --system user.name "NoReply Amazon"

# 5. Install AWS CLI v2 (needed at runtime to fetch CodeArtifact auth tokens).
#    The uv base image doesn't include it.
RUN ARCH=$(uname -m) && \
    case "$ARCH" in \
      x86_64)  AWS_URL="https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" ;; \
      aarch64) AWS_URL="https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" ;; \
      *) echo "Unsupported architecture: $ARCH"; exit 1 ;; \
    esac && \
    curl -sSL "$AWS_URL" -o /tmp/awscliv2.zip && \
    unzip -q /tmp/awscliv2.zip -d /tmp && \
    /tmp/aws/install && \
    rm -rf /tmp/awscliv2.zip /tmp/aws && \
    aws --version

# ----------

COPY . .
# Install local toolkit from build context, then install example deps
RUN --mount=type=bind,from=toolkit,source=.,target=/toolkit \
    uv pip install /toolkit && uv pip install .



RUN uv pip install aws-opentelemetry-distro==0.12.2

# 6. Install Maven settings.xml that mirrors all traffic through AWS CodeArtifact.
#    Placed at two locations:
#      - /opt/maven/conf/settings.xml (global, used by any user running mvn)
#      - /home/bedrock_agentcore/.m2/settings.xml (per-user, explicit)
#    Both point to the same file; Maven reads the user location if present,
#    otherwise the global.
COPY maven-settings.xml /opt/maven/conf/settings.xml

# 7. Install entrypoint script that fetches a CodeArtifact auth token at
#    container startup. No in-container refresh loop is needed because
#    AgentCore spawns a fresh container per session (<= 30min lifetime),
#    always within a single 12h token TTL. Future containers spawned hours
#    or days later will fetch their own fresh tokens at their own startup.
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 8. Create non-root user and set up their Maven config.
#    The symlink points the user-level settings at the global one so there's
#    only one file to maintain.
RUN useradd -m -u 1000 bedrock_agentcore && \
    mkdir -p /home/bedrock_agentcore/.m2 && \
    ln -sf /opt/maven/conf/settings.xml /home/bedrock_agentcore/.m2/settings.xml && \
    chown -R bedrock_agentcore:bedrock_agentcore /home/bedrock_agentcore

USER bedrock_agentcore

EXPOSE 9000
EXPOSE 8000
EXPOSE 8080


# Use the full module path
# Entrypoint obtains the CodeArtifact auth token, starts a background
# token-refresh loop, then execs the original agent command.
CMD ["/app/entrypoint.sh"]
