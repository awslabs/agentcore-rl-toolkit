FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app

# All environment variables in one layer
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER_CONTAINER=1 \
    AWS_REGION=us-west-2 \
    AWS_DEFAULT_REGION=us-west-2

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

# 3. Install git
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    git config --system --add safe.directory '*' && \
    git config --system user.email "no-reply@amazon.com" && \
    git config --system user.name "NoReply Amazon"

# ----------

COPY . .
# Install local toolkit from build context, then install example deps
RUN --mount=type=bind,from=toolkit,source=.,target=/toolkit \
    uv pip install /toolkit && uv pip install .



RUN uv pip install aws-opentelemetry-distro==0.12.2


# Create non-root user
RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

EXPOSE 9000
EXPOSE 8000
EXPOSE 8080


# Use the full module path

CMD ["opentelemetry-instrument", "python", "-m", "dev_app"]
