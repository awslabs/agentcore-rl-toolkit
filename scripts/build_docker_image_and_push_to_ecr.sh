#!/bin/bash
set -e

# Initialize variables
DOCKERFILE_PATH=""
DOCKER_TAG=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dockerfile=*)
      DOCKERFILE_PATH="${1#*=}"
      shift
      ;;
    --tag=*)
      DOCKER_TAG="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --dockerfile=PATH --tag=TAG"
      echo "  --dockerfile Path to Dockerfile (required)"
      echo "  --tag        Docker image tag to use (required)"
      exit 1
      ;;
  esac
done

# Check if required parameters are provided
if [ -z "$DOCKERFILE_PATH" ]; then
  echo "Error: --dockerfile parameter is required"
  echo "Usage: $0 --dockerfile=PATH --tag=TAG"
  exit 1
fi

if [ -z "$DOCKER_TAG" ]; then
  echo "Error: --tag parameter is required"
  echo "Usage: $0 --dockerfile=PATH --tag=TAG"
  exit 1
fi

echo "Building and pushing Docker image to ECR"
echo "=========================================="
echo "Dockerfile: ${DOCKERFILE_PATH}"
echo "Tag:        ${DOCKER_TAG}"
echo "=========================================="

# Load and validate environment
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    exit 1
fi

# ECR_REPO_NAME, AWS_REGION, and AWS_ACCOUNT should be specified in .env
source .env

required_vars=("ECR_REPO_NAME" "AWS_REGION" "AWS_ACCOUNT")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: $var is not set in .env file"
        exit 1
    fi
done

# Ensure buildx builder exists
echo ""
echo "Checking docker buildx..."
if ! docker buildx inspect >/dev/null 2>&1; then
    echo "Creating buildx builder for multi-platform builds..."
    docker buildx create --use
else
    echo "Buildx builder already configured"
fi

# Create ECR repository if it doesn't exist
echo ""
echo "Checking ECR repository..."
if ! aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} --region ${AWS_REGION} >/dev/null 2>&1; then
    echo "Creating ECR repository: ${ECR_REPO_NAME}"
    aws ecr create-repository --repository-name ${ECR_REPO_NAME} --region ${AWS_REGION}
else
    echo "ECR repository ${ECR_REPO_NAME} already exists"
fi

# Login to ECR
echo ""
echo "Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build and push to ECR
echo ""
echo "Building and pushing Docker image..."
echo "Image: ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${DOCKER_TAG}"
# Build to arm64 as required by AgentCore runtime, which uses AWS Graviton
docker buildx build --platform linux/arm64 -f ${DOCKERFILE_PATH} -t ${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${DOCKER_TAG} --push .

# Verify the image
echo ""
echo "Verifying pushed image..."
aws ecr describe-images --repository-name ${ECR_REPO_NAME} --region ${AWS_REGION}

echo ""
echo "Successfully built and pushed image: ${ECR_REPO_NAME}:${DOCKER_TAG}"
