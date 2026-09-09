#!/usr/bin/env bash

# Unpack a SWE-bench/Gym task image, named in the task JSON at $1, into this container.

set -eux

task_file=$1
namespace=$(jq -r '.docker_image_namespace' "$task_file")
uri=$(jq -r '.docker_image_uri' "$task_file")
image="$namespace/$uri"

# ECR auth: the username is always the literal "AWS", and the region is parsed out of
# the registry host <account>.dkr.ecr.<region>.amazonaws.com.
registry=${namespace%%/*}
region=$(echo "$registry" | sed -n 's/.*\.dkr\.ecr\.\([^.]*\)\.amazonaws\.com/\1/p')
username="AWS"
password=$(aws ecr get-login-password --region "$region")

# skopeo + `umoci unpack --rootless` rather than `crane export` (flattens layers and
# mishandles cross-layer hard links) or anything needing mount/chroot/new namespaces,
# which this container's seccomp profile and empty capability set reject.
workdir=$(mktemp -d)

# --insecure-policy avoids requiring /etc/containers/policy.json to be present.
skopeo copy --insecure-policy \
    --src-creds "$username:$password" \
    "docker://$image" \
    "oci:$workdir/oci:latest"

umoci unpack --rootless --image "$workdir/oci:latest" "$workdir/bundle"

# `cp -a` preserves the intra-tree hard links umoci produced.
cp -a "$workdir/bundle/rootfs/opt/miniconda3" /opt/miniconda3
cp -a "$workdir/bundle/rootfs/testbed" /testbed

rm -rf "$workdir"

# So that Bash tool calls get the testbed Python environment on PATH.
/opt/miniconda3/bin/conda init
echo "conda activate testbed" >>$HOME/.bashrc
