#!/usr/bin/env bash

set -eux

# $1 is a path to a JSON file holding the task contents; pull out the fields we
# need with jq. The full image reference is the namespace (registry host plus a
# path prefix) joined with the repo/tag URI.

task_file=$1
namespace=$(jq -r '.docker_image_namespace' "$task_file")
uri=$(jq -r '.docker_image_uri' "$task_file")
image="$namespace/$uri"

# username is always the literal "AWS" and the password is a short-lived token
# from `aws ecr get-login-password`. The registry region is parsed from the
# namespace host: <account>.dkr.ecr.<region>.amazonaws.com/...
registry=${namespace%%/*}
region=$(echo "$registry" | sed -n 's/.*\.dkr\.ecr\.\([^.]*\)\.amazonaws\.com/\1/p')
username="AWS"
password=$(aws ecr get-login-password --region "$region")

# We do NOT use `crane export`. That command flattens all
# layers into a single tar via go-containerregistry's mutate.Extract, which
# mishandles cross-layer hard links.

# This must run inside a heavily restricted container: empty capability set,
# noNewPrivileges, and a seccomp profile that blocks mount/pivot_root/unshare/
# setns and namespace-creating clone(). That rules out anything that builds a
# rootfs via mount, chroot, or a new user namespace (podman/buildah rootless,
# apptainer). skopeo copy is a plain download; `umoci unpack --rootless` only
# writes files into a directory and skips the chown/mknod/setcap operations that
# the empty capability set would reject.
workdir=$(mktemp -d)

# --insecure-policy avoids requiring /etc/containers/policy.json to be present.
skopeo copy --insecure-policy \
    --src-creds "$username:$password" \
    "docker://$image" \
    "oci:$workdir/oci:latest"

umoci unpack --rootless --image "$workdir/oci:latest" "$workdir/bundle"

# Copy out only the two folders the agent needs; `cp -a` preserves the
# intra-tree hard links produced by umoci.
# This is very specific to this particular benchmark family (SWE Bench / Gym).
cp -a "$workdir/bundle/rootfs/opt/miniconda3" /opt/miniconda3
cp -a "$workdir/bundle/rootfs/testbed" /testbed

rm -rf "$workdir"

# We assume that the agent harness will use Bash to execute tool calls.
# This ensures that Bash will have the right Python environment on PATH.
/opt/miniconda3/bin/conda init
echo "conda activate testbed" >>$HOME/.bashrc
