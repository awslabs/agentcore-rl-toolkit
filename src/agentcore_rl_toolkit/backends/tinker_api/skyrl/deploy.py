"""Deploy the pinned Qwen3.5-4B SkyRL endpoint on one private EC2 P4d."""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import boto3
import httpx
import yaml
from aws_setup import (
    bootstrap_policy,
    client_network,
    ensure_profile,
    ensure_security_group,
    resolve_subnet,
    verify_instance_network,
)

HERE = Path(__file__).resolve().parent
REGION = "us-west-2"


def write_json(path, value):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def wait_healthy(sky, cluster, job_id, endpoint, timeout):
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=5, trust_env=False) as client:
        while time.monotonic() < deadline:
            statuses = sky.get(sky.job_status(cluster, job_ids=[job_id]))
            status = statuses.get(job_id)
            if status is not None and status.is_terminal():
                raise RuntimeError(f"Endpoint job {job_id} ended: {status}. Run sky logs.")
            if status is not None and status.value == "RUNNING":
                try:
                    response = client.get(endpoint + "/api/v1/healthz")
                    if response.status_code == 200 and response.json().get("status") == "ok":
                        return
                except (httpx.HTTPError, ValueError):
                    pass
            print(f"Waiting for endpoint (job {job_id}: {status})...", flush=True)
            time.sleep(15)
    raise TimeoutError(f"Endpoint not ready after {timeout}s. Run sky logs {cluster} {job_id}")


def run(args, state_dir):
    session = boto3.Session(region_name=REGION)
    account = session.client("sts").get_caller_identity()["Account"]
    policy_path = state_dir / "deployer-iam-policy.json"
    write_json(policy_path, bootstrap_policy(account))
    ec2 = session.client("ec2")
    subnet, subnet_name = resolve_subnet(ec2, args.subnet_id)
    task = yaml.safe_load((HERE / "endpoint.yaml").read_text())
    task["resources"]["infra"] = f"aws/{REGION}/{subnet['AvailabilityZone']}"
    task["workdir"] = str(HERE)
    base_model = "/home/ubuntu/skyrl/models/" + task["envs"]["MODEL_ID"].rsplit("/", 1)[-1]
    desired = {
        "account": account,
        "subnet_id": args.subnet_id,
        "client_cidr": args.client_cidr,
        "security_group_id": args.security_group_id,
        "reservation": args.reservation,
        "task": task,
        "scripts": {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in ("environment.sh", "setup.sh", "serve.sh")
        },
    }
    print(json.dumps(desired, indent=2), flush=True)
    print(f"Scoped IAM policy for the deployer: {policy_path}", flush=True)
    if args.plan:
        return
    manifest_path = state_dir / "deployment.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if old and old["desired"] != desired:
        raise ValueError(
            "Deployment settings changed. Stop the cluster and use a new --state-dir "
            "to apply new settings; existing jobs are never replaced automatically."
        )
    ensure_profile(session.client("iam"))
    group_id, group_name = ensure_security_group(
        ec2, subnet["VpcId"], args.cluster, args.client_cidr, args.security_group_id
    )
    config = {
        "aws": {
            "subnet_names": [subnet_name],
            "security_group_name": group_name,
            "use_internal_ips": True,
            "use_ssm": False,
        }
    }
    if args.reservation:
        config["aws"]["specific_reservations"] = [args.reservation]
    config_path = state_dir / "sky-config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    os.environ["SKYPILOT_CONFIG"] = str(config_path)
    # Import only after selecting this deployment's config; no ~/.sky edits.
    import sky

    records = sky.get(sky.status(cluster_names=[args.cluster]))
    if records:
        verify_instance_network(ec2, records[0]["handle"].cluster_name_on_cloud, args.subnet_id, group_id)
    if records and records[0]["status"].value == "UP":
        active = sky.get(sky.queue(args.cluster, skip_finished=True, all_users=True))
        if active:
            if (
                len(active) != 1
                or active[0]["job_id"] != old.get("job_id")
                or active[0]["job_name"] != task["name"]
                or old.get("status") != "ready"
            ):
                raise RuntimeError(
                    "This cluster has an active job without matching ready state. Inspect sky queue/logs "
                    "and explicitly cancel it before retrying; no second server was started."
                )
            wait_healthy(sky, args.cluster, old["job_id"], old["endpoint"], 60)
            print(f"Reusing endpoint: {old['endpoint']}\nbase_model: {base_model}")
            return
    elif records and records[0]["status"].value != "STOPPED":
        raise RuntimeError("Cluster provisioning is already in progress; inspect sky status/logs")
    manifest = {
        "desired": desired,
        "cluster": args.cluster,
        "region": REGION,
        "security_group_id": group_id,
        "instance_profile": "skypilot-v1",
        "base_model": base_model,
        "status": "launching",
    }
    write_json(manifest_path, manifest)
    # SkyPilot consumes (pops fields from) its input dictionary. Keep the
    # manifest's desired configuration intact for repeat deployment checks.
    request = sky.launch(sky.Task.from_yaml_config(copy.deepcopy(task)), cluster_name=args.cluster)
    manifest["request_id"] = request
    write_json(manifest_path, manifest)
    print(f"SkyPilot request: {request}", flush=True)
    job_id, handle = sky.stream_and_get(request)
    manifest["instance_id"] = verify_instance_network(ec2, handle.cluster_name_on_cloud, args.subnet_id, group_id)
    endpoint = f"http://{handle.head_ip}:18080"
    manifest.update(job_id=job_id, endpoint=endpoint, status="waiting")
    write_json(manifest_path, manifest)
    wait_healthy(sky, args.cluster, job_id, endpoint, args.timeout)
    manifest["status"] = "ready"
    write_json(manifest_path, manifest)
    print(
        f"Endpoint HTTP ready: {endpoint}\nbase_model: {base_model}\nDetails: {manifest_path}\n"
        "The first training client and sampler initialize GPU workers lazily."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", default="skyrl-tinker-endpoint")
    parser.add_argument("--subnet-id", required=True, help="Existing private subnet in us-west-2")
    parser.add_argument("--client-cidr", required=True, type=client_network)
    parser.add_argument("--reservation", help="Optional matching targeted capacity reservation")
    parser.add_argument("--security-group-id", help="Reuse a preconfigured SG without changing it")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds to wait for HTTP readiness")
    parser.add_argument("--plan", action="store_true", help="Resolve subnet and print plan; no AWS writes")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,39}", args.cluster):
        parser.error("--cluster must be a lowercase SkyPilot name, at most 40 characters")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    state_dir = (args.state_dir or Path(os.environ["TMPDIR"]) / "skyrl-endpoints" / args.cluster).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "deploy.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args, state_dir)
        except Exception:
            print(
                f"Deployment did not complete. State/logs: {state_dir}\n"
                f"Inspect: ./deploy.sh sky queue {args.cluster}\n"
                f"Logs: ./deploy.sh sky logs {args.cluster}\n"
                f"Stop billing for compute: ./deploy.sh sky stop {args.cluster}\n"
                "No resources were automatically deleted.",
                file=sys.stderr,
            )
            raise


if __name__ == "__main__":
    main()
