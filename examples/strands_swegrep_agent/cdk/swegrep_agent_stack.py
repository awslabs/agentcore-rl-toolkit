"""The SWE-Grep agent on Amazon Bedrock AgentCore Runtime, in the GPU host's VPC.

The runtime calls the rollout gateway (training, :9090) and vLLM (benchmarks, :8399) on the host's private
IP, so it runs in that VPC. Settings come from config.yaml, each overridable with `-c key=value`:

  vpcId                    the GPU host's VPC (default: the default VPC)
  subnetType               public | private-egress | private-isolated (default: private-egress)
  gatewaySecurityGroupId   the host's security group; if set, :9090 and :8399 are opened to the runtime
  gatewayPort              rollout gateway port (default: 9090)
  createVpcEndpoints       create the ECR, logs and S3 endpoints (default: true); false to reuse existing ones
  endpointSecurityGroupId  with reused endpoints, their security group, opened to the runtime on :443
  runtimeSecurityGroupId   an existing security group for the runtime (default: create one)
  retainRuntimeSg          keep a created runtime security group on delete (default: true)
  corpusBucket             an existing bucket of repository snapshots (default: create one)
  runtimeName              default: swegrep_agent_rl
  recipe                   the recipe whose beta the runtime gets (default: recipes/final.json)
"""

import json
import os

from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_bedrockagentcore as agentcore
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_s3 as s3
from constructs import Construct

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
AGENT_DIR = os.path.join(REPO_ROOT, "agent")
EVAL_ENGINE_PORT = 8399  # the benchmark vLLM server (endpoint/serve_eval.sh)

_SUBNET_TYPES = {
    "public": ec2.SubnetType.PUBLIC,
    "private-egress": ec2.SubnetType.PRIVATE_WITH_EGRESS,
    "private-isolated": ec2.SubnetType.PRIVATE_ISOLATED,
}


def _as_bool(value, default: bool) -> bool:
    """config.yaml gives a bool, `-c key=value` a string."""
    if value is None:
        return default
    return value if isinstance(value, bool) else str(value).strip().lower() not in ("false", "0", "no")


def _recipe_beta(recipe: str) -> str:
    with open(os.path.join(REPO_ROOT, recipe)) as f:
        return str(json.load(f)["beta"])


class SweGrepAgentCoreStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ctx = self.node.try_get_context
        runtime_name = ctx("runtimeName") or "swegrep_agent_rl"
        existing_corpus = ctx("corpusBucket")
        vpc_id = ctx("vpcId")
        subnet_type = _SUBNET_TYPES[ctx("subnetType") or "private-egress"]
        gateway_sg_id = ctx("gatewaySecurityGroupId")
        gateway_port = int(ctx("gatewayPort") or 9090)
        # A VPC takes one S3 gateway endpoint per route table and one private-DNS endpoint per service.
        create_endpoints = _as_bool(ctx("createVpcEndpoints"), True)
        shared_endpoint_sg_id = ctx("endpointSecurityGroupId")
        # AgentCore holds on to the runtime's network interfaces for ~8 hours after a delete, so deleting
        # its security group would hang `cdk destroy`.
        runtime_sg_id = ctx("runtimeSecurityGroupId")
        retain_runtime_sg = _as_bool(ctx("retainRuntimeSg"), True)
        recipe = ctx("recipe") or "recipes/final.json"

        # Repository snapshots (scripts/prepare_corpus.py uploads them, agent/rollout/repo_store.py reads them).
        if existing_corpus:
            corpus_bucket = s3.Bucket.from_bucket_name(self, "CorpusBucket", existing_corpus)
        else:
            corpus_bucket = s3.Bucket(
                self,
                "CorpusBucket",
                removal_policy=RemovalPolicy.RETAIN,  # expensive to rebuild; keep
                enforce_ssl=True,
            )

        # AgentCore runs linux/arm64; building on x86 needs QEMU (see README.md).
        artifact = agentcore.AgentRuntimeArtifact.from_asset(AGENT_DIR, platform=ecr_assets.Platform.LINUX_ARM64)

        if vpc_id:
            vpc = ec2.Vpc.from_lookup(self, "GatewayVpc", vpc_id=vpc_id)
        else:
            vpc = ec2.Vpc.from_lookup(self, "GatewayVpc", is_default=True)

        if runtime_sg_id:
            runtime_sg = ec2.SecurityGroup.from_security_group_id(self, "RuntimeSg", runtime_sg_id, mutable=True)
        else:
            runtime_sg = ec2.SecurityGroup(
                self,
                "RuntimeSg",
                vpc=vpc,
                allow_all_outbound=True,
                description=f"AgentCore swegrep runtime ENIs (reach gateway :{gateway_port})",
            )
            if retain_runtime_sg:
                runtime_sg.apply_removal_policy(RemovalPolicy.RETAIN)

        subnets = ec2.SubnetSelection(subnet_type=subnet_type)
        network_configuration = agentcore.RuntimeNetworkConfiguration.using_vpc(
            self,
            vpc=vpc,
            vpc_subnets=subnets,
            security_groups=[runtime_sg],
        )

        # The runtime has only private IPs: without these endpoints (or a NAT) it can't pull its image,
        # write logs or read S3.
        if create_endpoints:
            endpoint_sg = ec2.SecurityGroup(
                self,
                "VpcEndpointSg",
                vpc=vpc,
                allow_all_outbound=True,
                description="HTTPS from the swegrep runtime ENIs to interface VPC endpoints",
            )
            endpoint_sg.add_ingress_rule(
                ec2.Peer.security_group_id(runtime_sg.security_group_id),
                ec2.Port.tcp(443),
                "HTTPS from swegrep AgentCore runtime",
            )
            for endpoint_id, service in [
                ("EcrApiEndpoint", ec2.InterfaceVpcEndpointAwsService.ECR),
                ("EcrDkrEndpoint", ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER),
                ("LogsEndpoint", ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS),
            ]:
                vpc.add_interface_endpoint(
                    endpoint_id,
                    service=service,
                    subnets=subnets,
                    security_groups=[endpoint_sg],
                    private_dns_enabled=True,
                )
            vpc.add_gateway_endpoint(
                "S3Endpoint",
                service=ec2.GatewayVpcEndpointAwsService.S3,
                subnets=[subnets],
            )
        elif shared_endpoint_sg_id:
            shared_endpoint_sg = ec2.SecurityGroup.from_security_group_id(
                self, "SharedEndpointSg", shared_endpoint_sg_id, mutable=True
            )
            shared_endpoint_sg.add_ingress_rule(
                ec2.Peer.security_group_id(runtime_sg.security_group_id),
                ec2.Port.tcp(443),
                "HTTPS from swegrep AgentCore runtime (reusing shared VPC endpoints)",
            )

        env_vars = {
            "SWEGREP_BETA": _recipe_beta(recipe),
            "SWEGREP_CORPUS_BUCKET": corpus_bucket.bucket_name,
            "AWS_REGION": self.region,
            "OTEL_METRICS_EXPORTER": "none",
        }

        runtime = agentcore.Runtime(
            self,
            "SweGrepAgentRuntime",
            runtime_name=runtime_name,
            agent_runtime_artifact=artifact,
            network_configuration=network_configuration,
            environment_variables=env_vars,
        )

        if gateway_sg_id:
            gateway_sg = ec2.SecurityGroup.from_security_group_id(self, "GatewaySg", gateway_sg_id, mutable=True)
            gateway_sg.add_ingress_rule(
                ec2.Peer.security_group_id(runtime_sg.security_group_id),
                ec2.Port.tcp(gateway_port),
                "rollout gateway from AgentCore swegrep runtime",
            )
            gateway_sg.add_ingress_rule(
                ec2.Peer.security_group_id(runtime_sg.security_group_id),
                ec2.Port.tcp(EVAL_ENGINE_PORT),
                "benchmark vLLM from AgentCore swegrep runtime",
            )

        corpus_bucket.grant_read(runtime.role)
        # deploy.sh reads these outputs by name.
        CfnOutput(
            self,
            "AgentRuntimeArn",
            value=runtime.agent_runtime_arn,
            description="AGENTCORE_AGENT_ARN for .env",
        )
        CfnOutput(
            self,
            "CorpusBucketName",
            value=corpus_bucket.bucket_name,
            description="SWEGREP_CORPUS_BUCKET for .env (populate with scripts/prepare_corpus.py)",
        )
        CfnOutput(
            self,
            "RuntimeSecurityGroupId",
            value=runtime_sg.security_group_id,
            description=f"Allow inbound :{gateway_port} and :{EVAL_ENGINE_PORT} from this SG on the gateway host",
        )
