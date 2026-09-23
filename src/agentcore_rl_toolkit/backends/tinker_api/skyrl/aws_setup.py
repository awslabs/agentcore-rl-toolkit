"""Small AWS bootstrap for this example; SkyPilot provisions the EC2 instance."""

import ipaddress
import json
import time

from botocore.exceptions import ClientError

PROFILE = "skypilot-v1"
OWNER_TAG = "skyrl-tinker-endpoint"
TRUST = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}


def bootstrap_policy(account):
    """Permissions an administrator grants the deployer, never the GPU role."""
    role = f"arn:aws:iam::{account}:role/{PROFILE}"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "iam:GetInstanceProfile",
                    "iam:CreateInstanceProfile",
                    "iam:AddRoleToInstanceProfile",
                ],
                "Resource": f"arn:aws:iam::{account}:instance-profile/{PROFILE}",
            },
            {
                "Effect": "Allow",
                "Action": ["iam:GetRole", "iam:CreateRole"],
                "Resource": role,
            },
            {
                "Effect": "Allow",
                "Action": "iam:PassRole",
                "Resource": role,
                "Condition": {"StringEquals": {"iam:PassedToService": "ec2.amazonaws.com"}},
            },
        ],
    }


def get_optional(call, **kwargs):
    try:
        return call(**kwargs)
    except ClientError as error:
        if error.response["Error"]["Code"] != "NoSuchEntity":
            raise
        return None


def ensure_profile(iam):
    profile = get_optional(iam.get_instance_profile, InstanceProfileName=PROFILE)
    roles = profile["InstanceProfile"]["Roles"] if profile else []
    if roles and [role["RoleName"] for role in roles] != [PROFILE]:
        raise ValueError(f"{PROFILE} already contains another role; refusing to replace it")
    role = get_optional(iam.get_role, RoleName=PROFILE)
    if role:
        # Do not alter an existing role's trust or permissions.
        statements = role["Role"]["AssumeRolePolicyDocument"]["Statement"]
        if isinstance(statements, dict):
            statements = [statements]
        if not any(
            statement.get("Effect") == "Allow"
            and statement.get("Principal") == {"Service": "ec2.amazonaws.com"}
            and statement.get("Action") in ("sts:AssumeRole", ["sts:AssumeRole"])
            and not statement.get("Condition")
            for statement in statements
        ):
            raise ValueError(f"{PROFILE} does not have the expected EC2 trust policy")
    else:
        iam.create_role(RoleName=PROFILE, AssumeRolePolicyDocument=json.dumps(TRUST))
    if profile is None:
        iam.create_instance_profile(InstanceProfileName=PROFILE)
    if not roles:
        iam.add_role_to_instance_profile(InstanceProfileName=PROFILE, RoleName=PROFILE)
        for _ in range(30):
            populated = iam.get_instance_profile(InstanceProfileName=PROFILE)
            if [r["RoleName"] for r in populated["InstanceProfile"]["Roles"]] == [PROFILE]:
                break
            time.sleep(2)
        else:
            raise TimeoutError("IAM profile binding has not propagated; retry shortly")
    # A populated profile prevents SkyPilot from attaching its broad default
    # policies. Deliberately never call PutRolePolicy or AttachRolePolicy here.
    return PROFILE


def resolve_subnet(ec2, subnet_id):
    subnet = ec2.describe_subnets(SubnetIds=[subnet_id])["Subnets"][0]
    name = next(
        (tag["Value"] for tag in subnet.get("Tags", []) if tag["Key"] == "Name"),
        None,
    )
    if not name or any(char in name for char in "*?"):
        raise ValueError("The subnet needs a literal Name tag for SkyPilot selection")
    matches = ec2.describe_subnets(Filters=[{"Name": "tag:Name", "Values": [name]}])["Subnets"]
    if len(matches) != 1 or matches[0]["SubnetId"] != subnet_id:
        raise ValueError("The subnet Name tag must identify exactly one subnet in the region")
    return subnet, name


def verify_instance_network(ec2, cloud_cluster_name, subnet_id, group_id):
    response = ec2.describe_instances(
        Filters=[
            {"Name": "tag:ray-cluster-name", "Values": [cloud_cluster_name]},
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running", "stopping", "stopped"],
            },
        ]
    )
    instances = [instance for reservation in response["Reservations"] for instance in reservation["Instances"]]
    if len(instances) != 1:
        raise ValueError("Expected exactly one EC2 instance for this SkyPilot cluster")
    instance = instances[0]
    if instance["SubnetId"] != subnet_id or {group["GroupId"] for group in instance["SecurityGroups"]} != {group_id}:
        raise ValueError(
            "Existing cluster uses a different subnet/security group. "
            "Choose another cluster name; its network will not be changed automatically."
        )
    return instance["InstanceId"]


def client_network(value):
    network = ipaddress.ip_network(value, strict=True)
    if network.version != 4 or network.prefixlen == 0:
        raise ValueError("Use a specific IPv4 client CIDR, typically PRIVATE_IP/32")
    return str(network)


def ingress_rules(group_id, cidr):
    return [
        {
            "IpProtocol": "tcp",
            "FromPort": port,
            "ToPort": port,
            "IpRanges": [{"CidrIp": cidr}],
        }
        for port in (22, 18080)
    ] + [{"IpProtocol": "-1", "UserIdGroupPairs": [{"GroupId": group_id}]}]


def permits(rule, wanted):
    if wanted["IpProtocol"] == "-1":
        return rule["IpProtocol"] == "-1" and any(
            pair.get("GroupId") == wanted["UserIdGroupPairs"][0]["GroupId"] for pair in rule.get("UserIdGroupPairs", [])
        )
    return (
        rule["IpProtocol"] in ("tcp", "-1")
        and (rule["IpProtocol"] == "-1" or rule["FromPort"] <= wanted["FromPort"] <= rule["ToPort"])
        and any(
            ipaddress.ip_network(wanted["IpRanges"][0]["CidrIp"]).subnet_of(ipaddress.ip_network(item["CidrIp"]))
            for item in rule.get("IpRanges", [])
        )
    )


def ensure_security_group(ec2, vpc_id, cluster, cidr, group_id=None):
    name = f"skyrl-tinker-{cluster}"
    if group_id:
        groups = ec2.describe_security_groups(GroupIds=[group_id])["SecurityGroups"]
    else:
        groups = ec2.describe_security_groups(
            Filters=[
                {"Name": "vpc-id", "Values": [vpc_id]},
                {"Name": "group-name", "Values": [name]},
            ]
        )["SecurityGroups"]
    if groups:
        group = groups[0]
        if group["VpcId"] != vpc_id:
            raise ValueError("The security group and subnet must be in the same VPC")
        if not group_id and {"Key": OWNER_TAG, "Value": cluster} not in group.get("Tags", []):
            raise ValueError(f"Security group {name} exists but is not owned by this example")
    else:
        created = ec2.create_security_group(
            GroupName=name,
            Description="Private SkyRL Tinker endpoint",
            VpcId=vpc_id,
            TagSpecifications=[
                {
                    "ResourceType": "security-group",
                    "Tags": [{"Key": OWNER_TAG, "Value": cluster}],
                }
            ],
        )
        group = {
            "GroupId": created["GroupId"],
            "GroupName": name,
            "IpPermissions": [],
        }
    for rule in ingress_rules(group["GroupId"], cidr):
        if any(permits(existing, rule) for existing in group["IpPermissions"]):
            continue
        if group_id:
            raise ValueError(f"Provided security group is missing required ingress: {rule}")
        try:
            ec2.authorize_security_group_ingress(GroupId=group["GroupId"], IpPermissions=[rule])
        except ClientError as error:
            if error.response["Error"]["Code"] != "InvalidPermission.Duplicate":
                raise
    return group["GroupId"], group["GroupName"]
