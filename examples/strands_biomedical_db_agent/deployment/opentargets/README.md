# Open Targets GraphQL service

This recipe runs the official Open Targets Platform 26.06 disk images with API
26.6.3, ClickHouse 25.8.18.1, and OpenSearch 3.5.0. Release versions and the upstream
archive SHA-1 checksums are recorded in `release-26.06.env`.

Use a dedicated Linux database host with Docker Engine, Docker Compose v2, `curl`,
`pigz`, and at least 300 GiB of storage for extracted data and download scratch
space. Set `TMPDIR` to a sufficiently large filesystem. Provision and mount the
data volume before running these commands.

From this directory:

```bash
export OT_DATA_ROOT=/path/to/persistent/opentargets/26.06
bash prepare.sh
# OpenSearch runs as the UID/GID pinned in the release file.
sudo chown -R 10021:10021 "$OT_DATA_ROOT/opensearch"
docker compose --env-file release-26.06.env up -d --build
docker compose --env-file release-26.06.env ps
```

`prepare.sh` downloads to `$TMPDIR`, verifies each archive, and extracts into the
new data directory. A completed directory is reused. An existing directory without
a matching completion marker is rejected; inspect any interrupted extraction and
choose a clean destination before retrying. The script does not format disks or
modify host services.

The API binds to loopback on port 8081 by default. For access from AgentCore in a
VPC, set `OT_API_BIND` to the host's private interface address before starting
Compose, and allow TCP 8081 from the Runtime security group. `OT_API_PORT` changes
the host port. ClickHouse and OpenSearch remain bound to loopback on the host.

Check release metadata:

```bash
curl --fail http://127.0.0.1:8081/api/v4/graphql \
  -H 'Content-Type: application/json' \
  -d '{"query":"{ meta { dataVersion { year month iteration } } }"}'
```

If you selected a private bind address, use that address in the check. Configure
the agent with:

```dotenv
OPEN_TARGETS_GRAPHQL_URL=http://DATABASE_PRIVATE_HOST:8081/api/v4/graphql
OPEN_TARGETS_EXPECTED_RELEASE=26.06
```

From the installed example environment, verify schema loading:

```bash
python -c 'from biomedical_db_agent.opentargets.schema import load_schema; print(len(load_schema()))'
```

Then run an illustrative task from the example README to verify query execution.
The agent rejects a release mismatch. Treat changes to the database release,
API image, and agent prompt/schema as a coordinated update.

Stop the services with:

```bash
docker compose --env-file release-26.06.env down
```

Database files remain in `OT_DATA_ROOT`. Preserve or remove that directory according
to your storage needs. See the [official disk-image source](https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.06/disk_images/)
and [Platform API](https://github.com/opentargets/platform-api) for upstream release information.
