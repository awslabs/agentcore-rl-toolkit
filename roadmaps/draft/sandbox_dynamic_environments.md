# Dynamic Sandbox Environments on AgentCore

## Summary

The sandbox SDK can support dynamic, Harbor-style task environments under a
single AgentCore runtime ARN, but not by running Docker-in-Docker. The viable
approach is to deploy one fixed loader runtime image and dynamically load one
Docker image filesystem per AgentCore session.

Each session should load, execute, collect results, and clean up one environment
independently. The same runtime ARN can then be reused across tasks without
requiring nested containers or one runtime per task image.

## Findings

AgentCore runtime binds a runtime ARN to one deployed container image. A session
cannot directly choose a different runtime image at `start()` time.

Docker-in-Docker was tested as a possible way to load arbitrary task images under
one runtime ARN. It was not viable in the tested AgentCore configuration:
`dockerd` could be started only in a restricted mode, and nested container
execution failed. The implementation should avoid relying on Docker-in-Docker.

## Proposed Design

Deploy a fixed AgentCore loader image containing:

- `agentcore-sandboxd`
- `crane` or `skopeo` for pulling/exporting OCI images
- `tar`
- `proot`
- shell utilities
- optional package managers and runtime tools useful for task setup

Per session:

1. Start an AgentCore sandbox session.
2. Pull/export the requested Docker image filesystem.
3. Unpack it into a session-local rootfs directory.
4. Run commands inside that rootfs with `proot`.
5. Run task/verifier commands.
6. Collect stdout/stderr/reward/artifacts.
7. Delete the rootfs and task staging directory.
8. Terminate the AgentCore session.

Example internal command flow:

```bash
rm -rf /runtime/rootfs /runtime/task
mkdir -p /runtime/rootfs /runtime/task

crane export ubuntu:24.04 - | tar -C /runtime/rootfs -xf -

proot \
  -R /runtime/rootfs \
  -b /runtime/task:/harbor \
  -b /tmp:/tmp \
  -w /app \
  /bin/sh -lc 'echo hello from loaded image'
```

This approximates Docker image filesystem semantics without starting nested
containers.

## SDK Surface

Initial public API sketch:

```python
client = SandboxClient(runtime_arn=loader_runtime_arn)

with client.start() as sb:
    env = sb.load_image("ubuntu:24.04")
    result = env.exec("python3 --version")
    env.destroy()
```

Harbor-oriented sketch:

```python
with client.start() as sb:
    env = sb.load_harbor_task(task_dir="path/to/task")
    env.exec_agent("bash /harbor/solution/solve.sh")
    result = env.run_verifier()
```

Supporting methods likely needed:

- `Sandbox.upload_file(source, target)`
- `Sandbox.upload_dir(source, target)`
- `Sandbox.load_image(image_ref, name=None, workdir=None)`
- `LoadedEnvironment.exec(command, timeout=None, cwd=None, env=None, user=None)`
- `LoadedEnvironment.destroy()`

## Harbor Compatibility

This design should work well for Harbor tasks whose Dockerfiles primarily define
filesystem state:

- base OS image
- package installs
- copied files
- users/groups
- environment variables
- working directory

It does not fully support Docker runtime semantics:

- no nested container isolation
- no Docker Compose or multi-container services
- no Docker networking model
- no per-environment cgroups
- no automatic `ENTRYPOINT`/`CMD` execution unless explicitly implemented
- limited support for services, systemd, privileged operations, or kernel
  features

For Harbor-format tasks, the first target should be single-container tasks with
plain shell or Python verifier scripts.

## Implementation Plan

### Phase 1: Loader Image Prototype

- Create an AgentCore-compatible loader Dockerfile.
- Include `agentcore-sandboxd`, `crane`, `proot`, `tar`, and shell tools.
- Deploy it once as a reusable AgentCore runtime ARN.
- Add a manual probe script that loads `ubuntu:24.04` and runs a simple command
  via `proot`.

### Phase 2: SDK File Transfer

- Add `upload_file` and `upload_dir` to `Sandbox`.
- Implement upload using shell-safe streamed/base64/tar chunks through
  `Sandbox.exec()` until a better AgentCore file API exists.
- Add focused tests for quoting, binary safety, nested directories, empty
  directories, and cleanup.

### Phase 3: Dynamic Image Loading

- Add `LoadedEnvironment`.
- Implement `Sandbox.load_image(image_ref)`.
- Use `crane export <image> - | tar -C <rootfs> -xf -`.
- Execute commands through `proot -R <rootfs>`.
- Add cleanup and idempotent `destroy()`.

### Phase 4: Harbor Task Smoke Path

- Add `load_harbor_task(task_dir)`.
- Parse minimal task structure:
  - `environment/Dockerfile`
  - `[environment].docker_image`
  - `instruction.md`
  - `tests/`
  - optional `solution/`
- Initially support tasks with `[environment].docker_image`.
- For Dockerfile-only tasks, support a documented subset or require a prebuilt
  image in the first version.
- Run verifier and read `/logs/verifier/reward.txt`.

### Phase 5: Integration Tests

- Add env-gated live integration tests.
- Skip unless `SANDBOX_LOADER_RUNTIME_ARN` is set.
- Test:
  - load public image
  - execute command via `proot`
  - upload a small Harbor-style task
  - run verifier
  - cleanup

## Open Questions

- Whether to use `crane` or `skopeo` as the default image tool.
- How to handle private registry authentication cleanly.
- Whether `proot` is available and reliable on the AgentCore kernel/runtime.
- How much Dockerfile parsing/build behavior should be supported versus
  requiring prebuilt images.
- Whether command execution should preserve Docker image `ENV`, `WORKDIR`,
  `USER`, `ENTRYPOINT`, and `CMD` metadata.

## Recommendation

Proceed with the dynamic rootfs-loading approach. It matches the desired
single-runtime-ARN workflow and avoids the nested-container limitations observed
with Docker-in-Docker on AgentCore.
