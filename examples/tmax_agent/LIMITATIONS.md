# Task compatibility limitations

Known environment issues in the AgentCore evaluation of OpenThoughts-TBLite 2.0,
as of 2026-10-06. Task sources are pinned to
[`f075e463`](https://github.com/open-thoughts/OpenThoughts-TBLite/tree/f075e463472c7790b85793b392dff1fff20cc0e3).

| Task | Attribution | Evidence and impact | Current handling |
|---|---|---|---|
| [mlflow-register] | AgentCore Runtime: port 8080 requirement | The task and verifier require MLflow on `localhost:8080`, already occupied by sandboxd for the Runtime HTTP contract. Observed `Address already in use` and verifier requests reaching sandboxd instead of MLflow. | Await the platform port fix, then recheck. |
| [token-auth-websocket] | AgentCore Runtime: port 8080 requirement | The task requires port 8080 to be free; the verifier starts Spring Boot on that port. Observed startup failure and verifier requests reaching sandboxd. | Await the platform port fix, then recheck. |
| [reverse-engineer-stack-vm] | AgentCore Runtime: ARM64-only execution | The original verifier pins the SHA256 of an x86-64 validator binary. The x86-64 build matches; the ARM64 build has a different checksum and fails the integrity check. | Record as incompatible with the current platform architecture. |
| [systemd-log-monitoring] | Original task image: incomplete systemd environment | The instruction assumes systemd is preinstalled. Its `debian:bookworm-slim` Dockerfile does not install or start systemd. The evaluated container runs sandboxd as PID 1, so installing `systemctl` alone does not provide a running service manager. | Defer image changes. |

For `systemd-log-monitoring`, running a corrected image with systemd on AgentCore
has not been validated; the confirmed issue is the missing task environment.

[mlflow-register]: https://github.com/open-thoughts/OpenThoughts-TBLite/tree/f075e463472c7790b85793b392dff1fff20cc0e3/mlflow-register
[token-auth-websocket]: https://github.com/open-thoughts/OpenThoughts-TBLite/tree/f075e463472c7790b85793b392dff1fff20cc0e3/token-auth-websocket
[reverse-engineer-stack-vm]: https://github.com/open-thoughts/OpenThoughts-TBLite/tree/f075e463472c7790b85793b392dff1fff20cc0e3/reverse-engineer-stack-vm
[systemd-log-monitoring]: https://github.com/open-thoughts/OpenThoughts-TBLite/tree/f075e463472c7790b85793b392dff1fff20cc0e3/systemd-log-monitoring
