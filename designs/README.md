# Design Documents

This directory contains design documents for significant features and architectural
changes. A design remains here after implementation as part of the project's decision
history.

Design status and implementation progress are separate:

- **Status** records the design decision: `Proposed`, `Accepted`, `Deprecated`, or
  `Superseded`.
- **Implementation** records delivery progress: `Not started`, `In progress`, or
  `Shipped`.

Every design document must declare both fields immediately after its title.

Do not move documents between directories when either status changes. Update the metadata
and this index instead so links remain stable.

This directory does not track project priorities. Planned work belongs in GitHub issues or
a GitHub Project; these documents preserve the rationale and decisions behind significant
changes.

## Index

| Design | Status | Implementation |
| --- | --- | --- |
| [Linear-history mode for the rollout gateway](./gateway_linear.md) | Accepted | Shipped |
| [Stable optimizer-step batching for variable-row verl rollouts](./verl_variable_trajectory_batching.md) | Accepted | Shipped |
| [Runtime Invocation Protocol](./runtime_invocation_protocol.md) | Proposed | Not started |
| [Dynamic Sandbox Environments on AgentCore](./sandbox_dynamic_environments.md) | Proposed | Not started |

## Lifecycle

1. Create a design with `Status: Proposed` before implementing a significant public API or
   architectural change.
2. Change the status to `Accepted` when the design is approved. Acceptance does not imply
   that implementation is complete.
3. Update `Implementation` as delivery progresses, and link the implementation pull
   request from the document.
4. Keep accepted designs as historical records after they ship.
5. Mark a design `Deprecated` when it no longer represents a supported direction, or
   `Superseded` when another design replaces it. A superseded design must link to its
   replacement.

Small fixes and implementation details that do not introduce a durable architectural
decision do not need a design document.
