# TODO — current limits on terminal-bench-2

We can run terminal-bench-2 end to end on AgentCore Runtime today (serverless
arm64 microVM). A handful of tasks still don't grade cleanly. These are the
known limits, and none of them is a problem with the agent or the scoring —
they're environment limits.

## 1. Long silent commands time out

A few heavy tasks run a single command that works for a long time while printing
nothing (for example, compiling a large library from source). The connection
that carries output has an idle limit, so after enough silence it gives up and
the task is dropped instead of being scored.

**Fix:** keep the connection alive during silent commands — send a heartbeat, or
start the command in the background and poll it. Not done yet.

## 2. One command can't run longer than an hour

A single command in a sandbox is capped at one hour by the service. Tasks that
ask for more are stopped at the cap and graded on whatever they finished. This
mostly overlaps with limit #1.

## Future: x86 / EC2

This example runs arm64-only. An earlier version could also place tasks on an
x86 EC2 capacity provider; it was removed to keep the code simple, since it
didn't improve results and added a lot of complexity (a bare x86 image ships no
`curl`/`python`, and its root is capability-stripped, so the harness had to
inject a static `curl` + CA bundle before anything could run). Re-adding x86
would also have to handle a per-image size limit — a few x86 images come out
around 6 GB, over the limit, so their runtime can't be created (the same tasks
are smaller and run fine on arm64).
