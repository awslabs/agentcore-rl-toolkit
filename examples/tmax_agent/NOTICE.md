The loop in `harness.py`, helpers in `vanillux.py`, and
`vanillux_prompts.yaml` are adapted from
[TMax](https://github.com/hamishivi/tmax), commit `6d3d606c`:
`Vanillux2Agent/agent.py`, `rl_data/generator/sample_solutions.py`, and
`rl_data/generator/vanillux_solver.py`. TMax is Apache-2.0 licensed;
see the repository's root `LICENSE`.

Copyright 2026 AllenAI. All rights reserved.

TMax's prompts derive from
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) (MIT).
The prompt text is retained unchanged.

This adaptation replaces Harbor execution with sandboxd and LiteLLM with
OpenAI Chat Completions. It omits API billing estimates and Podman warning
filtering. It retains the first tool call, as the upstream parser does, but
discards extra calls and supplies format errors as user messages to keep
the subsequent Chat Completions request valid. Sandbox output above 256 KiB
per stream is already truncated before the harness's 10,000-character
head/tail formatting; artifacts record that condition.

---

MIT License

Copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
