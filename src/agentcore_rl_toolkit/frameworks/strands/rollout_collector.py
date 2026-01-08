"""Base rollout collector for Strands framework with hooks-based data collection."""


class RolloutCollector:
    """Base rollout collector using Strands hooks to collect conversation data and compute rewards."""

    def __init__(self):
        self.turns = []

    def register_hooks(self, registry):
        """Register hooks for rollout collection with Strands HookRegistry."""
        try:
            from strands.experimental.hooks import BeforeModelInvocationEvent
            from strands.hooks import AfterInvocationEvent
        except ImportError:
            raise ImportError("Strands not installed. Install with: " "uv pip install strands-agents[openai]") from None

        registry.add_callback(BeforeModelInvocationEvent, self.collect_messages)
        registry.add_callback(AfterInvocationEvent, self.prepare_rollout)

    def collect_messages(self, event: "BeforeModelInvocationEvent"):  # noqa: F821
        """Collect messages before model invocation."""

        agent = event.agent
        tool_specs = agent.tool_registry.get_all_tool_specs()
        formatted_request = agent.model.format_request(agent.messages, tool_specs, agent.system_prompt)

        # Store the complete formatted messages for this turn
        self.turns.append(
            {
                "turn_id": len(self.turns),
                "formatted_request": formatted_request,
            }
        )

    def prepare_rollout(self, event: "AfterInvocationEvent"):  # noqa: F821
        if len(self.turns) == 0:
            return

        # Since hook is triggered before model invocation, all turns end with the user message
        # This loop turns [[u1], [u1, a1, u2], [u1, a1, u2, a2, u3], ..., [u1, ...a(n-1), u(n)]] into
        # [[u1, a1], [u1, a1, u2, a2], [u1, a1, u2, a2, u3, a3], ..., [u1, ...a(n-1), u(n)]]
        for i in range(1, len(self.turns)):
            self.turns[i - 1]["formatted_request"]["messages"].append(
                self.turns[i]["formatted_request"]["messages"][-2],  # second to last is assistant message
            )

        # Gather final response
        agent = event.agent
        if agent.messages[-1]["role"] == "assistant":  # successful invocation
            tool_specs = agent.tool_registry.get_all_tool_specs()
            formatted_request = agent.model.format_request(agent.messages, tool_specs, agent.system_prompt)
            final_response = formatted_request["messages"][-1]
            self.turns[-1]["formatted_request"]["messages"].append(final_response)

    def get_rollout_data(self) -> list:
        """Return collected rollout data without computing rewards."""
        return self.turns
