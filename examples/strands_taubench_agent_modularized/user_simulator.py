"""User simulator: plays the customer side of a multi-turn conversation.

``UserSimulator`` is a small, benchmark-agnostic interface: ``reset`` binds the simulator to
one episode's task and environment, and ``step`` turns the conversation history into the
user's next turn. The caller owns the conversation loop and the history; the simulator owns
user behavior (model, prompt, the user-facing projection of history, user tools, and
termination). ``TauBenchUserSimulator`` is the first implementation.

The simulated user is an environment-side component, not part of the trainable policy: its
model is called directly (not through the rollout gateway), so its tokens are never trained on.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from constants import DOMAINS_USER_TOOLS, TERMINATION_KEYWORDS, USER_MODEL_CONFIG, USER_SYSTEM_PROMPT
from strands.models import BedrockModel
from tau2.user.user_simulator import get_global_user_sim_guidelines
from utils import extract_text, make_strands_tool, run_agent_turn


@dataclass
class UserTurnResult:
    """What the user produced in one turn.

    Attributes:
        messages: Every message produced this turn (text, and for telecom any user tool
            calls and their results), in order, each tagged ``from="user"``.
        text: The text of the turn's last message, i.e. what the user finally said.
        terminated: Whether the user ended the conversation this turn.
        termination_reason: The termination keyword found in ``text`` when terminated.
        metadata: Optional implementation-specific details about the turn.
    """

    messages: list[dict]
    text: str
    terminated: bool
    termination_reason: Optional[str] = None
    metadata: Optional[dict] = None


class UserSimulator(Protocol):
    """Produces the user's side of the conversation, one turn at a time."""

    def reset(self, task: dict, env: Any) -> None:
        """Initialize simulator state for a new episode.

        Args:
            task: The episode's task definition.
            env: The episode's environment; user tools act on this same state.
        """
        ...

    def step(self, conversation: list[dict]) -> UserTurnResult:
        """Generate the next user turn given the conversation history.

        Args:
            conversation: The shared conversation history (each message tagged with
                ``from``). Not mutated; the caller appends ``UserTurnResult.messages``.

        Returns:
            The user's turn.
        """
        ...

    def describe(self) -> dict:
        """Return a JSON-serializable description of the simulator for rollout metadata."""
        ...


def detect_termination_reason(text: str) -> Optional[str]:
    """Return the first termination keyword contained in ``text``, or None."""
    return next((kw for kw in TERMINATION_KEYWORDS if kw in text), None)


def render_user_system_prompt(task: dict, domain: str) -> str:
    """Render the TauBench user system prompt for one task.

    Args:
        task: The task from the payload; its ``user_scenario`` becomes the scenario block.
        domain: The tau2 domain; domains with user tools get the tool-use guidelines.

    Returns:
        The rendered system prompt.
    """
    return USER_SYSTEM_PROMPT.format(
        global_user_sim_guidelines=get_global_user_sim_guidelines(use_tools=(domain in DOMAINS_USER_TOOLS)),
        instructions=json.dumps(task["user_scenario"]),
    )


def build_bedrock_user_model(config: dict) -> BedrockModel:
    """Instantiate the Bedrock model backing the user simulator.

    Args:
        config: A ``USER_MODEL_CONFIG`` entry (model_id, temperature, max_tokens,
            thinking_enabled, and thinking_budget when thinking is enabled).

    Returns:
        A configured ``BedrockModel``.
    """
    return BedrockModel(
        model_id=config["model_id"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        additional_request_fields={
            "thinking": (
                {"type": "enabled", "budget_tokens": config.get("thinking_budget", 1024)}
                if config["thinking_enabled"]
                else {"type": "disabled"}
            )
        },
    )


class TauBenchUserSimulator:
    """The tau2-bench user simulator: each user turn is one Strands agent (ReAct) turn.

    ``step`` projects the shared history into the user's view (``utils.convert_message_role``
    with ``to_role="user"``): the assistant's text appears as the other party, while the
    assistant's tool calls and reasoning are hidden. The agent is rebuilt from that view every
    turn, so the only per-episode state is what ``reset`` binds: the system prompt, the user
    tools, and a turn counter for logging.

    Args:
        config: User model config; defaults to ``USER_MODEL_CONFIG["bedrock"]``.
        model: The Strands model playing the user; defaults to a Bedrock model built from
            ``config``.
    """

    def __init__(self, config: Optional[dict] = None, model: Any = None):
        self.config = USER_MODEL_CONFIG["bedrock"] if config is None else config
        self.model = build_bedrock_user_model(self.config) if model is None else model
        self.system_prompt: Optional[str] = None
        self.tools: list = []
        self._turn = 0

    def reset(self, task: dict, env: Any) -> None:
        domain = task["domain"]
        self.system_prompt = render_user_system_prompt(task, domain)
        # User tools (telecom only) wrap this episode's env, so user and assistant tools act on
        # the same state.
        self.tools = (
            [make_strands_tool(env, t.name, t, "user") for t in env.get_user_tools()]
            if domain in DOMAINS_USER_TOOLS
            else []
        )
        self._turn = 0

    def step(self, conversation: list[dict]) -> UserTurnResult:
        if self.system_prompt is None:
            raise RuntimeError("reset() must be called before step()")
        response, messages = run_agent_turn(
            self._turn, "user", conversation, self.model, self.tools, self.system_prompt
        )
        self._turn += 1
        text = extract_text(response.message)
        reason = detect_termination_reason(text)
        return UserTurnResult(messages=messages, text=text, terminated=reason is not None, termination_reason=reason)

    def describe(self) -> dict:
        return {"source": "bedrock", **self.config}
