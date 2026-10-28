USER_MODEL_CONFIG = {
    "bedrock": {
        "model_id": "global.anthropic.claude-opus-4-6-v1",
        "temperature": 1.0,
        "max_tokens": 2048,
        "thinking_enabled": False,
        "thinking_budget": 1024,  # only used when thinking_enabled is True
    },
}

ASSISTANT_MODEL_DEFAULTS = {
    "max_tokens": 2048,
    "extra_body": {
        "chat_template_kwargs": {"enable_thinking": True},
    },
}

ORCHESTRATOR_CONFIG = {
    "max_turns": 50,
    "strip_thinking_from_history": True,  # if true, assistant's thinking is stripped from
    # conversation history (forces self-contained reasoning
    # per turn, avoids context length blowup in RL rollouts)
    "db_path": None,  # custom DB path (e.g., db_working.json with new records from task init)
    # if None, uses the default DB bundled with tau-bench
}


DOMAINS_USER_TOOLS = ["telecom"]


ASSISTANT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time. After tool call, you should send message to user to communicate information.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.

""".strip()

ASSISTANT_SYSTEM_PROMPT = """
<instructions>
{assistant_instruction}
</instructions>
<policy>
{assistant_policy}
</policy>
""".strip()


USER_SYSTEM_PROMPT = """
{global_user_sim_guidelines}

<scenario>
{instructions}
</scenario>
""".strip()

# Appended for stablizing user simulator
USER_SYSTEM_PROMPT += """

# Verification Checklist:
[] I've taken actions that the assistant asked, e.g. if assistant asks me to reboot device,
I literally call `reboot_device()` instead of saying "I'll do it later".
[] I've faithfully complied with the task instructions given to me, on every constraint,
without altering them.
[] If I am going to change my mind about an item the assistant has just proposed an action on
(e.g., assistant says "I will do A on item X" and I want B on that same item X instead), I MUST
raise this BEFORE confirming. I say "wait, for item X please do B instead of A, because ..." and
only confirm after the assistant updates the plan. I do NOT say "yes, proceed" and then ask to undo it
 — once the assistant has acted on item X, that action cannot be reliably reverted.
""".strip()


# User termination commands
OUT_OF_SCOPE = "###OUT-OF-SCOPE###"
STOP = "###STOP###"
TRANSFER = "###TRANSFER###"
TERMINATION_KEYWORDS = [STOP, TRANSFER, OUT_OF_SCOPE]

# Malformed tool call markers that models may leak into text
MALFORMED_TOOL_CALL_TAGS = ["<tool_call>"]

# Strands content-block keys
TOOL_USE_KEY = "toolUse"
TOOL_RESULT_KEY = "toolResult"
