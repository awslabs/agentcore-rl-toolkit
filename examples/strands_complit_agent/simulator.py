"""The simulated researcher for the Cumulative setting, verbatim from upstream complit/simulator.py
(dayoon-ko/CompLit@2654a93f, MIT) except for the model client.

Each turn's prepared utterance (from the dataset) is sent as written, unless the agent's last reply asked something
or proposed a paper (`ADAPTER_RESPONSE_RE`). Then the simulator model (temperature 0) delivers the prepared text with
one added sentence answering the agent, and a tagger checks that every detail scheduled for the turn was really
stated (up to 3 attempts).

The model is a Bedrock model through the Converse API (upstream: gpt-5.4-mini through OpenRouter). Converse needs
the chat to open with a user message, so the simulator's view (roles inverted: `assistant` is the researcher it
plays) opens with OPENER, and the turn instruction upstream sends as a final system message is a second system
block. The tagger's JSON is read from its reply (upstream: JSON mode).
"""

import json
import re

import boto3

SIM_TEMPERATURE = 0.0
OPENER = "(The chat begins.)"


class SimModel:
    """The simulator's model on Bedrock, at temperature 0."""

    def __init__(self, model_id: str, region: str):
        self.model_id, self.client = model_id, boto3.client("bedrock-runtime", region_name=region)

    def complete(self, system: list[str], messages: list[dict], max_tokens: int) -> str:
        r = self.client.converse(
            modelId=self.model_id,
            system=[{"text": s} for s in system],
            messages=[{"role": m["role"], "content": [{"text": m["content"] or " "}]} for m in messages],
            inferenceConfig={"temperature": SIM_TEMPERATURE, "maxTokens": max_tokens},
        )
        return "".join(b.get("text", "") for b in r["output"]["message"]["content"]).strip()


TYPE_ORDER = {"content": 0, "vision": 1, "multi_doc": 2, "metadata": 3}


def fact_line(c):
    """One constraint as the sim/tagger-facing line; the [type/field] tag gives degenerate
    values ('true', a bare number) their meaning."""
    return f"[{c.get('type')}/{c.get('field')}] {c.get('rendered_value')}"


def canonical_constraints(q, type_order=""):
    """Default order content -> vision -> multi_doc; `type_order` overrides which type leads."""
    cs = q.get("constraints") or []
    order = TYPE_ORDER
    if type_order:
        order = {t.strip(): i for i, t in enumerate(type_order.split(",")) if t.strip()}
    return sorted(cs, key=lambda c: order.get(c.get("type"), 9))


def simulator_turn(client, sim_sys, sim_msgs, instruction):
    """-> message. The simulator's own view of the chat has the roles inverted: `assistant` is the researcher it
    plays, `user` is the agent."""
    return client.complete(
        [sim_sys, "THIS MESSAGE\n" + instruction], [{"role": "user", "content": OPENER}] + sim_msgs, max_tokens=4000
    )


TAG_SYS = (
    "You are given a numbered list of FACTS and one chat MESSAGE a researcher just sent. "
    "Decide which facts the message conveys. Mark a fact ONLY if the message states that "
    "fact's SPECIFIC content (a faithful paraphrase counts). Mentioning merely the general "
    "topic, area, or field the fact belongs to does NOT count, and saying they do not "
    "know or remember something does NOT count. The [tags] name each fact's attribute -- "
    "use them to understand the fact, they do not need to appear in the message. For every "
    "fact you mark, quote the contiguous substring of MESSAGE that conveys it, copied "
    "EXACTLY from MESSAGE. Reply with ONLY a JSON object: "
    '{"revealed": [{"fact": <number>, "quote": "<exact substring of MESSAGE>"}]}. '
    "Empty list if none."
)


def _squash(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


_NUM_WORDS = {
    "0": ("zero",),
    "1": ("one", "single"),
    "2": ("two", "both"),
    "3": ("three",),
    "4": ("four",),
    "5": ("five",),
    "6": ("six",),
    "7": ("seven",),
    "8": ("eight",),
    "9": ("nine",),
    "10": ("ten",),
    "100": ("hundred",),
    "1000": ("thousand",),
}


def _numeric_tokens(value):
    """Distinctive numerals in a constraint value ('98%', '0.5-0.8', '2000-4000 sentences')."""
    return re.findall(r"\d+(?:\.\d+)?", value or "")


def _states_numbers(quote, message, value):
    """A numeric fact counts as conveyed only when its numerals are actually spoken.

    Numerals are checked against the quote first, then the whole message.
    """
    nums = _numeric_tokens(value)
    if not nums:
        return True
    hay = _squash(quote) + " || " + _squash(message)
    for n in nums:
        alt = _NUM_WORDS.get(n, ())
        if n in hay or any(w in hay for w in alt):
            continue
        # 0.50 vs 0.5, 2,000 vs 2000
        loose = n.rstrip("0").rstrip(".") if "." in n else n
        if loose and loose in hay.replace(",", ""):
            continue
        return False
    return True


def tag_revealed(client, unrevealed, message):
    """-> indices into unrevealed conveyed by message.

    A fact counts only when the tagger quotes a substring of the message that states it
    and the quote really occurs in the message.
    """
    if not unrevealed:
        return []
    lst = "\n".join(f"{i}. {fact_line(c)}" for i, c in enumerate(unrevealed))
    try:
        reply = client.complete(
            [TAG_SYS], [{"role": "user", "content": f"FACTS:\n{lst}\n\nMESSAGE:\n{message}"}], max_tokens=800
        )
        items = json.loads(reply[reply.find("{") : reply.rfind("}") + 1] or "{}").get("revealed") or []
        hay = _squash(message)
        idx = set()
        for it in items:
            if isinstance(it, int):  # tolerate the old bare-number shape
                i, quote = it, None
            elif isinstance(it, dict):
                i, quote = it.get("fact"), it.get("quote")
            else:
                continue
            if not isinstance(i, int) or not 0 <= i < len(unrevealed):
                continue
            if not (quote and _squash(quote) and _squash(quote) in hay):
                continue
            if not _states_numbers(quote, message, unrevealed[i].get("rendered_value")):
                continue  # numerals not spoken -> keep it scheduled
            idx.add(i)
        return sorted(idx)
    except Exception:
        return []


SIM_SYS_SCHEDULED_NEED = """You are role-playing a researcher talking to a literature-search \
assistant in a chat. Your project needs ONE specific paper and you have NOT read it: your \
project puts a set of requirements on that paper, and you are laying them out over the \
conversation.

YOUR SCENARIO
Your overall motivation:
{intention}

The requirements you have raised so far, or are raising in this message. You will bring up more \
facets of your need later in the conversation; until then this is everything you have put on \
the table:
{all_facts}

RULES
- Write ONLY the researcher's next chat message. No narration, no lists of "requirements", no \
  meta-commentary, no quotation marks around the whole message. 2-4 sentences, natural chat tone.
- Speak in the first person as a working researcher. Phrase each requirement in your own words \
  as a need of YOUR project (something you must build on, compare against, or see evidence of), \
  never as something you recall about a particular paper; keep every number, unit, and quoted \
  work name exactly as written.
- Plain researcher language: "requirement" is OUR word in these instructions, never yours. Do \
  not call the paper a "candidate", and do not call what you need "constraints", \
  "requirements", "criteria", or "filters" -- say the need itself ("the method has to ...", \
  "I need it shown that ..."). Such a word is fine only when it is part of a requirement's own \
  technical content.
- You do not know which paper satisfies your need: never guess or invent a title, an author, a \
  venue, or an arXiv id.
- Do not invent requirements. Your scenario is everything you know. If the assistant asks about \
  an aspect your scenario does not cover, make clear in your own words that your project has no \
  requirement there, in whatever words and position fit the message naturally; vary the phrasing \
  and never reuse a wording you already used earlier in the conversation. Never answer a direct \
  question with a guess or a hedge that sounds like a fact.
- Do not accept or reject a paper the assistant proposes: you have not read it, so you cannot \
  tell whether it fits. When it proposes a specific paper, treat the proposal as unverified: \
  from then on, frame every requirement you bring up as a CHECK on it ("for it to work for me, \
  it should also ..."), asking the assistant to make sure the paper satisfies each remaining \
  requirement too, until every requirement has been checked. Never drop or soften a requirement \
  because of the proposal, and never state or imply that the proposal is right or wrong.
- NEVER name, endorse, rank, or choose between candidate papers the assistant brings up, and do \
  not repeat a title or author name from the assistant's messages. You cannot tell which \
  candidate fits. If the assistant asks you to pick one, say you cannot tell and ask it to \
  decide from the requirements you gave.
- Your requirements are FIXED. Never adopt, quote, or accept a value, number, name or claim the \
  assistant reports, and never relax a requirement to fit it ("close enough", "that would do"). \
  If the assistant reports something that differs from a requirement, restate the requirement \
  as your project needs it and leave the discrepancy for the assistant to resolve.
- Do not repeat requirements you have already stated in earlier messages; add the new ones.
- When a requirement contrasts what you need against another work whose name is quoted, that \
  other work is one you already know and have set aside: keep its name spelled exactly as \
  quoted, and make clear the paper you need is NOT it."""

ADAPTER_RESPONSE_RE = re.compile(
    r"\?|\*\*|arxiv|[\"“][A-Z]"
    r"|\bplease\s+(?:share|tell|provide|give|send|describe|specify|clarify|paste|include)\b"
    r"|\blet\s+me\s+know\b|\bcould\s+you\s+(?:share|tell|provide|give|clarify|specify)\b",
    re.I,
)


def adapter_instruction(ref, is_last):
    lines = [
        "You have a PREPARED MESSAGE for this turn; it already carries everything this turn must say:",
        '"""',
        ref,
        '"""',
        "Deliver it as your next chat message:",
        "- The assistant's last message asked you something or proposed a paper. You MUST "
        "add exactly ONE sentence of your own dealing with it (it may come before the "
        "prepared text): when something you have already given, or are giving in the "
        "prepared text, covers the question, answer it with that; otherwise say plainly "
        "that your project has no requirement on that point. Never ignore the question, "
        "and never answer it with anything beyond what you have. When a specific paper "
        'was proposed, you may also recast the prepared sentences as checks on it ("for '
        'it to work for me, it should also ...") without changing what any of them '
        "says.",
        "- Beyond that, keep the prepared sentences essentially as written: keep every "
        "number, unit, and quoted work name exactly as written, do not drop a sentence, "
        "and do not add other new content.",
        "- Your added sentence speaks plain researcher language: never the words "
        '"requirements", "constraints", "criteria", or "candidate" -- refer to '
        '"what I described", "the points I raised", or the things themselves.',
    ]
    if is_last:
        lines.append(
            "- This is your LAST message: keep its closing request that the "
            "assistant commit now: its single best answer, or that no paper satisfies everything you have described."
        )
    return "\n".join(lines)


def naturalize_turn(client, sim_sys, sim_msgs, instruction, scheduled, max_retries=2):
    """Speak one turn and verify the scheduled facts were really stated.

    -> (message, conveyed indices into `scheduled`)

    The check is the same quote-and-numeral tagger the conversation uses for its ledger: the
    naturalizer is free to rephrase, but a fact only counts as delivered when a substring of the
    message states it. A message that drops one is regenerated with that fact named, up to
    `max_retries` times; what is still missing after that is recorded, never silently dropped.
    """
    attempt_instr = instruction
    best_msg, best_conveyed = "", []
    for _ in range(max_retries + 1):
        msg = simulator_turn(client, sim_sys, sim_msgs, attempt_instr)
        if not scheduled:
            return msg, []
        conveyed = tag_revealed(client, scheduled, msg)
        if len(conveyed) > len(best_conveyed) or not best_msg:
            best_msg, best_conveyed = msg, conveyed
        missing = [i for i in range(len(scheduled)) if i not in set(conveyed)]
        if not missing:
            return msg, conveyed
        attempt_instr = (
            instruction
            + "\n\nYOUR PREVIOUS DRAFT did not actually state "
            + str(len(missing))
            + " of the required detail(s):\n"
            + "\n".join(f"- {fact_line(scheduled[i])}" for i in missing)
            + "\nRewrite the message so each of those is stated explicitly, with its "
            "numbers and names intact, keeping the rest of the message as it was."
        )
    return best_msg, best_conveyed
