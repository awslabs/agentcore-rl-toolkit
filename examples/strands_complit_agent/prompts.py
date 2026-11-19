"""The single user prompt used for every set (no system message). Verbatim from upstream complit/prompts.py
(dayoon-ko/CompLit@2654a93f, MIT)."""

TASK = (
    "A researcher describes, in their own words, a paper they want to find. The "
    "description gives the paper's distinguishing details (methods, datasets, metrics, "
    "venue, year, authors, comparisons). Identify the paper being described."
)

CONTRACT = (
    "End your reply with ONLY a single JSON object on the final line and nothing after it, with an "
    "empty title if no paper satisfies the question:\n"
    '{"title": "<exact verified paper title, or empty if no answer>", '
    '"arxiv_id": "<verified arxiv id e.g. 1709.00308, or empty>", '
    '"rationale": "<one paragraph summary of how you arrived at this conclusion>", '
    '"confidence": "high|medium|low"}'
)


def tool_paragraph(use_read_paper: bool, use_web_search: bool, read_paper_images: int) -> str:
    descs = [
        "arxiv_search runs a free-text query over arXiv metadata (titles, abstracts, "
        "authors) and returns matching papers as a list with title, arXiv id, year, "
        "authors and a truncated abstract."
    ]
    if use_read_paper:
        fig = (
            (f", with up to {read_paper_images} of its figures attached as images in document order")
            if read_paper_images > 0
            else ""
        )
        chk = ", including what its figures show," if read_paper_images > 0 else ""
        descs.append(
            "read_paper takes an arXiv id and returns that paper's full text (title, "
            f"abstract, body, tables; truncated when long){fig}, so a candidate{chk} can "
            "be checked against the request before answering."
        )
    if use_web_search:
        descs.append(
            "web_search is a general web (Google) search that returns result titles, urls "
            "and snippets; it covers venues, task framings, reported numbers and other "
            "information that arXiv metadata does not contain."
        )
    n = {1: "one tool", 2: "two tools", 3: "three tools"}[len(descs)]
    return f"You have {n}. " + " ".join(descs)


def build_prompt(
    question: str, *, use_read_paper: bool = True, use_web_search: bool = True, read_paper_images: int = 0
) -> str:
    return (
        f"{TASK}\n\n"
        f"{tool_paragraph(use_read_paper, use_web_search, read_paper_images)}\n\n"
        'Researcher\'s request:\n"""\n'
        f"{question}\n"
        '"""\n\n'
        f"{CONTRACT}"
    )


# ---- forced final answer (single-turn) -------------------------------------------------
# A run that ends without committing (no answer JSON) gets ONE tool-free call with this prompt,
# as the paper does for the Inspect agents ("forced final answer", counted toward accuracy).
FORCED_ANSWER = (
    "The search budget is exhausted, so no more tool calls are available. Based on "
    "what you have seen so far, give your final answer now.\n\n" + CONTRACT
)


# ---- Cumulative (multi-turn) -----------------------------------------------------------
# Turn 1 opens with this task statement and the tool paragraph; every turn wraps the
# researcher's message as 'Researcher:\n"""\n...\n"""'; the last turn appends the contract.
FRAMED_TASK_CONV = (
    "A researcher is describing, over a chat conversation, a paper they want to find. The "
    "description arrives a few details at a time: each message adds distinguishing details "
    "(methods, datasets, metrics, venue, year, authors, comparisons), and every detail "
    "given so far is binding. A near-miss that violates even one is the wrong paper. "
    "Respond to each message as the conversation unfolds; when the researcher asks you to "
    "commit, answer in the format they ask for."
)

TURN_NUDGE = (
    "You are out of search budget for this reply. Do NOT search or fetch anything further. "
    "Using only what you have already found and verified, reply to the researcher now."
)

TOOL_BUDGET_MSG = (
    "The tool budget for this turn is exhausted: no more tool calls until the researcher's "
    "next message. Reply to the researcher now using what you have already found."
)


def researcher_turn(
    message: str, *, turn: int, n_turns: int, use_web_search: bool = True, read_paper_images: int = 0
) -> str:
    body = 'Researcher:\n"""\n' + message + '\n"""'
    if turn == 1:
        tools = tool_paragraph(True, use_web_search, read_paper_images)
        body = f"{FRAMED_TASK_CONV}\n\n{tools}\n\n{body}"
    if turn == n_turns:
        body += "\n\n" + CONTRACT
    return body
