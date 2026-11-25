"""Single source of truth for turning a model response into a scored answer.

The training reward and the reported metric are this one module: the reward is
exact match between the canonical prediction and the canonical label.

Two principles decide every rule below:

1. **Authority.** What counts as correct is the benchmark's, never the trainer's. The
   canonical form and the comparison are the metric's.
2. **Never guess.** Where a response does not commit to one answer, extraction
   returns ``None`` and the reward is 0. The alternative -- picking the last
   candidate a response happens to mention -- is learnable: a policy can
   enumerate candidates and let the grader commit on its behalf, which removes
   the pressure to answer at all. Two upstream rules did exactly that and are
   deliberately absent here (see ``_rsid`` and ``_gene``).
"""

from __future__ import annotations

import re
from typing import Any

LAB_BENCH_TASKS = frozenset({"lab_bench_dbqa", "lab_bench_seqqa"})
LETTER_TASKS = LAB_BENCH_TASKS
GENE_TASKS = frozenset(
    {
        "gwas_causal_gene_gwas_catalog",
        "gwas_causal_gene_opentargets",
        "screen_gene_retrieval",
    }
)
SUPPORTED_TASKS = LETTER_TASKS | GENE_TASKS | {"gwas_variant_prioritization"}

_SOLUTION_OPEN_RE = re.compile(r"<solution\b[^>]*>", re.IGNORECASE)
_SOLUTION_CLOSE_RE = re.compile(r"</\s*solution\s*>?", re.IGNORECASE)
_TRAILING_MARKUP_RE = re.compile(r"(?:\s*<\s*/?\s*(?:[A-Za-z_][\w-]*)?\s*/?\s*>?)+\s*$")
_FENCE_OPEN_RE = re.compile(r"^```(?:json|python|text)?\s*", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\s*```$")

_ANSWER_TAG_LETTER_RE = re.compile(r"\[ANSWER\]\s*([A-Za-z])\s*\[/ANSWER\]", re.IGNORECASE)
_ANSWER_TAG_ANY_RE = re.compile(r"\[ANSWER\]\s*(.*?)\s*\[/ANSWER\]", re.IGNORECASE | re.DOTALL)
_PROSE_LETTER_RES = (
    re.compile(r"\b(?:final\s+)?answer\s*(?:is|:)\s*(?:option\s+)?\(?([A-Za-z])\)?\b", re.IGNORECASE),
    re.compile(r"\boption\s+([A-Za-z])\s+is\s+(?:the\s+)?correct\s+answer\b", re.IGNORECASE),
)
_RSID_RE = re.compile(r"\brs\d+\b", re.IGNORECASE)
_GENE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FINAL_HEADING_RE = re.compile(r"(?im)^#{1,6}\s*(?:final\s+)?(?:conclusion|final answer|final diagnosis|diagnosis)\s*$")


# --------------------------------------------------------------------------- text


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(value)


def final_segment(response_text: Any) -> str:
    """Content of the LAST ``<solution>`` opening tag.

    A paired regex is wrong here: models emit two openings and one close after a
    stretch of meta-commentary ("... tag in my response. Since I've already
    completed the analysis ..."), and a paired match starts at the *first*
    opening and swallows the commentary, burying the committed answer.
    """
    text = _as_text(response_text).strip()
    opens = list(_SOLUTION_OPEN_RE.finditer(text))
    if opens:
        tail = text[opens[-1].end() :]
        close = _SOLUTION_CLOSE_RE.search(tail)
        text = (tail[: close.start()] if close else tail).strip()
    tagged = _ANSWER_TAG_ANY_RE.fullmatch(text)
    if tagged:
        text = tagged.group(1).strip()
    text = _FENCE_CLOSE_RE.sub("", _FENCE_OPEN_RE.sub("", text))
    # Shed stray/malformed markup the non-greedy capture left behind:
    # "<solution> FOLR1 </</solution>" and "<solution> E </execute></solution>".
    return _TRAILING_MARKUP_RE.sub("", text).strip()


def _final_section(text: str) -> str:
    sections = _FINAL_HEADING_RE.split(text)
    return sections[-1] if len(sections) > 1 else text[-2500:]


def _ident(value: Any) -> str:
    return str(value).strip().strip("\"'`.,;:()[]{}<>*_ ").upper()


def _unique(values) -> str | None:
    uniq = list(dict.fromkeys(values))
    return uniq[0] if len(uniq) == 1 else None


# ----------------------------------------------------------------- per-task rules


def _letter(segment: str) -> str | None:
    """Option letter, explicit commitment first.

    Upstream matched ``(?:answer|option|choice)...([A-F])`` before looking for
    LAB-Bench's own ``[ANSWER]X[/ANSWER]`` wrapper, so elimination prose
    ("3. Option B, despite having perfect gene matching, uses KpnI ...") beat an
    explicit ``[ANSWER]D[/ANSWER]`` on the next line. Precedence is inverted here.
    """
    tags = _ANSWER_TAG_LETTER_RE.findall(segment)
    if tags:
        letter = tags[-1].upper()
        return letter if re.fullmatch(r"[A-Z]", letter) else None
    token = _ident(segment)
    if re.fullmatch(r"[A-Z]", token):
        return token
    for pattern in _PROSE_LETTER_RES:
        hits = [h.upper() for h in pattern.findall(segment)]
        hits = [h for h in hits if re.fullmatch(r"[A-Z]", h)]
        if hits:
            return hits[-1]
    return None


def _rsid(segment: str) -> str | None:
    """Unique rsID, or abstain.

    Deliberately not ``variants[-1]``: that paid full reward for a ranked table
    of eleven rsIDs whose last row happened to be the label.
    """
    # findall already covers a bare rsID, so there is no reachable second
    # branch: if it matched nothing, no substring of the segment is an rsID.
    return _unique(m.lower() for m in _RSID_RE.findall(segment))


def candidate_genes(prompt: str, task_name: str) -> list[str]:
    """Closed answer space declared by the prompt."""
    if not prompt:
        return []
    if task_name.startswith("gwas_causal_gene"):
        locus = prompt.split("Genes in locus:", 1)[-1]
        return [v.strip() for v in re.findall(r"\{([^{}]+)\}", locus) if v.strip()]
    if task_name == "screen_gene_retrieval":
        match = re.search(r"Candidate genes:\s*([^\n]+)", prompt, re.IGNORECASE)
        return [v.strip() for v in match.group(1).split(",") if v.strip()] if match else []
    return []


def _candidate_pattern(candidate: str) -> str:
    """Whole-symbol match that still ends at a sentence period.

    The boundary must reject a name *continuing* -- RPL36A must not match inside
    RPL36A-HNRNPH2, and 30 of the 200 candidates in this benchmark are such
    prefix pairs (TRIM39/TRIM39-RPP21, ARPIN/ARPIN-AP3S2) -- while accepting the
    period that ends a sentence. A single "not followed by [A-Za-z0-9.-]" class
    conflates the two and dropped "the strongest effect is CNOT3."
    """
    return rf"(?<![A-Za-z0-9_-])(?<!\.)" rf"{re.escape(candidate)}" rf"(?![A-Za-z0-9_-])(?!\.[A-Za-z0-9])"


def _gene(segment: str, prompt: str, task_name: str) -> str | None:
    candidates = candidate_genes(prompt, task_name)
    token = _ident(segment)
    if _GENE_TOKEN_RE.fullmatch(token):
        if not candidates:
            return token
        for candidate in candidates:
            if token == candidate.upper():
                return candidate.upper()
    if not candidates:
        return None
    final = _final_section(segment)
    upper = {c.upper() for c in candidates}
    for pattern in (
        r"\banswer\b\s*(?:is|:)?\s*[*_`]*([A-Za-z0-9.-]+)",
        r"\bcausal\s+gene\b\s*(?:is|:)?\s*[*_`]*([A-Za-z0-9.-]+)",
        r"\bgene\s+symbol\b\s*(?:is|:)?\s*[*_`]*([A-Za-z0-9.-]+)",
    ):
        hits = [v.upper() for v in re.findall(pattern, final, re.IGNORECASE) if v.upper() in upper]
        if hits:
            return hits[-1]
    # A single candidate named in the closing section counts as a commitment;
    # several distinct ones do not. Upstream fell back to "last mention
    # anywhere", which is the same guess _rsid refuses.
    mentioned = [c.upper() for c in candidates if re.search(_candidate_pattern(c), final, re.IGNORECASE)]
    return _unique(mentioned)


# ------------------------------------------------------------------- public API


def extract(task_name: str, response_text: Any, prompt: str = "") -> str | None:
    """Canonical prediction for ``task_name``, or ``None`` if uncommitted."""
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task_name}")
    segment = final_segment(response_text)
    if task_name in LETTER_TASKS:
        return _letter(segment)
    if task_name == "gwas_variant_prioritization":
        return _rsid(segment)
    return _gene(segment, prompt, task_name)


def normalize_gold(task_name: str, gold: Any) -> str | None:
    """Canonical form of a stored label, in the same space as ``extract``."""
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task_name}")
    if task_name in LETTER_TASKS:
        letter = _ident(gold)
        return letter if re.fullmatch(r"[A-Z]", letter) else None
    if task_name == "gwas_variant_prioritization":
        return _ident(gold).lower() or None
    return _ident(gold) or None


def score(task_name: str, response_text: Any, gold: Any, prompt: str = "") -> float:
    prediction = extract(task_name, response_text, prompt)
    expected = normalize_gold(task_name, gold)
    return 1.0 if prediction is not None and expected is not None and prediction == expected else 0.0
