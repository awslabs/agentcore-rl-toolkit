"""HERB's own code, loaded from the downloaded repository (scripts/fetch_herb.py; HERB_DIR) at run time, with three
edits made in memory:

- `from utils import get_gpt4_response` -> `helper_llm` (Claude Haiku 5.5 on Bedrock in place of GPT-4o);
- the relative data paths -> HERB_DIR/data;
- a fix in `role_to_id_search`, which formats its prompt with `question=` while the template expects `{text}`, so
  every call raises KeyError.

The system prompt is react.py's CONTEXT_AGENT, and `retrieve_all`'s description is react.py's. The unanswerable
judge is evaluate.py's `unanswerability_eval` with one added rule: an answer concluding that nothing matching the
question exists does not answer it.
"""

import os
import sys
import types
from functools import cache
from pathlib import Path

HERB_DIR = Path(os.environ.get("HERB_DIR", Path(__file__).parent / "herb"))
RULE_B = (
    '- Reply "No" also if the answer concludes that nothing matching the question exists (for example, '
    '"No UX Researchers reviewed the document" or "no such PRs were reverted"): stating that the requested '
    "items do not exist is not providing them.\n"
)


def _load(name: str, source: str, edits: list[tuple[str, str]]) -> types.ModuleType:
    s = (HERB_DIR / "code" / source).read_text()
    s = s.replace('"../data/metadata/', f'"{HERB_DIR}/data/metadata/').replace(
        '"../data/products"', f'"{HERB_DIR}/data/products"'
    )
    if "../data" in s:
        raise RuntimeError(f"{source}: a relative data path is left")
    for old, new in [("from utils import get_gpt4_response", "from helper_llm import get_gpt4_response"), *edits]:
        if old not in s:
            raise RuntimeError(f"{source}: expected text not found: {old!r}")
        s = s.replace(old, new)
    module = types.ModuleType(name)
    module.__file__ = str(HERB_DIR / "code" / source)
    sys.modules[name] = module
    exec(compile(s, module.__file__, "exec"), module.__dict__)  # noqa: S102 - the pinned, sha256-checked upstream code
    return module


@cache
def metadata_tools() -> types.ModuleType:
    """The 8 structured tools (llama_index FunctionTool objects)."""
    fix = (".format(roles=list(all_roles), question=text)", ".format(roles=list(all_roles), text=text)")
    return _load("herb_metadata_tools", "metadata_search_functions.py", [fix])


@cache
def evaluation() -> types.ModuleType:
    """The official evaluation (answer_likert_score, answer_to_*, f1_score_sets), and judge B."""
    rule = (
        "even if it explicitly states that the information is not available.\n",
        "even if it explicitly states that the information is not available.\n" + RULE_B,
    )
    return _load("herb_eval", "evaluate.py", [rule])


@cache
def system_prompt() -> str:
    src = (HERB_DIR / "code" / "react.py").read_text()
    start = src.index('CONTEXT_AGENT = """') + len('CONTEXT_AGENT = """')
    return src[start : src.index('"""', start)]


@cache
def retrieve_all_description() -> str:
    src = (HERB_DIR / "code" / "react.py").read_text()
    start = src.index('description="', src.index('name="retrieve_all"')) + len('description="')
    return src[start : src.index('"', start)]
