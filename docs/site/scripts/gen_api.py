#!/usr/bin/env -S uv run --with "griffe>=0.47" --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["griffe>=0.47"]
# ///
"""Generate Starlight-flavored Markdown API reference from docstrings.

Walks `src/agentcore_rl_toolkit/` with griffe, emits one Markdown file per
module under `src/content/docs/api/`. Each file is committed to git so
reviewers can see API changes in the diff and the Astro build has no
Python dependency.

Run with:  pnpm gen:api

Output structure:
    api/
      core/
        app.md
        client.md
        reward.md
      backends/
        slime/
          runner.md
"""
from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import griffe

HERE = Path(__file__).resolve().parent
SITE_DIR = HERE.parent
REPO_ROOT = SITE_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
OUT_DIR = SITE_DIR / "src" / "content" / "docs" / "api"


@dataclass(frozen=True)
class ModuleSpec:
    """Specifies one or more modules to render into a single page.

    If more than one dotted_path is given, the page concatenates the
    rendered modules in order, using each path's trailing segment as
    an H2 section within the page.
    """

    dotted_paths: tuple[str, ...]  # one or more dotted module paths
    out_path: str  # relative to OUT_DIR, e.g. "backends/slime/runner.md"
    title: str
    description: str
    # Allowlist of top-level symbol names to emit. None = everything
    # public (leading-underscore names are always dropped). This is
    # how we keep internal helpers out of the rendered API reference.
    include: tuple[str, ...] | None = None


MODULES: list[ModuleSpec] = [
    # --- Core --- #
    # app: only AgentCoreRLApp. RolloutConfig is an internal dataclass.
    ModuleSpec(
        dotted_paths=("agentcore_rl_toolkit.app",),
        out_path="core/app.md",
        title="app",
        description="AgentCoreRLApp and the @rollout_entrypoint decorator.",
        include=("AgentCoreRLApp",),
    ),
    # client: the trainer/evaluator-side surface. Drop ACRRateLimiter
    # (internal throttling helper).
    ModuleSpec(
        dotted_paths=("agentcore_rl_toolkit.client",),
        out_path="core/client.md",
        title="client",
        description="RolloutClient, RolloutFuture, and the BatchResult family.",
        include=(
            "RolloutClient",
            "RolloutFuture",
            "BatchResult",
            "AsyncBatchResult",
            "BatchItem",
        ),
    ),
    # reward: just the ABC users subclass.
    ModuleSpec(
        dotted_paths=("agentcore_rl_toolkit.reward_function",),
        out_path="core/reward.md",
        title="reward",
        description="RewardFunction base class.",
        include=("RewardFunction",),
    ),
    # --- Backends › slime --- #
    # Only the public user surface: the SlimeRunner entry point.
    # integration/rollout.py and integration/rewards.py are load-bearing
    # plugin paths but users don't import them — SlimeRunner wires them
    # into the job.
    ModuleSpec(
        dotted_paths=("agentcore_rl_toolkit.backends.slime.runner",),
        out_path="backends/slime/runner.md",
        title="SlimeRunner",
        description="One Python entry point for slime-backed training.",
        include=("SlimeRunner",),
    ),
]


def _render_signature(obj: griffe.Function | griffe.Class) -> str:
    """Render a parameter list for a function/class; skip 'self'/'cls'."""
    params: list[str] = []
    for p in obj.parameters:
        if p.name in ("self", "cls"):
            continue
        piece = p.name
        if p.annotation is not None:
            piece += f": {p.annotation}"
        if p.default is not None:
            piece += f" = {p.default}"
        params.append(piece)
    return ", ".join(params)


_SECTION_COMMENT = re.compile(r"^\s*#\s*---\s*(.+?)\s*---\s*$")
_FIELD_LINE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*[:=]")


def _field_sections(source_file: str | None) -> dict[str, str]:
    """Scan a dataclass's source file for `# --- Section ---` markers
    and return a {first-field-in-section: section-header} mapping.

    The renderer uses this to interleave section comments between
    parameter lines on multi-line constructor signatures.
    """
    if not source_file:
        return {}
    try:
        with open(source_file) as f:
            lines = f.readlines()
    except OSError:
        return {}

    mapping: dict[str, str] = {}
    pending: str | None = None
    for line in lines:
        m = _SECTION_COMMENT.match(line)
        if m:
            pending = m.group(1).strip()
            continue
        fm = _FIELD_LINE.match(line)
        if fm and pending is not None:
            mapping[fm.group(1)] = pending
            pending = None
    return mapping


def _render_call_multiline(name: str, obj: griffe.Function | griffe.Class) -> str:
    """Render `name(arg1, arg2, ...)` with one parameter per line.

    If the source class carries `# --- Section ---` dataclass-field
    comments, those are emitted as interleaved `# Section` comments
    between the corresponding parameter groups.
    """
    # Best-effort: extract section comments from the enclosing class
    # source file. For a `__init__` inside a dataclass, griffe attaches
    # the dataclass's source path to the enclosing object.
    source_file: str | None = None
    parent = getattr(obj, "parent", None)
    if parent is not None:
        fp = getattr(parent, "filepath", None)
        if fp is not None:
            source_file = str(fp)
    sections = _field_sections(source_file)

    lines: list[str] = []
    for p in obj.parameters:
        if p.name in ("self", "cls"):
            continue
        if p.name in sections:
            header = sections[p.name]
            # Blank line before each section except the first.
            if lines:
                lines.append("")
            lines.append(f"    # --- {header} ---")
        piece = p.name
        if p.annotation is not None:
            piece += f": {p.annotation}"
        if p.default is not None:
            piece += f" = {p.default}"
        lines.append(f"    {piece},")

    if not lines:
        return f"{name}()"
    body = "\n".join(lines)
    return f"{name}(\n{body}\n)"


def _fence(text: str, lang: str = "python") -> str:
    return f"```{lang}\n{text}\n```"


def _render_docstring(doc: griffe.Docstring | None) -> str:
    """Render a griffe Docstring as Markdown. Google-style sections
    (Args, Returns, Raises, Yields, Example) get turned into Markdown
    sub-sections; plain prose is passed through verbatim.
    """
    if doc is None:
        return ""
    out: list[str] = []
    try:
        parsed = doc.parse("google")
    except Exception:
        # Fall back to raw value if parsing fails — still renders fine.
        return doc.value.strip()

    for section in parsed:
        kind = str(section.kind)
        if kind.endswith("text"):
            out.append(section.value.strip())
        elif kind.endswith("parameters"):
            out.append("**Parameters**")
            for item in section.value:
                ann = f" *({item.annotation})*" if item.annotation else ""
                default = f" — default `{item.default}`" if item.default else ""
                desc = item.description.strip() if item.description else ""
                out.append(f"- `{item.name}`{ann}{default}: {desc}")
        elif kind.endswith("returns"):
            out.append("**Returns**")
            for item in section.value:
                ann = f" *({item.annotation})*" if item.annotation else ""
                desc = item.description.strip() if item.description else ""
                out.append(f"- {item.name or ''}{ann}: {desc}".lstrip())
        elif kind.endswith("raises"):
            out.append("**Raises**")
            for item in section.value:
                ann = f" *({item.annotation})*" if item.annotation else ""
                desc = item.description.strip() if item.description else ""
                out.append(f"- {ann}: {desc}".lstrip())
        elif kind.endswith("examples"):
            for ex in section.value:
                # ex is (kind, text) for parsed examples
                if isinstance(ex, tuple) and len(ex) == 2:
                    _, text = ex
                    out.append(_fence(text.strip()))
                else:
                    out.append(str(ex))
        else:
            # Fall back: dump whatever value the section holds.
            val = getattr(section, "value", "")
            if isinstance(val, str):
                out.append(val.strip())

    return "\n\n".join(s for s in out if s)


def _is_public(name: str) -> bool:
    return not name.startswith("_") or name in ("__init__", "__call__")


def _render_function(fn: griffe.Function, *, heading: str = "###") -> str:
    sig = _render_signature(fn)
    returns = ""
    if fn.returns:
        returns = f" -> {fn.returns}"
    doc = _render_docstring(fn.docstring)
    pieces = [f"{heading} `{fn.name}({sig}){returns}`"]
    if doc:
        pieces.append(doc)
    return "\n\n".join(pieces)


def _render_class(cls: griffe.Class) -> str:
    base_names = [str(b) for b in cls.bases] if cls.bases else []
    bases = f"({', '.join(base_names)})" if base_names else ""
    pieces: list[str] = [f"## `class {cls.name}{bases}`"]
    doc = _render_docstring(cls.docstring)
    if doc:
        pieces.append(doc)

    # __init__ signature if present.
    init = cls.members.get("__init__")
    if isinstance(init, griffe.Function):
        call = _render_call_multiline(cls.name, init)
        pieces.append(f"**Constructor**\n\n{_fence(call)}")
        init_doc = _render_docstring(init.docstring)
        if init_doc:
            pieces.append(init_doc)

    # Public methods.
    methods: list[griffe.Function] = []
    for name, member in cls.members.items():
        if name in ("__init__",):
            continue
        if not _is_public(name):
            continue
        if isinstance(member, griffe.Function):
            methods.append(member)

    if methods:
        pieces.append("### Methods")
        for m in sorted(methods, key=lambda x: x.name):
            pieces.append(_render_function(m, heading="####"))

    # Public attributes (dataclass fields, class vars with annotations).
    attrs = [
        (name, member)
        for name, member in cls.members.items()
        if _is_public(name) and isinstance(member, griffe.Attribute)
    ]
    if attrs:
        pieces.append("### Attributes")
        for name, attr in sorted(attrs, key=lambda x: x[0]):
            ann = f" *({attr.annotation})*" if attr.annotation else ""
            desc = _render_docstring(attr.docstring)
            pieces.append(f"- `{name}`{ann}{' — ' + desc if desc else ''}")

    return "\n\n".join(pieces)


def _render_module_body(
    mod: griffe.Module,
    include: tuple[str, ...] | None = None,
) -> str:
    """Render the body of a module (no frontmatter). Can be
    concatenated with other module bodies onto a single page."""
    pieces: list[str] = []
    mod_doc = _render_docstring(mod.docstring)
    if mod_doc:
        pieces.append(mod_doc)

    # Collect classes and functions. If `include` is set, only emit
    # those top-level names (in declaration order). Otherwise emit
    # every public member.
    allowed = set(include) if include is not None else None
    classes: list[griffe.Class] = []
    functions: list[griffe.Function] = []
    for name, member in mod.members.items():
        if allowed is not None:
            if name not in allowed:
                continue
        elif not _is_public(name):
            continue
        if isinstance(member, griffe.Class):
            classes.append(member)
        elif isinstance(member, griffe.Function):
            functions.append(member)

    if allowed is not None:
        order = {name: i for i, name in enumerate(include or ())}
        classes.sort(key=lambda c: order.get(c.name, 0))
        functions.sort(key=lambda f: order.get(f.name, 0))
    else:
        functions.sort(key=lambda f: f.name)

    for cls in classes:
        pieces.append(_render_class(cls))

    if functions:
        if len(functions) > 1 and not classes:
            pieces.append("## Functions")
        for fn in functions:
            pieces.append(
                _render_function(
                    fn,
                    heading="##" if not classes and len(functions) == 1 else "###",
                )
            )

    return "\n\n".join(pieces)


def _render_frontmatter(title: str, description: str, order: int) -> str:
    return dedent(
        f"""\
        ---
        title: {title}
        description: {description}
        sidebar:
          order: {order}
        ---
        """
    )


def main() -> int:
    if not SRC_DIR.is_dir():
        print(f"[gen:api] missing src dir: {SRC_DIR}", file=sys.stderr)
        return 1

    # Reset generated subtrees so deletions in source propagate. Preserve
    # hand-written index.md at the root.
    for subdir in ("core", "backends"):
        target = OUT_DIR / subdir
        if target.exists():
            shutil.rmtree(target)

    loader = griffe.GriffeLoader(search_paths=[str(SRC_DIR)])
    pkg = loader.load("agentcore_rl_toolkit")

    written = 0
    for idx, spec in enumerate(MODULES):
        sections: list[str] = [_render_frontmatter(spec.title, spec.description, idx)]
        multiple = len(spec.dotted_paths) > 1

        for dotted in spec.dotted_paths:
            parts = dotted.split(".")[1:]
            mod: griffe.Object = pkg
            for p in parts:
                mod = mod.members[p]
            if not isinstance(mod, griffe.Module):
                print(f"[gen:api] {dotted} resolved to non-module; skipping")
                continue
            # Anchor each rendered module with its full dotted import
            # path. On multi-module pages this doubles as the section
            # separator; on single-module pages it tells the reader
            # exactly what to import.
            if multiple:
                sections.append(f"## `{dotted}`")
            else:
                sections.append(f"_module: `{dotted}`_")
            sections.append(_render_module_body(mod, spec.include))

        text = "\n\n".join(s for s in sections if s).rstrip() + "\n"
        out_file = OUT_DIR / spec.out_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(text)
        written += 1
        print(f"[gen:api] wrote {out_file.relative_to(SITE_DIR)}")

    print(f"[gen:api] done ({written} page(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
