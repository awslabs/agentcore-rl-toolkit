"""The agent's grep, glob and read tools, bound to one checkout. Their docstrings are the tool
descriptions the model sees, and their limits are the ones the published model was trained with."""

import re
import subprocess
from collections import defaultdict
from pathlib import Path

from strands import tool

GREP_MAX_MATCHES = 120  # lines per grep page
GREP_MAX_PER_FILE = 10  # so one file can't fill the page
GREP_MAX_ALTERNATIVES = 8  # long chains of guessed identifiers never match and waste a turn
GREP_MAX_PATTERN_CHARS = 120
GLOB_MAX_RESULTS = 200
READ_MAX_LINES = 800
MAX_CHARS = 8000  # per tool result

# Never searched: version control, dependencies, build output and caches.
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".nox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".eggs", "site-packages", "dist", "build", "cdk.out", ".idea", ".vscode",
}  # fmt: skip


def top_level_listing(repo_root: str | Path, limit: int = 60) -> str:
    """The checkout's top-level entries (directories end in "/"), for the system prompt."""
    try:
        entries = sorted(e.name + "/" if e.is_dir() else e.name for e in Path(repo_root).iterdir())
    except OSError:
        return ""
    if len(entries) > limit:
        entries = entries[:limit] + [f"... (+{len(entries) - limit} more)"]
    return "\n".join(entries)


def _gitignored_dirs(root: Path) -> set[str]:
    """Plain names (no "/" inside, no wildcards) in the root .gitignore, skipped at any depth."""
    try:
        lines = (root / ".gitignore").read_text(errors="replace").splitlines()
    except OSError:
        return set()
    names = {line.strip().strip("/") for line in lines if not line.strip().startswith(("#", "!"))}
    return {name for name in names if name and "/" not in name and not any(c in name for c in "*?[]")}


def _resolve(root: Path, rel: str) -> Path | None:
    """``rel`` under ``root``, or None if it would leave the checkout."""
    root = root.resolve()
    target = (root / str(rel or ".").lstrip("/")).resolve()
    return target if target.is_relative_to(root) else None


def _page(lines: list[str], offset: int, max_lines: int) -> str:
    """One page of results that fits in MAX_CHARS, ending with how to get the next page."""
    if offset < 0:
        return "error: offset must be non-negative."
    if offset >= len(lines):
        return f"No more results at offset {offset} (total {len(lines)})."
    budget = MAX_CHARS - 160  # room for the next-page note
    page, used = [], 0
    for line in lines[offset : offset + max_lines]:
        if page and used + len(line) + 1 > budget:
            break
        if len(line) > budget:  # shorten the text, keeping a grep hit's "path:line:" prefix
            prefix = ":".join(line.split(":", 2)[:2]) + ":" if ":" in line else line
            line = line[: max(len(prefix), budget - 45)] + " ... [match text shortened; use read]"
        page.append(line)
        used += len(line) + 1
    next_offset = offset + len(page)
    if next_offset < len(lines):
        page.append(
            f"... (more results: next_offset={next_offset}; "
            f"repeat the same query with offset={next_offset}, or narrow the query)"
        )
    return "\n".join(page)


def _round_robin_by_file(lines: list[str]) -> list[str]:
    """Interleave grep hits across files, so the first page covers many files."""
    by_file = defaultdict(list)
    for line in lines:
        by_file[line.split(":", 1)[0]].append(line)
    depth = max(map(len, by_file.values()), default=0)
    return [by_file[f][i] for i in range(depth) for f in sorted(by_file) if i < len(by_file[f])]


def build_repo_tools(repo_root: str | Path) -> list:
    root = Path(repo_root)
    skip_dirs = SKIP_DIRS | _gitignored_dirs(root)
    exclude = [f"--exclude-dir={d}" for d in sorted(skip_dirs)]

    @tool
    def grep(pattern: str = "", path: str = ".", offset: int = 0) -> str:
        """Search the repository text with a regular expression (POSIX ERE).
        Returns matching `path:line: text` lines, repo-root-relative (at most a
        few matches per file — read the file for full context).

        Args:
            pattern: Regex to search for (e.g. "def resolve_url" or "TimeoutError").
            path: Optional repo-root-relative subtree to search (default: whole repo).
            offset: Result offset; use next_offset from a previous page with the same query.
        """
        if not pattern.strip():
            return "error: provide a non-empty pattern."
        alternatives = len(re.findall(r"(?<!\\)\|", pattern)) + 1
        if len(pattern) > GREP_MAX_PATTERN_CHARS:
            reason = f"pattern is {len(pattern)} chars (max {GREP_MAX_PATTERN_CHARS})"
        elif alternatives > GREP_MAX_ALTERNATIVES:
            reason = f"{alternatives} alternatives (max {GREP_MAX_ALTERNATIVES})"
        else:
            reason = None
        if reason:
            return f"error: pattern too broad ({reason}). Grep ONE concrete symbol or message per call."
        target = _resolve(root, path)
        if target is None or not target.exists():
            return f"error: path not found in repo: {path}"
        rel = str(target.relative_to(root.resolve())) if target != root.resolve() else "."
        command = ["grep", "-rnHIE", "-m", str(GREP_MAX_PER_FILE), *exclude, "--", pattern, rel]
        try:
            proc = subprocess.run(command, cwd=root, capture_output=True, text=True, errors="replace", timeout=60)
        except subprocess.TimeoutExpired:
            return f"error: grep timed out (60s) for /{pattern}/. Narrow the pattern or path."
        if proc.returncode > 1:  # 1 means no matches
            first_line = (proc.stderr.strip().splitlines() or ["unknown error"])[0]
            return f"error: grep failed: {first_line}"
        hits = [line.removeprefix("./") for line in proc.stdout.replace("\r", "").strip().splitlines()]
        if not hits:
            return f"No matches for /{pattern}/. Try a different pattern."
        return _page(_round_robin_by_file(hits), int(offset), GREP_MAX_MATCHES)

    @tool
    def glob(pattern: str, offset: int = 0) -> str:
        """Discover files whose path matches a glob pattern (e.g. "**/*router*.py",
        "src/**/*.py"). Returns repo-root-relative paths.

        Args:
            pattern: Glob pattern, repo-root-relative.
            offset: Result offset; use next_offset from a previous page with the same query.
        """
        pattern = str(pattern or "").strip()
        if not pattern or ".." in Path(pattern).parts:
            return "error: provide a valid repo-root-relative glob (no '..')."
        paths = set()
        for match in root.glob(pattern.lstrip("/")):
            rel = str(match.relative_to(root))
            if match.is_file() and _resolve(root, rel) and skip_dirs.isdisjoint(Path(rel).parts[:-1]):
                paths.add(rel)
        if not paths:
            return f"No files match {pattern!r}."
        return _page(sorted(paths), int(offset), GLOB_MAX_RESULTS)

    @tool
    def read(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a file's contents with line numbers.

        Args:
            path: Repo-root-relative file path (e.g. "src/router.py").
            start_line: 1-based first line to return (default 1).
            end_line: 1-based last line. Reads have line and character limits;
                a wide window is truncated
                (with a "... (truncated)" marker); page with start_line to see more.
        """
        target = _resolve(root, path)
        if target is None or not target.exists():
            return f"error: file not found: {path}"
        if not target.is_file():
            return f"error: not a file: {path}"
        data = target.read_bytes()
        if b"\0" in data[:8192]:  # a NUL in the first 8 KiB means binary, roughly how git and grep -I decide
            return f"error: binary file: {path}"
        # Line numbers must match grep -n and the gold labels, so split on "\n" only (str.splitlines also
        # splits on \f, \x85, ...). Trailing whitespace is dropped, as in the reader the model trained with.
        text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        lines = text.rstrip().split("\n") if text.strip() else []
        start = max(1, int(start_line))
        stop = int(end_line) if end_line else start + READ_MAX_LINES - 1
        if stop < start:
            return f"error: end_line {stop} is before start_line {start}."
        stop = min(stop, start + READ_MAX_LINES - 1, len(lines))
        if start > len(lines):
            return f"error: start_line {start} beyond end of file ({len(lines)} lines)."
        out = f"# {path} (lines {start}-{stop} of {len(lines)})\n" + "\n".join(
            f"{i}\t{lines[i - 1]}" for i in range(start, stop + 1)
        )
        return out[:MAX_CHARS] + "\n... (truncated)" if len(out) > MAX_CHARS else out

    return [grep, glob, read]
