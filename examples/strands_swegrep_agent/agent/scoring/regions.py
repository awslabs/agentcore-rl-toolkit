"""Code regions: parsing the agent's answer, checking it against the checkout, and spotting test files.

A region is ``path:start-end``, a repo-relative path with a 1-indexed, inclusive line range. A bare
``path:line`` is read as a one-line region.
"""

import re
from dataclasses import dataclass
from pathlib import Path

_REGION = re.compile(r"^([^\s:]+):(\d+)(?:-(\d+))?$")

# Test files and directories, used to drop tests from the training labels (data/build_training_data.py).
# Changing it changes the training data and its sha256 in data/manifests/expected.json.
_TEST_PATH = re.compile(
    r"""
    # Any language: a test directory anywhere in the path (case-insensitive)
      (?i:(^|/)(tests?|__tests__|testing|testdata|testfixtures?|test_?utils|specs?)/)
    # Python and Rust: tests.py, test.rs, test_foo.py, conftest.py
    | (^|/)tests?\.(py|rs)$
    | (^|/)test_helpers?\.\w+$
    | (^|/)test_[^/]+\.py$
    # Go, Rust, C/C++, Ruby, Elixir, Python: foo_test.go, foo_test.cc, ...
    | _test\.(py|go|rs|cc|cpp|c|rb|exs?)$
    | (^|/)conftest\.py$
    # JVM, .NET, Swift, PHP: FooTest.java, FooTests.cs, TestFoo.kt
    | [a-z0-9_]Tests?\.(java|kt|scala|cs|swift|php)$
    | (^|/)Tests?[A-Z][^/]*\.(java|kt|cs)$
    # Ruby: foo_spec.rb
    | _spec\.rb$
    # JavaScript / TypeScript: foo.test.ts, foo.spec.jsx, foo.test.mjs
    | \.(test|spec)\.[cm]?[jt]sx?$
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Region:
    path: str
    start_line: int
    end_line: int


def is_test_path(path: str) -> bool:
    # lstrip(".") also strips "../" and ".tests/"; the published labels were built this way.
    return bool(_TEST_PATH.search(path.replace("\\", "/").lstrip("./")))


def parse_regions(block: str) -> tuple[list[Region], list[str]]:
    """The regions in an answer block, in order, and the lines that are not valid regions."""
    regions, invalid = [], []
    for raw in block.splitlines():
        match = _REGION.fullmatch(raw.strip())
        if not match:
            if raw.strip():
                invalid.append(raw)
            continue
        path, start, end = match[1], int(match[2]), int(match[3] or match[2])
        if 1 <= start <= end:
            regions.append(Region(path, start, end))
        else:
            invalid.append(raw)
    return regions, invalid


def validate_regions(regions: list[Region], repo_root: str | Path) -> tuple[list[Region], list[str]]:
    """Drop regions whose file is missing or outside the checkout, and clip ends to the file's length."""
    root = Path(repo_root).resolve()
    valid, invalid = [], []
    line_counts = {}
    for region in regions:
        try:
            path = (root / region.path).resolve()
            if path.is_relative_to(root) and path.is_file():
                if path not in line_counts:
                    with path.open("rb") as f:
                        line_counts[path] = sum(1 for _ in f)
                end = min(region.end_line, line_counts[path])
                if region.start_line <= end:
                    valid.append(Region(path.relative_to(root).as_posix(), region.start_line, end))
                    continue
        except (OSError, ValueError, RuntimeError):  # unreadable, a NUL in the path, a symlink loop
            pass
        invalid.append(f"{region.path}:{region.start_line}-{region.end_line}")
    return valid, invalid
