"""Put this recipe's own directory on ``sys.path`` for the tests under ``tests/``.

The scripts here are entrypoints run from this directory -- ``./deploy.py``,
``./evaluate.py``, ``./analyze.py``, ``./preprocess.py`` -- so they import their
siblings as top-level modules (``from config import ...``, ``import iam_policy``).
pytest runs from the repo root instead and prepends only the test file's own directory
(``tests/``), which does not contain them.

Adding the recipe root here rather than rewriting the tests to some other name keeps
one import identity for these modules: exactly what the scripts use. There is no
package to import them as -- ``examples/`` is not importable, and this directory is a
recipe rather than a distribution.
"""

import sys
from pathlib import Path

RECIPE_DIR = Path(__file__).resolve().parent
if str(RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(RECIPE_DIR))
