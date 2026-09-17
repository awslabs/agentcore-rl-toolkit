"""Put this recipe's own directory on ``sys.path`` so the tests under ``tests/`` can
import its scripts as top-level modules, the same way the scripts import each other.

``src`` goes on it too, for the two modules that live on both sides of the container
boundary: ``preprocess`` writes the setup script and ``swe_agent_server.evaluation`` undoes
what it did, and the tests that pin that contract need both. Importing the server package
here works only because the modules they reach are stdlib-only -- the rest of it needs the
image's venv, which is why those tests run there (see the README).
"""

import sys
from pathlib import Path

RECIPE_DIR = Path(__file__).resolve().parent
for path in (RECIPE_DIR, RECIPE_DIR / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
