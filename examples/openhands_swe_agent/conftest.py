"""Put this recipe's own directory on ``sys.path`` so the tests under ``tests/`` can
import its scripts as top-level modules, the same way the scripts import each other.
"""

import sys
from pathlib import Path

RECIPE_DIR = Path(__file__).resolve().parent
if str(RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(RECIPE_DIR))
