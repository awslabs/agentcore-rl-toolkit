"""Put this example's own directory on ``sys.path`` so the tests under ``tests/`` can
import its modules as top-level modules, the same way the modules import each other.
"""

import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))
