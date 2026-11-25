"""Where the tool service finds the environment data (prepare_data.py) and its shared rate-limit stamps.

BIOMNI_DATA_DIR overrides the data location; the rate-limit stamps coordinate online requests across every
tool-service process on the node.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

DATA = Path(os.environ.get("BIOMNI_DATA_DIR") or Path(__file__).resolve().parents[2] / "data")
RATE_DIR = Path(tempfile.gettempdir()) / "biomni_research_agent_rate"
