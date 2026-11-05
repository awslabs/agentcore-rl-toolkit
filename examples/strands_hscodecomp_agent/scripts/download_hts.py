"""Download the HTS 2025 Revision 15 tables of the official HSCodeComp repository (pinned, sha256-checked).

    python scripts/download_hts.py DATA_DIR
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = (
    "https://raw.githubusercontent.com/AIDC-AI/Marco-Search-Agent/2be90e40519f67916b4d7a621491e04efac0b6de/"
    "Marco-DeepResearch-Family/HSCodeComp/code/data/"
)
FILES = {
    "hts_tree.json": "02505dbf9a3bd2dde04ec02b3bb9a7f0a5b81db5b4ca17df44bfe7b587b6f97f",
    "hts_2025_revision_15.json": "83e5fc468a032b72c7f3759add0290b6bd2ba53fca40f73a5da0c59f9511a444",
}


def main(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, sha256 in FILES.items():
        data = urllib.request.urlopen(BASE + name, timeout=120).read()
        if hashlib.sha256(data).hexdigest() != sha256:
            sys.exit(f"{name}: sha256 mismatch")
        (data_dir / name).write_bytes(data)
        print(f"{name}: {len(data)} bytes")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
