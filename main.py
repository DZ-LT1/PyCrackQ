"""Launch PyCrackQ from a source checkout."""

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from PyCrackQ.main import main


if __name__ == "__main__":
    main()
