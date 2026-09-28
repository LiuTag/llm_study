"""Make sibling source modules importable for pytest and direct test execution."""

import sys
from pathlib import Path


SOURCE_DIR = str(Path(__file__).resolve().parent.parent)
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)
