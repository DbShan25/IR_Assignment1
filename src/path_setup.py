"""Add project paths and verify required modules exist before import."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

REQUIRED_SRC_FILES = (
    "indexing.py",
    "preprocessing.py",
    "datasets.py",
    "trees.py",
    "tolerant_retrieval.py",
    "__init__.py",
)


def setup_paths() -> Path:
    """Ensure project root and src/ are on sys.path. Return project root."""
    for folder in (ROOT, SRC):
        path_str = str(folder)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return ROOT


def verify_project_files() -> None:
    missing = [name for name in REQUIRED_SRC_FILES if not (SRC / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Incomplete project copy. Missing in src/: "
            + ", ".join(missing)
            + f"\n\nExpected folder layout:\n  {ROOT}/app.py\n  {SRC}/indexing.py\n  ..."
            + "\n\nFix: clone the full repository or re-download the zip, then run:\n"
            "  cd "
            + str(ROOT.name)
            + "\n  streamlit run app.py"
        )
