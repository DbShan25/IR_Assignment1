#!/usr/bin/env python3
"""Verify all required project files exist before running the app."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

REQUIRED = [
    ROOT / "app.py",
    ROOT / "requirements.txt",
    ROOT / "data" / "20newsgroups_subset.csv",
    SRC / "__init__.py",
    SRC / "indexing.py",
    SRC / "preprocessing.py",
    SRC / "datasets.py",
    SRC / "trees.py",
    SRC / "tolerant_retrieval.py",
    SRC / "path_setup.py",
]


def main() -> int:
    print(f"Project root: {ROOT}\n")
    missing = [p for p in REQUIRED if not p.exists()]
    if missing:
        print("MISSING FILES:")
        for p in missing:
            print(f"  - {p.relative_to(ROOT)}")
        print("\nRe-clone the full repo: https://github.com/bindlish04/IR_Assignment1")
        return 1
    print("All required files present.")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(SRC))
    try:
        import indexing  # noqa: F401
        import preprocessing  # noqa: F401
        print("Imports OK.")
    except ImportError as e:
        print(f"Import failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
