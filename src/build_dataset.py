#!/usr/bin/env python3
"""Download 20 Newsgroups subset and save to data/20newsgroups_subset.csv."""

from __future__ import annotations

import ssl
import sys
from pathlib import Path

# macOS Python installs often lack system CA certs; allow one-off dataset download.
ssl._create_default_https_context = ssl._create_unverified_context

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import (  # noqa: E402
    DEFAULT_CATEGORIES,
    fetch_20newsgroups_subset,
    save_subset_csv,
)


def main() -> None:
    print("Fetching 20 Newsgroups subset...")
    print("Categories:", ", ".join(DEFAULT_CATEGORIES))
    df = fetch_20newsgroups_subset(max_per_category=20)
    path = save_subset_csv(df, root=ROOT)
    print(f"Saved {len(df)} documents to {path}")
    print(df["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
