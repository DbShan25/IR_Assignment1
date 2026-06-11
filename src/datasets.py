"""Load public datasets for the IR assignment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Bundled subset of the 20 Newsgroups corpus (see data/DATASET.md).
DEFAULT_CSV = "20newsgroups_subset.csv"

# Categories used when building or refreshing the bundled file.
DEFAULT_CATEGORIES = [
    "rec.sport.baseball",
    "sci.med",
    "talk.politics.misc",
]

DATASET_INFO = {
    "name": "20 Newsgroups (subset)",
    "source": "https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-dataset",
    "license": "Public domain / custom (see original 20 Newsgroups README)",
    "description": (
        "Classic text classification corpus: Usenet posts from 20 discussion groups. "
        "This project uses a fixed subset of three categories bundled as CSV for offline use."
    ),
}


def data_dir(root: Path | None = None) -> Path:
    return (root or Path(__file__).resolve().parents[1]) / "data"


def csv_path(root: Path | None = None) -> Path:
    return data_dir(root) / DEFAULT_CSV


def load_from_csv(path: Path | None = None) -> tuple[dict[str, str], pd.DataFrame]:
    """Return doc_id -> text and full metadata dataframe."""
    path = path or csv_path()
    df = pd.read_csv(path)
    df = df.dropna(subset=["text"] if "text" in df.columns else [df.columns[-1]])
    id_col = "doc_id" if "doc_id" in df.columns else df.columns[0]
    text_col = "text" if "text" in df.columns else df.columns[-1]
    docs = {str(row[id_col]): str(row[text_col]) for _, row in df.iterrows()}
    return docs, df


def fetch_20newsgroups_subset(
    categories: list[str] | None = None,
    max_per_category: int = 20,
    root: Path | None = None,
    *,
    allow_insecure_ssl: bool = False,
) -> pd.DataFrame:
    """
    Download 20 Newsgroups via scikit-learn and return a tabular subset.
    Requires network on first fetch (cached by sklearn under ~/scikit_learn_data).
    """
    if allow_insecure_ssl:
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context

    from sklearn.datasets import fetch_20newsgroups

    categories = categories or DEFAULT_CATEGORIES
    bunch = fetch_20newsgroups(
        subset="train",
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=("headers", "footers", "quotes"),
    )

    rows: list[dict[str, str]] = []
    counts: dict[str, int] = {c: 0 for c in categories}
    cat_names = bunch.target_names

    for text, target in zip(bunch.data, bunch.target):
        category = cat_names[target]
        if counts[category] >= max_per_category:
            continue
        doc_id = f"{category.replace('.', '_')}_{counts[category] + 1:03d}"
        rows.append(
            {
                "doc_id": doc_id,
                "category": category,
                "text": text.strip(),
            }
        )
        counts[category] += 1
        if all(counts[c] >= max_per_category for c in categories):
            break

    return pd.DataFrame(rows)


def save_subset_csv(df: pd.DataFrame, root: Path | None = None) -> Path:
    out = csv_path(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def ensure_bundled_dataset(root: Path | None = None) -> Path:
    """Use bundled CSV if present; otherwise download and save."""
    path = csv_path(root)
    if path.exists():
        return path
    df = fetch_20newsgroups_subset(root=root)
    return save_subset_csv(df, root=root)


def load_default_documents(root: Path | None = None) -> tuple[dict[str, str], pd.DataFrame | None]:
    path = csv_path(root)
    if path.exists():
        return load_from_csv(path)
    try:
        df = fetch_20newsgroups_subset(root=root, allow_insecure_ssl=True)
        save_subset_csv(df, root=root)
        return {str(r.doc_id): str(r.text) for r in df.itertuples()}, df
    except Exception:
        # Minimal fallback if offline and CSV missing
        fallback = data_dir(root) / "sample_documents.txt"
        docs: dict[str, str] = {}
        if fallback.exists():
            for line in fallback.read_text(encoding="utf-8").strip().splitlines():
                if "|" in line:
                    doc_id, text = line.split("|", 1)
                    docs[doc_id.strip()] = text.strip()
        return docs, None
