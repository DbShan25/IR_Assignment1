"""Text preprocessing utilities for the IR assignment."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable

import nltk

_nltk_ready = False
_USE_NLTK_TOKENIZER = False
_USE_NLTK_STOPWORDS = False
_USE_NLTK_STEM = False

_BUILTIN_STOPS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "as",
    "by", "with", "from", "is", "are", "was", "were", "be", "been", "being", "that",
    "this", "these", "those", "it", "its", "can", "may", "such", "their", "they",
}


def ensure_nltk() -> None:
    global _nltk_ready, _USE_NLTK_TOKENIZER, _USE_NLTK_STOPWORDS, _USE_NLTK_STEM
    if _nltk_ready:
        return
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.data.find(f"corpora/{pkg}")
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
        _USE_NLTK_TOKENIZER = True
    except LookupError:
        try:
            nltk.data.find("tokenizers/punkt/english.pickle")
            _USE_NLTK_TOKENIZER = True
        except LookupError:
            _USE_NLTK_TOKENIZER = False
    try:
        nltk.data.find("corpora/stopwords")
        _USE_NLTK_STOPWORDS = True
    except LookupError:
        _USE_NLTK_STOPWORDS = False
    try:
        nltk.data.find("corpora/wordnet")
        _USE_NLTK_STEM = True
    except LookupError:
        _USE_NLTK_STEM = False
    _nltk_ready = True


def tokenize(text: str) -> list[str]:
    ensure_nltk()
    if _USE_NLTK_TOKENIZER:
        from nltk.tokenize import word_tokenize

        return word_tokenize(text)
    return re.findall(r"\b\w+(?:'\w+)?\b", text)


def handle_hyphens(text: str) -> str:
    """Merge hyphenated line-breaks and normalize hyphenated compounds."""
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    text = re.sub(r"(\w+)-(\w+)", r"\1\2", text)
    return text


def lowercase_tokens(tokens: list[str]) -> list[str]:
    return [t.lower() for t in tokens]


def remove_stopwords(tokens: list[str]) -> list[str]:
    ensure_nltk()
    if _USE_NLTK_STOPWORDS:
        from nltk.corpus import stopwords

        stops = set(stopwords.words("english"))
    else:
        stops = _BUILTIN_STOPS
    return [t for t in tokens if t.lower() not in stops and t.isalnum()]


def _porter_stem(word: str) -> str:
    """Lightweight Porter-like suffix stripping when NLTK stemmer unavailable."""
    w = word.lower()
    for suffix in ("ing", "ed", "es", "s", "ly", "ment", "tion"):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return w[: -len(suffix)]
    return w


def stem_tokens(tokens: list[str]) -> list[str]:
    ensure_nltk()
    if _USE_NLTK_STEM:
        from nltk.stem import PorterStemmer

        stemmer = PorterStemmer()
        return [stemmer.stem(t) for t in tokens]
    return [_porter_stem(t) for t in tokens]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    ensure_nltk()
    if _USE_NLTK_STEM:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(t) for t in tokens]
    return [_porter_stem(t) for t in tokens]


def preprocess_document(
    text: str,
    *,
    apply_lower: bool = True,
    apply_stop: bool = True,
    apply_hyphen: bool = True,
    apply_stem: bool = False,
    apply_lemma: bool = False,
) -> list[str]:
    if apply_hyphen:
        text = handle_hyphens(text)
    tokens = tokenize(text)
    if apply_lower:
        tokens = lowercase_tokens(tokens)
    tokens = [t for t in tokens if t.isalnum() or t.replace("'", "").isalnum()]
    if apply_stop:
        tokens = remove_stopwords(tokens)
    if apply_stem:
        tokens = stem_tokens(tokens)
    elif apply_lemma:
        tokens = lemmatize_tokens(tokens)
    return tokens


def build_inverted_index(docs: dict[str, str], preprocess_fn: Callable[[str], list[str]]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for doc_id, text in docs.items():
        for term in preprocess_fn(text):
            index[term].add(doc_id)
    return dict(index)


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = sum(v * v for v in vec_a.values()) ** 0.5
    norm_b = sum(v * v for v in vec_b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def doc_term_vector(tokens: list[str]) -> dict[str, float]:
    vec: dict[str, float] = defaultdict(float)
    for t in tokens:
        vec[t] += 1.0
    return dict(vec)


def average_query_similarity(docs: dict[str, str], queries: list[str], preprocess_fn: Callable[[str], list[str]]) -> float:
    scores = []
    for q in queries:
        q_vec = doc_term_vector(preprocess_fn(q))
        for text in docs.values():
            d_vec = doc_term_vector(preprocess_fn(text))
            scores.append(cosine_similarity(q_vec, d_vec))
    return sum(scores) / len(scores) if scores else 0.0
