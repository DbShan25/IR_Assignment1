"""Tolerant retrieval: wildcards, edit distance, k-grams, phonetic correction."""

from __future__ import annotations

import re
from collections import defaultdict

try:
    import jellyfish
except ImportError:
    jellyfish = None


def wildcard_to_regex(pattern: str) -> re.Pattern[str]:
    """* matches any sequence; ? matches single character."""
    escaped = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
    return re.compile(f"^{escaped}$", re.IGNORECASE)


def wildcard_search(pattern: str, vocabulary: list[str]) -> list[str]:
    regex = wildcard_to_regex(pattern)
    return sorted(t for t in vocabulary if regex.match(t))


def edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i]
        for j, c2 in enumerate(s2, 1):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def spelling_correction(query_term: str, vocabulary: list[str], max_distance: int = 2) -> list[tuple[str, int]]:
    suggestions = []
    for term in vocabulary:
        d = edit_distance(query_term, term)
        if d <= max_distance:
            suggestions.append((term, d))
    suggestions.sort(key=lambda x: (x[1], x[0]))
    return suggestions[:10]


def build_kgram_index(vocabulary: list[str], k: int = 2) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for term in vocabulary:
        padded = f"${term}$"
        for i in range(len(padded) - k + 1):
            gram = padded[i : i + k]
            index[gram].add(term)
    return dict(index)


def kgram_candidates(query: str, kgram_index: dict[str, set[str]], k: int = 2) -> set[str]:
    padded = f"${query}$"
    grams = [padded[i : i + k] for i in range(len(padded) - k + 1)]
    if not grams:
        return set()
    candidates = set(kgram_index.get(grams[0], set()))
    for g in grams[1:]:
        candidates &= kgram_index.get(g, set())
    return candidates


def kgram_spelling_correction(query: str, vocabulary: list[str], k: int = 2, max_distance: int = 2) -> list[tuple[str, int]]:
    kindex = build_kgram_index(vocabulary, k=k)
    candidates = kgram_candidates(query, kindex, k=k)
    if not candidates:
        candidates = set(vocabulary)
    return spelling_correction(query, sorted(candidates), max_distance=max_distance)


def phonetic_key(term: str) -> str:
    if jellyfish:
        return jellyfish.metaphone(term)
    # simple fallback: first letter + consonant skeleton
    term = term.lower()
    if not term:
        return ""
    first = term[0]
    rest = re.sub(r"[aeiou]", "", term[1:])
    return (first + rest)[:4]


def build_phonetic_index(vocabulary: list[str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for term in vocabulary:
        index[phonetic_key(term)].append(term)
    return dict(index)


def phonetic_correction(query: str, phonetic_index: dict[str, list[str]]) -> list[str]:
    key = phonetic_key(query)
    return sorted(phonetic_index.get(key, []))
