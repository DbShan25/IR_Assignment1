"""Inverted, biword, and positional index implementations."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable


def build_biword_index(docs: dict[str, list[str]]) -> dict[tuple[str, str], set[str]]:
    index: dict[tuple[str, str], set[str]] = defaultdict(set)
    for doc_id, tokens in docs.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            index[pair].add(doc_id)
    return dict(index)


def build_positional_index(docs: dict[str, list[str]]) -> dict[str, dict[str, list[int]]]:
    """term -> doc_id -> positions (1-based)."""
    index: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for doc_id, tokens in docs.items():
        for pos, term in enumerate(tokens, start=1):
            index[term][doc_id].append(pos)
    return {t: dict(postings) for t, postings in index.items()}


def phrase_search_biword(phrase_tokens: list[str], biword_index: dict[tuple[str, str], set[str]]) -> set[str]:
    if len(phrase_tokens) < 2:
        return set()
    first_pair = (phrase_tokens[0], phrase_tokens[1])
    if first_pair not in biword_index:
        return set()
    candidates = set(biword_index[first_pair])
    for i in range(1, len(phrase_tokens) - 1):
        pair = (phrase_tokens[i], phrase_tokens[i + 1])
        if pair not in biword_index:
            return set()
        candidates &= biword_index[pair]
    return candidates


def phrase_search_positional(phrase_tokens: list[str], positional_index: dict[str, dict[str, list[int]]]) -> set[str]:
    if not phrase_tokens:
        return set()
    if phrase_tokens[0] not in positional_index:
        return set()
    result_docs: set[str] = set()
    for doc_id, positions in positional_index[phrase_tokens[0]].items():
        for start_pos in positions:
            matched = True
            for offset, term in enumerate(phrase_tokens[1:], start=1):
                term_positions = positional_index.get(term, {}).get(doc_id, [])
                if start_pos + offset not in term_positions:
                    matched = False
                    break
            if matched:
                result_docs.add(doc_id)
    return result_docs


def boolean_search(term: str, inverted_index: dict[str, set[str]]) -> set[str]:
    return set(inverted_index.get(term, set()))


def rank_by_tf(query_tokens: list[str], doc_tokens: dict[str, list[str]]) -> list[tuple[str, int]]:
    scores: list[tuple[str, int]] = []
    q_set = set(query_tokens)
    for doc_id, tokens in doc_tokens.items():
        score = sum(1 for t in tokens if t in q_set)
        if score > 0:
            scores.append((doc_id, score))
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores
