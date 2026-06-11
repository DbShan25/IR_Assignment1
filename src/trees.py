"""Binary Search Tree and B-Tree dictionary implementations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class BSTNode:
    key: str
    postings: set[str] = field(default_factory=set)
    left: BSTNode | None = None
    right: BSTNode | None = None


class BinarySearchTree:
    def __init__(self) -> None:
        self.root: BSTNode | None = None
        self.size = 0

    def insert(self, key: str, doc_id: str) -> None:
        if self.root is None:
            self.root = BSTNode(key=key, postings={doc_id})
            self.size += 1
            return
        node = self.root
        while True:
            if key == node.key:
                node.postings.add(doc_id)
                return
            if key < node.key:
                if node.left is None:
                    node.left = BSTNode(key=key, postings={doc_id})
                    self.size += 1
                    return
                node = node.left
            else:
                if node.right is None:
                    node.right = BSTNode(key=key, postings={doc_id})
                    self.size += 1
                    return
                node = node.right

    def search(self, key: str) -> set[str]:
        node = self.root
        while node:
            if key == node.key:
                return set(node.postings)
            node = node.left if key < node.key else node.right
        return set()

    def inorder_keys(self) -> Iterator[str]:
        stack: list[BSTNode] = []
        node = self.root
        while stack or node:
            while node:
                stack.append(node)
                node = node.left
            node = stack.pop()
            yield node.key
            node = node.right


class BTreeNode:
    def __init__(self, leaf: bool = True, t: int = 3) -> None:
        self.leaf = leaf
        self.t = t
        self.keys: list[str] = []
        self.postings: list[set[str]] = []
        self.children: list[BTreeNode] = []

    def is_full(self) -> bool:
        return len(self.keys) == (2 * self.t) - 1


class BTree:
    """Simple B-tree of order t for string keys with posting lists."""

    def __init__(self, t: int = 3) -> None:
        self.t = t
        self.root = BTreeNode(leaf=True, t=t)
        self.size = 0

    def _split_child(self, parent: BTreeNode, index: int) -> None:
        t = self.t
        full = parent.children[index]
        mid = BTreeNode(leaf=full.leaf, t=t)
        mid.keys = full.keys[t:]
        mid.postings = full.postings[t:]
        full.keys = full.keys[: t - 1]
        full.postings = full.postings[: t - 1]
        if not full.leaf:
            mid.children = full.children[t:]
            full.children = full.children[:t]
        parent.keys.insert(index, full.keys.pop())
        parent.postings.insert(index, full.postings.pop())
        parent.children.insert(index + 1, mid)

    def insert_non_full(self, node: BTreeNode, key: str, doc_id: str) -> None:
        i = len(node.keys) - 1
        if node.leaf:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            if i >= 0 and key == node.keys[i]:
                node.postings[i].add(doc_id)
                return
            node.keys.insert(i + 1, key)
            node.postings.insert(i + 1, {doc_id})
            self.size += 1
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if node.children[i].is_full():
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self.insert_non_full(node.children[i], key, doc_id)

    def insert(self, key: str, doc_id: str) -> None:
        root = self.root
        if root.is_full():
            new_root = BTreeNode(leaf=False, t=self.t)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
        self.insert_non_full(self.root, key, doc_id)

    def search(self, key: str) -> set[str]:
        node = self.root
        while node:
            i = 0
            while i < len(node.keys) and key > node.keys[i]:
                i += 1
            if i < len(node.keys) and key == node.keys[i]:
                return set(node.postings[i])
            if node.leaf:
                return set()
            node = node.children[i]
        return set()


def build_dictionary_from_index(inverted_index: dict[str, set[str]]) -> list[str]:
    return sorted(inverted_index.keys())


def benchmark_tree_search(
    tree,
    queries: list[str],
    repeats: int = 50,
) -> dict[str, float]:
    total = 0.0
    hits = 0
    for _ in range(repeats):
        for q in queries:
            start = time.perf_counter()
            result = tree.search(q)
            total += time.perf_counter() - start
            if result:
                hits += 1
    return {
        "total_time_ms": total * 1000,
        "avg_time_ms": (total * 1000) / (len(queries) * repeats) if queries else 0,
        "hits": hits,
    }
