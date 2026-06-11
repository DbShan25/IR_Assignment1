# Information Retrieval Assignment 1 — Report

**Course:** Information Retrieval (AIMLCZG537 / DSECLZG537)  
**Assignment:** End-to-end Streamlit IR system  
**Dataset:** [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-dataset) subset (`data/20newsgroups_subset.csv`)

---

## Dataset description

| Field | Value |
|-------|--------|
| **Corpus** | 20 Newsgroups (Ken Lang; standard ML/IR benchmark) |
| **Loader** | `sklearn.datasets.fetch_20newsgroups` |
| **Subset** | 60 documents — 20 each from `rec.sport.baseball`, `sci.med`, `talk.politics.misc` |
| **Preprocessing at source** | Headers, footers, quotes removed (`remove=` parameter) |
| **Reproducibility** | `random_state=42`, `shuffle=True` |

The bundled CSV columns are `doc_id`, `category`, `text`. This is a **publicly available** dataset suitable for coursework and cited in numerous IR/NLP papers.

---

## 1. Implementation overview

| Module | Responsibility |
|--------|----------------|
| `datasets.py` | Load bundled CSV; optional sklearn download |
| `preprocessing.py` | Tokenization, stop words, stemming/lemmatization, inverted index |
| `indexing.py` | Biword, positional index, TF ranking |
| `trees.py` | BST and B-Tree with benchmarks |
| `tolerant_retrieval.py` | Wildcards, edit distance, k-gram, phonetic |

All tasks run through Streamlit tabs in `app.py`.

---

## 2. Task A — Streamlit workflow

1. Sidebar shows dataset name, category counts, and link to official documentation.
2. **Documents & Query** tab lists posts with category labels; optional category filter.
3. Users run ranked, boolean, or phrase retrieval from the UI.
4. Example queries: `baseball season`, phrase `new york`.

---

## 3. Task B — Text preprocessing

**Pipeline:** tokenization → hyphen handling → lowercasing → stop-word removal → optional stem/lemma → inverted index.

**Test queries on 20 Newsgroups:** `baseball game`, `medical research`, `government policy`

**Inference:** Stop-word removal and lowercasing improve retrieval on informal Usenet text by removing high-frequency words (*the*, *is*, *in*) and normalizing case. The vocabulary grows to thousands of terms (vs. 8 docs in a toy corpus), making preprocessing effects more visible in the inverted index sample.

---

## 4. Task C — Stemming vs lemmatization

**Method:** Average cosine similarity over queries such as `games`, `medical`, `government`, `president`.

**Inference:** Newsgroup posts contain inflected verbs and plurals. **Lemmatization** often yields higher similarity because forms like *studies* → *study* remain valid words. **Porter stemming** is faster but may produce non-words (*studi*). For this conversational English dataset, lemmatization is usually more suitable; stemming is acceptable at web scale when speed dominates.

---

## 5. Task D — Phrase query

**Example:** `new york` (appears in multiple politics/sports posts).

| Index | Behavior |
|-------|----------|
| Biword | Matches adjacent pair `(world, series)` |
| Positional | Requires full phrase at consecutive positions |

**Inference:** **Positional index is more accurate.** Biword can false-positive when the same pair appears in a different semantic context. On 60-document subset, phrase hits are concentrated in `rec.sport.baseball` for sports phrases.

---

## 6. Task E — BST vs B-Tree

With ~60 documents, vocabulary is typically **3,000–8,000** terms after preprocessing — large enough to observe lookup-time differences.

Run the in-app benchmark with queries: `baseball`, `medical`, `government`, etc.

**Inference:** Paste your timing table here after running the app. B-Tree height stays bounded; BST is simpler for teaching demos. At full 20 Newsgroups scale (~20k docs), B-Tree advantages increase.

| Structure | Total time (ms) | Avg per query (ms) |
|-----------|-----------------|---------------------|
| BST | *(from app)* | *(from app)* |
| B-Tree | *(from app)* | *(from app)* |

---

## 7. Task F — Tolerant retrieval

| Technique | Example on this dataset |
|-----------|-------------------------|
| Wildcard `med*` | Medical/science terms |
| Edit distance `basebal` | Corrects to `baseball` |
| K-gram `goverment` | Suggests `government` |
| Phonetic | Matches pronunciation-alike terms |

**Inference:** Larger vocabulary from newsgroups improves spelling correction vs. tiny toy corpora. Wildcards help exploratory search across morphological variants.

---

## 8. Task G — Limitations and improvements

**Limitations:** TF ranking only; English; in-memory index; subset is 0.3% of full corpus.

**Improvements:** BM25, category-aware ranking, full 20 Newsgroups, evaluation with precision@k per category.

---

## 9. Screenshots (add before submission)

1. Sidebar — dataset info and category counts  
2. Documents & Query — filtered baseball posts  
3. Preprocessing — inverted index sample  
4. Stem vs Lemma — comparison table  
5. Phrase query — biword vs positional  
6. BST vs B-Tree — benchmark table  
7. Tolerant retrieval — spelling correction  

---

## 10. Reproduce

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional: `python scripts/build_dataset.py` to refresh CSV from sklearn.
