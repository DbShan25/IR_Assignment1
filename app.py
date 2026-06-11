"""
Information Retrieval Assignment - Streamlit End-to-End Application
BITS IR (AIMLCZG537 / DSECLZG537) Assignment 1
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

# Path setup must run before other local imports (supports any cwd if app.py path is correct).
_APP_ROOT = Path(__file__).resolve().parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))
from src.path_setup import setup_paths, verify_project_files  # noqa: E402

ROOT = setup_paths()
verify_project_files()

import pandas as pd
import streamlit as st

try:
    from indexing import (  # noqa: E402
        build_biword_index,
        build_positional_index,
        phrase_search_biword,
        phrase_search_positional,
        rank_by_tf,
    )
    from preprocessing import (  # noqa: E402
        average_query_similarity,
        build_inverted_index,
        preprocess_document,
        tokenize,
    )
    from tolerant_retrieval import (  # noqa: E402
        build_phonetic_index,
        edit_distance,
        kgram_spelling_correction,
        phonetic_correction,
        spelling_correction,
        wildcard_search,
    )
    from datasets import DATASET_INFO, load_default_documents  # noqa: E402
    from trees import BinarySearchTree, BTree, benchmark_tree_search  # noqa: E402
except ImportError:
    from src.indexing import (  # noqa: E402
        build_biword_index,
        build_positional_index,
        phrase_search_biword,
        phrase_search_positional,
        rank_by_tf,
    )
    from src.preprocessing import (  # noqa: E402
        average_query_similarity,
        build_inverted_index,
        preprocess_document,
        tokenize,
    )
    from src.tolerant_retrieval import (  # noqa: E402
        build_phonetic_index,
        edit_distance,
        kgram_spelling_correction,
        phonetic_correction,
        spelling_correction,
        wildcard_search,
    )
    from src.datasets import DATASET_INFO, load_default_documents  # noqa: E402
    from src.trees import BinarySearchTree, BTree, benchmark_tree_search  # noqa: E402

st.set_page_config(page_title="IR Assignment System", layout="wide")
st.title("Information Retrieval System")
st.caption(
    "20 Newsgroups subset · preprocessing · phrase search · BST/B-Tree · tolerant retrieval"
)

# Example queries tuned for the bundled newsgroup categories
DEFAULT_SEARCH_QUERY = "baseball season"
DEFAULT_PHRASE_QUERY = "new york"
DEFAULT_BENCH_QUERIES = (
    "baseball\ngame\nmedical\ndoctor\ngovernment\npresident\n"
    "team\nhealth\npolicy\nplayer\n"
)
DEFAULT_WILDCARD = "med*"
DEFAULT_TYPO = "basebal"
DEFAULT_KGRAM_TYPO = "goverment"
DEFAULT_PHONETIC = "medicine"


def parse_uploaded_files(uploaded_files) -> tuple[dict[str, str], pd.DataFrame | None]:
    docs: dict[str, str] = {}
    meta_rows: list[dict] = []
    for f in uploaded_files:
        name = Path(f.name).stem
        raw = f.read().decode("utf-8", errors="ignore")
        if f.name.endswith(".txt") and "|" in raw.split("\n", 1)[0]:
            for line in raw.strip().splitlines():
                if "|" in line:
                    doc_id, text = line.split("|", 1)
                    docs[doc_id.strip()] = text.strip()
                    meta_rows.append({"doc_id": doc_id.strip(), "category": "uploaded", "text": text.strip()})
        elif f.name.endswith(".csv"):
            df = pd.read_csv(io.StringIO(raw))
            id_col = "doc_id" if "doc_id" in df.columns else df.columns[0]
            text_col = "text" if "text" in df.columns else df.columns[-1]
            cat_col = "category" if "category" in df.columns else None
            for _, row in df.iterrows():
                doc_id = str(row[id_col])
                text = str(row[text_col])
                docs[doc_id] = text
                meta_rows.append(
                    {
                        "doc_id": doc_id,
                        "category": str(row[cat_col]) if cat_col else "uploaded",
                        "text": text,
                    }
                )
        else:
            docs[name] = raw.strip()
            meta_rows.append({"doc_id": name, "category": "uploaded", "text": raw.strip()})
    meta = pd.DataFrame(meta_rows) if meta_rows else None
    return docs, meta


def get_preprocess_opts() -> dict:
    return {
        "apply_lower": st.session_state.get("opt_lower", True),
        "apply_stop": st.session_state.get("opt_stop", True),
        "apply_hyphen": st.session_state.get("opt_hyphen", True),
        "apply_stem": st.session_state.get("opt_stem", False),
        "apply_lemma": st.session_state.get("opt_lemma", False),
    }


def preprocess_all(docs: dict[str, str], **opts) -> dict[str, list[str]]:
    return {doc_id: preprocess_document(text, **opts) for doc_id, text in docs.items()}


def load_dataset_into_session() -> None:
    docs, meta = load_default_documents(ROOT)
    st.session_state.documents = docs
    st.session_state.doc_meta = meta
    st.session_state.doc_tokens = preprocess_all(docs)


def init_session():
    if "documents" not in st.session_state:
        load_dataset_into_session()
    if "doc_tokens" not in st.session_state:
        st.session_state.doc_tokens = preprocess_all(st.session_state.documents)


init_session()

with st.sidebar:
    st.header("Dataset")
    st.markdown(f"**{DATASET_INFO['name']}**")
    st.caption(DATASET_INFO["description"])
    meta_df = st.session_state.get("doc_meta")
    if meta_df is not None and "category" in meta_df.columns:
        st.write("**Documents per category:**")
        counts = meta_df["category"].value_counts().rename("count").reset_index()
        st.dataframe(counts, hide_index=True, use_container_width=True)
    st.link_button("Dataset documentation", "https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-dataset")

    uploaded = st.file_uploader(
        "Upload documents (.txt, .csv)",
        type=["txt", "csv"],
        accept_multiple_files=True,
    )
    if st.button("Reload 20 Newsgroups subset"):
        load_dataset_into_session()
        st.success(f"Loaded {len(st.session_state.documents)} documents from public corpus.")

    if uploaded:
        docs, meta = parse_uploaded_files(uploaded)
        st.session_state.documents = docs
        st.session_state.doc_meta = meta
        st.session_state.doc_tokens = preprocess_all(docs)
        st.success(f"Loaded {len(docs)} document(s).")

    st.divider()
    st.header("Preprocessing options")
    st.session_state.opt_lower = st.checkbox("Lowercasing", value=True)
    st.session_state.opt_stop = st.checkbox("Stop word removal", value=True)
    st.session_state.opt_hyphen = st.checkbox("Hyphen handling", value=True)
    st.session_state.opt_stem = st.checkbox("Stemming", value=False)
    st.session_state.opt_lemma = st.checkbox("Lemmatization", value=False)
    if st.session_state.opt_stem and st.session_state.opt_lemma:
        st.warning("Enable only one of Stemming or Lemmatization at a time for clear comparison.")

    if st.button("Apply preprocessing to collection"):
        opts = get_preprocess_opts()
        st.session_state.doc_tokens = preprocess_all(st.session_state.documents, **opts)
        st.session_state.inverted_index = build_inverted_index(
            st.session_state.documents,
            lambda t: preprocess_document(t, **opts),
        )
        st.success("Preprocessing applied.")

tab_upload, tab_prep, tab_stem, tab_phrase, tab_trees, tab_tolerant, tab_inference, tab_results = st.tabs(
    [
        "A. Documents & Query",
        "B. Preprocessing",
        "C. Stem vs Lemma",
        "D. Phrase Query",
        "E. BST vs B-Tree",
        "F. Tolerant Retrieval",
        "G. Inference",
        "H. Experimental Results"
    ]
)

docs = st.session_state.documents
meta_df = st.session_state.get("doc_meta")
opts = get_preprocess_opts()

doc_categories: dict[str, str] = {}
if meta_df is not None and "category" in meta_df.columns:
    doc_categories = dict(zip(meta_df["doc_id"].astype(str), meta_df["category"].astype(str)))

# --- Tab A: Upload & Query ---
with tab_upload:
    st.subheader("Document collection (20 Newsgroups subset)")
    category_filter = "All"
    if doc_categories:
        categories = sorted(set(doc_categories.values()))
        category_filter = st.selectbox("Filter by category", ["All"] + categories)

    shown = 0
    for doc_id, text in docs.items():
        cat = doc_categories.get(doc_id, "")
        if category_filter != "All" and cat != category_filter:
            continue
        shown += 1
        label = f"{doc_id}" + (f" · {cat}" if cat else "")
        with st.expander(label):
            st.write(text[:2000] + ("..." if len(text) > 2000 else ""))
    st.caption(f"Showing {shown} of {len(docs)} documents.")

    st.subheader("Search query")
    query = st.text_input("Enter search query", value=DEFAULT_SEARCH_QUERY)
    retrieval_mode = st.selectbox(
        "Retrieval technique",
        ["TF-based ranked retrieval", "Boolean (single term)", "Phrase query (positional)"],
    )

    if st.button("Run retrieval", type="primary"):
        tokens = preprocess_document(query, **opts)
        doc_tokens = preprocess_all(docs, **opts)
        inv = build_inverted_index(docs, lambda t: preprocess_document(t, **opts))

        if retrieval_mode == "Boolean (single term)" and tokens:
            hits = inv.get(tokens[0], set())
            st.write("**Boolean results:**", sorted(hits) if hits else "No match.")
        elif retrieval_mode == "Phrase query (positional)":
            pos_index = build_positional_index(doc_tokens)
            hits = phrase_search_positional(tokens, pos_index)
            st.write("**Phrase results (positional):**", sorted(hits) if hits else "No match.")
        else:
            ranked = rank_by_tf(tokens, doc_tokens)
            st.write("**Ranked results (TF score):**")
            if ranked:
                st.table(pd.DataFrame(ranked, columns=["Document", "Score"]))
            else:
                st.info("No documents matched the query.")

# --- Tab B: Preprocessing ---
with tab_prep:
    st.subheader("Text preprocessing pipeline")
    sample_doc = next(iter(docs.values()), "")
    st.write("**Sample document (raw):**", sample_doc[:300] + ("..." if len(sample_doc) > 300 else ""))

    steps = []
    text = sample_doc
    steps.append(("Raw tokens", tokenize(text)))

    if opts["apply_hyphen"]:
        from preprocessing import handle_hyphens

        text = handle_hyphens(text)
        steps.append(("After hyphen handling", tokenize(text)))

    if opts["apply_lower"]:
        from preprocessing import lowercase_tokens

        steps.append(("After lowercasing", lowercase_tokens(tokenize(text))))

    if opts["apply_stop"]:
        from preprocessing import remove_stopwords

        prev = steps[-1][1] if steps else tokenize(text)
        steps.append(("After stop word removal", remove_stopwords(prev)))

    final_tokens = preprocess_document(sample_doc, **opts)
    steps.append(("Final processed tokens", final_tokens))

    for label, toks in steps:
        st.markdown(f"**{label}** ({len(toks)} tokens)")
        st.code(", ".join(toks[:40]) + (" ..." if len(toks) > 40 else ""))

    inv = build_inverted_index(docs, lambda t: preprocess_document(t, **opts))
    st.subheader("Inverted index (sample terms)")
    sample_terms = sorted(inv.keys())[:15]
    inv_sample = {t: sorted(inv[t]) for t in sample_terms}
    st.json(inv_sample)
    st.caption(f"Total vocabulary size: {len(inv)} terms")

    st.subheader("Effect of preprocessing on retrieval")
    test_queries = ["baseball game", "medical research", "government policy"]
    configs = [
        ("Baseline (tokenize only)", {"apply_lower": False, "apply_stop": False, "apply_hyphen": False}),
        ("+ Lowercasing", {"apply_lower": True, "apply_stop": False, "apply_hyphen": False}),
        ("+ Stop words", {"apply_lower": True, "apply_stop": True, "apply_hyphen": False}),
        ("+ Hyphen + Stop + Lower", {"apply_lower": True, "apply_stop": True, "apply_hyphen": True}),
    ]
    rows = []
    for name, cfg in configs:
        base = {**opts, **cfg, "apply_stem": False, "apply_lemma": False}
        sim = average_query_similarity(docs, test_queries, lambda t: preprocess_document(t, **base))
        rows.append({"Configuration": name, "Avg cosine similarity": round(sim, 4)})
    st.table(pd.DataFrame(rows))

# --- Tab C: Stemming vs Lemmatization ---
with tab_stem:
    st.subheader("Stemming vs Lemmatization comparison")
    test_queries = ["games", "players", "medical", "doctors", "government", "president", "teams"]
    stem_opts = {**opts, "apply_stem": True, "apply_lemma": False}
    lemma_opts = {**opts, "apply_stem": False, "apply_lemma": True}
    none_opts = {**opts, "apply_stem": False, "apply_lemma": False}

    stem_sim = average_query_similarity(docs, test_queries, lambda t: preprocess_document(t, **stem_opts))
    lemma_sim = average_query_similarity(docs, test_queries, lambda t: preprocess_document(t, **lemma_opts))
    none_sim = average_query_similarity(docs, test_queries, lambda t: preprocess_document(t, **none_opts))

    comparison = pd.DataFrame(
        [
            {"Method": "No stemming/lemmatization", "Avg cosine similarity": round(none_sim, 4)},
            {"Method": "Porter stemming", "Avg cosine similarity": round(stem_sim, 4)},
            {"Method": "WordNet lemmatization", "Avg cosine similarity": round(lemma_sim, 4)},
        ]
    )
    st.table(comparison)

    st.subheader("Example transformations")
    examples = ["running", "studies", "playing", "government", "medical"]
    from preprocessing import lemmatize_tokens, stem_tokens

    ex_df = pd.DataFrame(
        {
            "Token": examples,
            "Stemmed": [stem_tokens([e])[0] for e in examples],
            "Lemmatized": [lemmatize_tokens([e])[0] for e in examples],
        }
    )
    st.table(ex_df)

    winner = "Lemmatization" if lemma_sim >= stem_sim else "Stemming"
    st.success(
        f"**Conclusion:** {winner} achieved higher average cosine similarity "
        f"({max(lemma_sim, stem_sim):.4f}) on this dataset. "
        "On informal newsgroup text, lemmatization often preserves readable word forms "
        "(e.g. *studies* → *study*) while Porter stemming may over-clip (*studies* → *studi*). "
        "Stemming is faster on very large corpora."
    )

# --- Tab D: Phrase Query ---
with tab_phrase:
    st.subheader("Phrase query: Biword vs Positional index")
    phrase_query = st.text_input("Phrase query", value=DEFAULT_PHRASE_QUERY)
    phrase_tokens = preprocess_document(phrase_query, **{**opts, "apply_stem": False, "apply_lemma": False})
    doc_tokens = preprocess_all(docs, **{**opts, "apply_stem": False, "apply_lemma": False})

    biword = build_biword_index(doc_tokens)
    positional = build_positional_index(doc_tokens)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Biword index (sample pairs)**")
        sample_pairs = list(biword.items())[:8]
        st.json({f"{a} {b}": sorted(v) for (a, b), v in sample_pairs})

    with col2:
        st.markdown("**Positional index (sample term)**")
        if phrase_tokens:
            t0 = phrase_tokens[0]
            st.json(positional.get(t0, {}))

    biword_hits = phrase_search_biword(phrase_tokens, biword)
    pos_hits = phrase_search_positional(phrase_tokens, positional)

    st.markdown("**Query results**")
    res_df = pd.DataFrame(
        [
            {"Index type": "Biword", "Documents": ", ".join(sorted(biword_hits)) or "None"},
            {"Index type": "Positional", "Documents": ", ".join(sorted(pos_hits)) or "None"},
        ]
    )
    st.table(res_df)

    st.markdown(
        """
        **False positive example (biword):** For phrase *"new york"* a biword hit only checks the
        pair `(new, york)`. If those tokens appear adjacently in unrelated contexts, biword may
        return extra documents. Positional index still requires the full ordered phrase at
        consecutive positions (and matches the same hits for two-word phrases).

        **Why positional is more accurate:** It stores term positions and checks that each term in the
        phrase appears at `pos, pos+1, pos+2, ...`, eliminating false positives from overlapping biwords
        in different contexts.
        """
    )
    if biword_hits != pos_hits:
        st.warning(f"Mismatch detected: biword={biword_hits}, positional={pos_hits}")

# --- Tab E: BST vs B-Tree ---
with tab_trees:
    st.subheader("Dictionary search: BST vs B-Tree")
    doc_tokens = preprocess_all(docs, **opts)
    inv = build_inverted_index(docs, lambda t: preprocess_document(t, **opts))

    bst = BinarySearchTree()
    btree = BTree(t=3)
    for term, postings in inv.items():
        for doc_id in postings:
            bst.insert(term, doc_id)
            btree.insert(term, doc_id)

    st.write(f"Dictionary size: **{len(inv)}** unique terms")

    bench_queries = st.text_area(
        "Queries for benchmark (one per line)",
        value=DEFAULT_BENCH_QUERIES,
    ).strip().splitlines()
    bench_queries = [q.strip().lower() for q in bench_queries if q.strip()]
    repeats = st.slider("Repetitions per query", 10, 200, 50)

    if st.button("Run BST vs B-Tree benchmark"):
        bst_stats = benchmark_tree_search(bst, bench_queries, repeats=repeats)
        btree_stats = benchmark_tree_search(btree, bench_queries, repeats=repeats)

        table = pd.DataFrame(
            [
                {
                    "Structure": "Binary Search Tree",
                    "Total time (ms)": round(bst_stats["total_time_ms"], 4),
                    "Avg time per query (ms)": round(bst_stats["avg_time_ms"], 6),
                    "Hits": bst_stats["hits"],
                },
                {
                    "Structure": "B-Tree (t=3)",
                    "Total time (ms)": round(btree_stats["total_time_ms"], 4),
                    "Avg time per query (ms)": round(btree_stats["avg_time_ms"], 6),
                    "Hits": btree_stats["hits"],
                },
            ]
        )
        st.table(table)

        faster = "B-Tree" if btree_stats["avg_time_ms"] <= bst_stats["avg_time_ms"] else "BST"
        st.info(
            f"**Inference:** On this vocabulary ({len(inv)} terms), **{faster}** showed lower average "
            "lookup time. B-trees keep height bounded for bulk dictionary storage; BST performance depends "
            "on insertion order. For larger collections on disk, B-trees are typically preferred."
        )

    demo_term = st.text_input("Single-term dictionary lookup", value="baseball")
    if demo_term:
        t0 = time.perf_counter()
        bst_res = bst.search(demo_term.lower())
        bst_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        btree_res = btree.search(demo_term.lower())
        btree_ms = (time.perf_counter() - t1) * 1000
        st.write(f"BST: {sorted(bst_res)} ({bst_ms:.4f} ms) | B-Tree: {sorted(btree_res)} ({btree_ms:.4f} ms)")

# --- Tab F: Tolerant Retrieval ---
with tab_tolerant:
    st.subheader("Tolerant retrieval")
    doc_tokens = preprocess_all(docs, **opts)
    inv = build_inverted_index(docs, lambda t: preprocess_document(t, **opts))
    vocabulary = sorted(inv.keys())

    mode = st.selectbox(
        "Technique",
        ["Wildcard query", "Edit distance spelling correction", "K-gram index", "Phonetic correction"],
    )

    if mode == "Wildcard query":
        pattern = st.text_input("Wildcard pattern (* = any, ? = one char)", value=DEFAULT_WILDCARD)
        matches = wildcard_search(pattern, vocabulary)
        st.write("**Matching terms:**", matches)
        if matches:
            all_docs: set[str] = set()
            for m in matches:
                all_docs |= inv.get(m, set())
            st.write("**Retrieved documents:**", sorted(all_docs))

    elif mode == "Edit distance spelling correction":
        typo = st.text_input("Misspelled term", value=DEFAULT_TYPO)
        max_d = st.slider("Max edit distance", 1, 3, 2)
        suggestions = spelling_correction(typo, vocabulary, max_distance=max_d)
        st.table(pd.DataFrame(suggestions, columns=["Suggestion", "Edit distance"]))
        if suggestions:
            corrected = suggestions[0][0]
            st.write("**Documents for best correction:**", sorted(inv.get(corrected, set())))

    elif mode == "K-gram index":
        typo = st.text_input("Query term (k-gram)", value=DEFAULT_KGRAM_TYPO)
        k = st.slider("K", 2, 4, 2)
        suggestions = kgram_spelling_correction(typo, vocabulary, k=k)
        st.table(pd.DataFrame(suggestions, columns=["Suggestion", "Edit distance"]))
        st.caption("K-gram index narrows candidates before edit-distance filtering.")

    else:
        typo = st.text_input("Query term (phonetic)", value=DEFAULT_PHONETIC)
        ph_index = build_phonetic_index(vocabulary)
        matches = phonetic_correction(typo, ph_index)
        st.write("**Phonetically similar terms:**", matches)
        st.write("**Sample phonetic keys:**", {w: ph_index.get(w, [])[:3] for w in list(ph_index.keys())[:5]})

# --- Tab G: Inference ---
with tab_inference:
    st.subheader("Inference and discussion")
    st.markdown(
        """
        ### Summary of experimental findings

        1. **Preprocessing:** On the **20 Newsgroups** subset, lowercasing and stop-word removal
           reduce noisy tokens from informal Usenet posts. Combined preprocessing typically
           increases average query–document cosine similarity for domain queries
           (*baseball game*, *medical research*, *government policy*).

        2. **Stemming vs lemmatization:** Compare scores on tab *Stem vs Lemma*. Newsgroup text
           contains inflected verbs and plurals; lemmatization often preserves readable forms
           while Porter stemming is faster but can over-stem informal vocabulary.

        3. **Phrase query:** Positional index is more accurate than biword for multi-word phrases
           because it enforces ordered adjacent positions; biword may admit false positives when
           the same pairs appear in non-phrase contexts.

        4. **BST vs B-Tree:** Run the benchmark on tab *BST vs B-Tree*. B-tree height stays bounded;
           BST is simpler for in-memory small vocabularies. Report table times from your run.

        5. **Tolerant retrieval:** Wildcards support partial terms; edit distance and k-grams correct
           typos; phonetic matching helps sound-alike errors. Effectiveness depends on vocabulary size.

        ### Limitations
        - Boolean/TF retrieval only (no BM25 or vector embeddings).
        - English-focused NLP (NLTK stopwords/stemmer).
        - Small in-memory index; not optimized for web scale.

        ### Improvements
        - Add BM25 ranking, query expansion, and snippet highlighting.
        - Persist indexes (e.g. SQLite/Whoosh) and support PDF/HTML parsing.
        - Evaluate with precision/recall on a judged query set.
        """
    )
    st.download_button(
        "Download REPORT.md",
        data=(ROOT / "REPORT.md").read_text(encoding="utf-8") if (ROOT / "REPORT.md").exists() else "",
        file_name="REPORT.md",
        mime="text/markdown",
    )

# --- Tab G: Inference ---
with tab_results:
    st.subheader("Experimental Results Summary")

    st.markdown("### Retrieval Queries")

    queries = [
        "baseball season",
        "medical treatment",
        "government policy"
    ]

    doc_tokens = preprocess_all(docs, **opts)

    rows = []

    for q in queries:
        q_tokens = preprocess_document(q, **opts)
        ranked = rank_by_tf(q_tokens, doc_tokens)

        top_doc = ranked[0][0] if ranked else "None"
        score = ranked[0][1] if ranked else 0

        rows.append({
            "Query": q,
            "Top Document": top_doc,
            "Score": score
        })

    st.table(pd.DataFrame(rows))
