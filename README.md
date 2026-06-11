# Information Retrieval Assignment (S2-25)

Streamlit-based end-to-end Information Retrieval system for BITS WILP Assignment 1 (AIMLCZG537 / DSECLZG537).

**Group members:** see **[GROUP_INSTRUCTIONS.md](GROUP_INSTRUCTIONS.md)** for step-by-step run and verification steps.

## Dataset

This project uses a **public subset of the [20 Newsgroups corpus](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-dataset)**:

| Category | Documents |
|----------|-----------|
| `rec.sport.baseball` | 20 |
| `sci.med` | 20 |
| `talk.politics.misc` | 20 |

Bundled file: `data/20newsgroups_subset.csv` (60 documents, works offline on BITS lab).

See `data/DATASET.md` for source, license notes, and how to regenerate.

## Features

- Load bundled 20 Newsgroups subset or upload your own `.txt` / `.csv`
- Filter documents by newsgroup category
- Text preprocessing, inverted index, stemming vs lemmatization
- Phrase query: biword vs positional index
- BST vs B-Tree dictionary benchmark
- Tolerant retrieval: wildcards, edit distance, k-gram, phonetic correction
- Inference tab and `REPORT.md`

## Installation

```bash
cd IR_Assignment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Optional NLTK data (falls back to built-in tokenization if download fails):

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Run the application

From the project root (the folder that contains `app.py` and `src/`):

```bash
python scripts/verify_project.py   # optional check
streamlit run app.py
```

Or: `./run.sh` (macOS/Linux) / `run.bat` (Windows).

If you see `No module named 'src.indexing'`, your clone is missing files — see **GROUP_INSTRUCTIONS.md** §10.

Open the URL shown in the terminal (typically http://localhost:8501).

## Regenerate dataset (optional)

If you need a fresh download from scikit-learn:

```bash
python scripts/build_dataset.py
```

## Project structure

```
IR_Assignment/
├── app.py
├── requirements.txt
├── README.md
├── REPORT.md
├── data/
│   ├── 20newsgroups_subset.csv   # Bundled public dataset
│   └── DATASET.md
├── scripts/
│   └── build_dataset.py
└── src/
    ├── datasets.py
    ├── preprocessing.py
    ├── indexing.py
    ├── trees.py
    └── tolerant_retrieval.py
```

## Custom upload format

**CSV:** `doc_id`, `category` (optional), `text`

**Pipe-delimited text:** `doc_id|document text` per line

## Submission checklist

- [ ] `app.py` and `src/` modules
- [ ] `data/20newsgroups_subset.csv`
- [ ] `REPORT.md` with screenshots from Streamlit tabs
- [ ] `README.md` with run instructions
- [ ] Screenshots / short screen recording of the app

## Suggested demo queries

| Task | Example |
|------|---------|
| Ranked search | `baseball season`, `medical treatment` |
| Phrase query | `new york`, `years ago` |
| Wildcard | `med*`, `gov*` |
| Spelling typo | `basebal` → `baseball` |

## Notes

- Run all workflows through the Streamlit UI only.
- Copy the entire project folder to BITS Virtual Lab; no download needed if CSV is included.
