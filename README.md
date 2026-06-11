# Information Retrieval Assignment (S2-25)

Streamlit-based end-to-end Information Retrieval system developed for **BITS WILP Assignment 1** (AIMLCZG537 / DSECLZG537).

## Group Members

| Register No | Name            |
| ----------- | --------------- |
| 2024DC04246 | DIVYABHARATHI S |
| 2024DC04243 | ABHAY BINDLISH  |
| 2024DC04241 | PRAGYA GAUR     |

---

# Live Streamlit Deployment

The application has been successfully deployed and verified through the BITS Virtual Lab environment.

**Deployment URL:**

```text
https://irassignment1-group27.streamlit.app/
```

---

# Dataset

This project uses a public subset of the **20 Newsgroups Dataset**.

| Category           | Documents |
| ------------------ | --------- |
| rec.sport.baseball | 19        |
| sci.med            | 20        |
| talk.politics.misc | 19        |
| **Total**          | **58**    |

Bundled dataset:

```text
data/20newsgroups_subset.csv
```

The dataset is included with the project and works offline.

See `data/DATASET.md` for source information and regeneration instructions.

---

# Features

## A. Streamlit-Based End-to-End Workflow

* Load bundled 20 Newsgroups subset
* Upload custom TXT and CSV document collections
* View uploaded documents
* Filter documents by category
* Enter search queries
* Display retrieval results
* Execute complete workflow through Streamlit UI

---

## B. Text Preprocessing

Implemented preprocessing operations:

* Tokenization
* Lowercasing
* Stop-word Removal
* Hyphen Handling
* Inverted Index Creation

The application displays:

* Raw tokens
* Processed tokens
* Vocabulary statistics
* Inverted index samples
* Retrieval quality comparison

### Preprocessing Results

| Configuration                | Average Cosine Similarity |
| ---------------------------- | ------------------------- |
| Baseline (Tokenization Only) | 0.0040                    |
| Lowercasing                  | 0.0051                    |
| Stop-word Removal            | 0.0092                    |
| Hyphen + Stop + Lower        | 0.0091                    |

**Inference:** Lowercasing and stop-word removal significantly improved retrieval quality.

---

## C. Stemming vs Lemmatization

Implemented:

* Porter Stemmer
* WordNet Lemmatizer

### Experimental Results

| Method                | Average Cosine Similarity |
| --------------------- | ------------------------- |
| No Normalization      | 0.0034                    |
| Porter Stemming       | 0.0064                    |
| WordNet Lemmatization | 0.0064                    |

**Inference:** Lemmatization preserves meaningful vocabulary while achieving retrieval effectiveness comparable to stemming.

---

## D. Phrase Query Processing

Implemented:

### Biword Index

Stores adjacent term pairs.

Example:

```text
new york city
```

Produces:

```text
(new york)
(york city)
```

### Positional Index

Stores exact term positions within documents.

Example:

```text
new -> [4]
york -> [5]
city -> [6]
```

### Example Phrase Query

```text
new york
```

**Inference:** Positional indexing provides more accurate phrase matching because it validates exact word positions.

---

## E. Dictionary Search using BST and B-Tree

Implemented:

* Binary Search Tree (BST)
* B-Tree (Degree = 3)

### Benchmark Results

Dictionary Size:

```text
1924 unique terms
```

Queries Tested:

```text
baseball
game
medical
doctor
```

Results:

| Structure          | Total Time (ms) | Avg Time per Query (ms) |
| ------------------ | --------------- | ----------------------- |
| Binary Search Tree | 0.6040          | 0.0012                  |
| B-Tree             | 0.6713          | 0.0013                  |

**Inference:** BST achieved slightly lower lookup time on the current dataset.

---

## F. Tolerant Retrieval

Implemented:

### Wildcard Queries

Example:

```text
med*
```

Matches:

```text
medical
medication
medicare
```

### Edit Distance Correction

Example:

```text
basebal → baseball
```

### K-Gram Correction

Example:

```text
goverment → government
```

### Phonetic Matching

Example:

```text
medisin → medicine
```

**Inference:** The retrieval model successfully handles spelling mistakes, wildcard searches, edit-distance corrections, and phonetic variations.

---

## G. Inference and Discussion

The application provides automated discussion and analysis regarding:

* Preprocessing effectiveness
* Stemming vs Lemmatization
* Phrase query accuracy
* BST vs B-Tree performance
* Tolerant retrieval effectiveness
* Limitations
* Future improvements

---

## H. Experimental Results

| Query             | Top Document           | Score |
| ----------------- | ---------------------- | ----- |
| baseball season   | rec_sport_baseball_004 | 5     |
| medical treatment | sci_med_020            | 2     |
| government policy | talk_politics_misc_018 | 6     |

---

# Installation

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

Optional NLTK downloads:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

# Run the Application

From the project root:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# Running in BITS Virtual Lab

The deployed application can be accessed directly without local installation:

```text
https://irassignment1-group27.streamlit.app/
```

Steps:

1. Open Firefox in Virtual Lab.
2. Navigate to the deployment URL.
3. Load bundled dataset or upload custom files.
4. Execute all assignment tasks through the Streamlit interface.

---

# Project Structure

```text
IR_Assignment1/
│
├── app.py
├── requirements.txt
├── README.md
├── REPORT.md
│
├── data/
│   ├── 20newsgroups_subset.csv
│   └── DATASET.md
│
├── scripts/
│   └── build_dataset.py
│
└── src/
    ├── datasets.py
    ├── preprocessing.py
    ├── indexing.py
    ├── trees.py
    └── tolerant_retrieval.py
```

---

# Suggested Demo Queries

| Task                | Example           |
| ------------------- | ----------------- |
| Ranked Search       | baseball season   |
| Ranked Search       | medical treatment |
| Ranked Search       | government policy |
| Phrase Query        | new york          |
| Phrase Query        | years ago         |
| Wildcard Query      | med*              |
| Wildcard Query      | gov*              |
| Spelling Correction | basebal           |
| Edit Distance       | goverment         |

---

# Submission Checklist

* [x] Streamlit Application
* [x] Dataset Included
* [x] Text Preprocessing
* [x] Stemming vs Lemmatization
* [x] Phrase Query Processing
* [x] BST vs B-Tree Benchmark
* [x] Tolerant Retrieval
* [x] Inference and Discussion
* [x] Experimental Results
* [x] Virtual Lab Execution
* [x] Deployment URL
* [x] README.md
* [x] REPORT.pdf
* [x] Screenshots
* [x] Demo Video

---

# Notes

* All workflows are executed through the Streamlit interface only.
* Dataset is bundled and works offline.
* Application was successfully verified from the BITS Virtual Lab environment.
* Streamlit deployment URL can be used for demonstration and evaluation.

---

# Deployment URL

https://irassignment1-group27.streamlit.app/
