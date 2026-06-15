# MuTech Phase 3 — NLP Text Mining & Preprocessing

Systematic Literature Review on Intelligent Music Processing (MIR)

This is the code for **Phase 3** of the MuTech SLR pipeline. It turns the screened
papers (CSV) into clean, tokenized, numerical **TF-IDF vectors** that Phase 4 feeds
into the clustering / topic models (K-Means, NMF, Agglomerative, LDA).

It was built to run **now**, in parallel, before the final filtered CSV is ready —
just point it at whatever CSV you have, and re-run on Marian's final ~84–100 papers
when she's done. Column names are auto-detected, so no code changes are needed.

## Two ways to run it

| File | Use it when |
|------|-------------|
| `MuTech_Phase3_Text_Mining.ipynb` | **Easiest** — open in Google Colab, run top to bottom, upload your CSV when prompted. Recommended for sharing with the team. |
| `text_mining.py` | For running locally or from a GitHub repo. |

### Option A — Google Colab (recommended)
1. Go to [colab.research.google.com](https://colab.research.google.com) → **File ▸ Upload notebook** → pick `MuTech_Phase3_Text_Mining.ipynb`.
2. Run the **Setup** cell (installs everything). If prompted, *Runtime ▸ Restart session* once, then continue.
3. Run the **Load Data** cell and upload your papers CSV.
4. Run the rest of the cells. Outputs download automatically at the end.

### Option B — Local script
```bash
pip install pandas numpy scipy scikit-learn spacy beautifulsoup4 lxml
python -m spacy download en_core_web_sm

python text_mining.py --input papers.csv --outdir phase3_output
# add --require-include to also drop papers with no music-related keyword
```

## What it does (matches the masterfile Phase 3 checklist)
1. **Load** the CSV (Scopus / Rayyan / Web of Science exports all auto-detected).
2. **Isolate metadata** — Title + Abstract + Keywords (author + index keywords merged).
3. **Clean** — strip HTML tags, lowercase, remove URLs / emails / numbers / punctuation.
4. **Relevance filter** — drop off-topic papers (the medical *MRI* ones) via editable
   `INCLUDE_KEYWORDS` / `EXCLUDE_KEYWORDS` lists. Every dropped paper is logged with a reason.
5. **Tokenize** with spaCy — lemmatize, remove stopwords / punctuation / short tokens,
   keep only content words (nouns, proper nouns, adjectives, verbs).
6. **Vectorize** — TF-IDF with unigrams + bigrams (e.g. *source separation*).

## Outputs (in `phase3_output/`)
| File | What it is |
|------|------------|
| `cleaned_corpus.csv` | One row per kept paper: raw, cleaned, and tokenized text + metadata. |
| `dropped_papers.csv` | Papers removed by the relevance filter, with the reason. |
| `tfidf_matrix.npz` | Sparse TF-IDF document-term matrix (the input to Phase 4). |
| `tfidf_features.txt` | The vocabulary (one term per line). |
| `tfidf_vectorizer.pkl` | The fitted vectorizer — reuse it in Phase 4. |
| `run_summary.txt` | Counts + settings for the run. |

## Handing off to Phase 4 (clustering)
```python
from scipy import sparse
import pickle

X = sparse.load_npz("phase3_output/tfidf_matrix.npz")          # documents x terms
vectorizer = pickle.load(open("phase3_output/tfidf_vectorizer.pkl", "rb"))
terms = vectorizer.get_feature_names_out()
# X -> KMeans / NMF / AgglomerativeClustering / LDA
```

## Tuning it for your data
Open the **Configuration** section (notebook) or the top of `text_mining.py`:
- **`EXCLUDE_KEYWORDS`** — add any other off-topic terms you spot (a paper with any of these is dropped).
- **`INCLUDE_KEYWORDS`** — the music/MIR vocabulary used to score relevance.
- **`REQUIRE_INCLUDE`** — set `True` to also drop papers that contain *no* music-related term.
- **`TFIDF_PARAMS`** — `max_features`, `min_df`, `max_df`, `ngram_range` if the vocabulary is too big/small.
- **`KEEP_POS` / `EXTRA_STOPWORDS`** — control which words survive tokenization.

> Note: `min_df=2` means a term must appear in at least 2 papers. On a very small
> test CSV that filters aggressively; on the real ~100-paper corpus it's appropriate.
