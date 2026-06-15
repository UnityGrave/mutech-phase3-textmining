"""
MuTech Project - Phase 3: NLP Text Mining & Preprocessing
==========================================================
Owner: Kean | Systematic Literature Review on Intelligent Music Processing (MIR)

WHAT THIS DOES (per the masterfile Phase 3 checklist)
-----------------------------------------------------
1. Load the screened papers from a CSV (Scopus / Rayyan / Web of Science exports
   are all auto-detected) into a single pandas DataFrame.
2. Isolate the metadata we actually mine: Title + Abstract + Keywords.
3. Clean the text: strip HTML tags, lowercase everything, remove URLs / emails /
   numbers / punctuation, and normalize whitespace.
4. Relevance filter: automatically drop off-topic papers (e.g. medical MRI studies)
   using editable INCLUDE / EXCLUDE keyword lists.
5. Tokenize with spaCy: lemmatize, drop stopwords + punctuation + short tokens,
   and keep only meaningful parts of speech.
6. Vectorize the cleaned corpus into numerical TF-IDF vectors -> ready for Phase 4
   (K-Means / NMF / Agglomerative / LDA topic clustering).

OUTPUTS (written to --outdir, default ./phase3_output)
------------------------------------------------------
  cleaned_corpus.csv   - one row per kept paper, with raw + cleaned + tokenized text
  dropped_papers.csv   - papers removed by the relevance filter, with the reason
  tfidf_matrix.npz     - sparse TF-IDF document-term matrix (load with scipy)
  tfidf_features.txt   - the vocabulary (feature names), one term per line
  tfidf_vectorizer.pkl - the fitted TfidfVectorizer (reuse it in Phase 4)
  run_summary.txt      - counts + settings used for this run

HOW TO RUN
----------
  # one-time setup
  pip install pandas numpy scipy scikit-learn spacy beautifulsoup4 lxml
  python -m spacy download en_core_web_sm

  # run it
  python text_mining.py --input papers.csv --outdir phase3_output

When Marian's final filtered CSV is ready, just point --input at it. No code changes
needed - the column names are auto-detected.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIGURATION  --  edit these lists to match your inclusion/exclusion criteria
# ----------------------------------------------------------------------------

# Candidate column names from different databases -> our standard field name.
# Matching is case-insensitive. Add more aliases here if a new export uses them.
COLUMN_ALIASES: dict[str, list[str]] = {
    "title":    ["title", "ti", "document title", "article title", "topic"],
    "abstract": ["abstract", "ab", "summary", "description"],
    "keywords": ["author keywords", "index keywords", "keywords", "de", "id",
                 "keyword", "key words"],
    "year":     ["year", "py", "publication year", "date"],
    "authors":  ["authors", "author", "au", "author full names", "author(s)"],
    "doi":      ["doi", "di", "digital object identifier"],
    "source":   ["source title", "source", "journal", "publication name", "so"],
}

# Relevance filter -------------------------------------------------------------
# A paper is DROPPED if any EXCLUDE term appears in its text (medical / off-topic).
# If REQUIRE_INCLUDE is True, a paper is also dropped if NONE of the INCLUDE terms
# appear. Otherwise INCLUDE terms are only used to compute a relevance score.
INCLUDE_KEYWORDS: list[str] = [
    "music", "musical", "singing", "singer", "vocal", "voice", "choir", "choral",
    "ensemble", "instrument", "instrumental", "audio", "sound", "acoustic",
    "melody", "pitch", "harmony", "rhythm", "tempo", "timbre", "midi", "score",
    "music information retrieval", "mir", "source separation", "transcription",
    "onset", "chord", "genre", "polyphonic", "monophonic", "f0", "spectrogram",
    "performance",
]

EXCLUDE_KEYWORDS: list[str] = [
    # medical / biomedical imaging & health (the "MRI eme eme" papers)
    "mri", "magnetic resonance imaging", "fmri", "ct scan", "x-ray", "tumor",
    "cancer", "clinical", "patient", "disease", "diagnosis", "diagnostic",
    "biomedical", "electrocardiogram", "ecg", "eeg", "surgery", "surgical",
    "pharmaceutical", "drug", "gene", "protein", "molecular", "cardiac",
    # other clearly off-topic domains
    "seismic", "geology", "agriculture", "crop", "stock market", "cryptocurrency",
]

# spaCy preprocessing ----------------------------------------------------------
# Keep only these parts of speech (content words). Set to None to keep everything.
KEEP_POS: set[str] | None = {"NOUN", "PROPN", "ADJ", "VERB"}
MIN_TOKEN_LEN = 3            # drop tokens shorter than this after lemmatization
EXTRA_STOPWORDS: set[str] = {  # domain noise words that aren't useful as topics
    "study", "paper", "result", "method", "approach", "propose", "proposed",
    "based", "use", "used", "using", "show", "shows", "shown", "present",
    "presented", "research", "article", "model", "models", "system", "systems",
    "elsevier", "springer", "ieee", "acm", "copyright", "rights", "reserved",
    "abstract", "introduction", "conclusion",
}

# TF-IDF settings --------------------------------------------------------------
TFIDF_PARAMS = dict(
    max_features=2000,   # cap vocabulary size
    min_df=2,            # ignore terms in fewer than 2 docs
    max_df=0.90,         # ignore terms in more than 90% of docs (too generic)
    ngram_range=(1, 2),  # unigrams + bigrams (e.g. "source separation")
    sublinear_tf=True,
)


# ----------------------------------------------------------------------------
# 1. LOAD + HARMONIZE COLUMNS
# ----------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Read a CSV with sensible fallbacks for encoding and delimiter."""
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Input file not found: {path}")
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, sep=None, engine="python",
                             on_bad_lines="skip")
            print(f"[load] Read {len(df)} rows x {len(df.columns)} cols "
                  f"(encoding={enc})")
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    sys.exit("[ERROR] Could not parse the CSV with any known encoding/delimiter.")


def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map messy source columns to our standard field names.

    Author Keywords + Index Keywords are merged into a single 'keywords' field.
    """
    lower_map = {c.lower().strip(): c for c in df.columns}
    out = pd.DataFrame(index=df.index)

    for field, aliases in COLUMN_ALIASES.items():
        matched = [lower_map[a] for a in aliases if a in lower_map]
        if not matched:
            out[field] = ""
            continue
        if field == "keywords":
            # combine every keyword-like column (author + index keywords)
            combined = df[matched].fillna("").astype(str).agg("; ".join, axis=1)
            out[field] = combined
        else:
            out[field] = df[matched[0]].fillna("").astype(str)

    found = [f for f in COLUMN_ALIASES if (out[f].str.strip() != "").any()]
    print(f"[harmonize] Detected fields: {', '.join(found) or 'NONE'}")
    if "title" not in found and "abstract" not in found:
        print("[WARN] Neither title nor abstract detected - check column names!")
    return out


def build_document(row: pd.Series) -> str:
    """Join the mined metadata (title + abstract + keywords) into one string.

    Parts with no actual letters (e.g. a stray "; " left over from joining empty
    keyword columns) are ignored so genuinely empty rows resolve to "".
    """
    parts = [str(row.get("title", "")), str(row.get("abstract", "")),
             str(row.get("keywords", ""))]
    return " ".join(p for p in parts
                    if p and p.lower() != "nan" and re.search(r"[a-zA-Z]", p))


# ----------------------------------------------------------------------------
# 2. CLEANING
# ----------------------------------------------------------------------------

_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"http\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_NONALPHA_RE = re.compile(r"[^a-z\s]")
_WS_RE = re.compile(r"\s+")

try:
    from bs4 import BeautifulSoup  # nicer HTML entity handling if available
    _HAS_BS4 = True
except Exception:
    _HAS_BS4 = False


def clean_text(text: str) -> str:
    """Strip HTML, lowercase, and remove URLs/emails/numbers/punctuation."""
    if not isinstance(text, str):
        return ""
    if _HAS_BS4 and "<" in text:
        text = BeautifulSoup(text, "lxml").get_text(separator=" ")
    else:
        text = _HTML_RE.sub(" ", text)
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _NONALPHA_RE.sub(" ", text)   # keep only letters + spaces
    text = _WS_RE.sub(" ", text).strip()
    return text


# ----------------------------------------------------------------------------
# 3. RELEVANCE FILTER
# ----------------------------------------------------------------------------

def _compile_terms(terms: Iterable[str]) -> list[tuple[str, re.Pattern]]:
    """Word-boundary patterns so 'mir' doesn't match 'admire'."""
    pats = []
    for t in terms:
        t = t.strip().lower()
        if not t:
            continue
        pats.append((t, re.compile(r"\b" + re.escape(t) + r"\b")))
    return pats


def relevance_filter(df: pd.DataFrame, text_col: str,
                     require_include: bool = False) -> pd.DataFrame:
    """Add keep/drop decision + reason + relevance_score columns."""
    inc = _compile_terms(INCLUDE_KEYWORDS)
    exc = _compile_terms(EXCLUDE_KEYWORDS)

    keep, reason, score = [], [], []
    for txt in df[text_col]:
        txt = txt or ""
        hit_exc = [t for t, p in exc if p.search(txt)]
        hit_inc = [t for t, p in inc if p.search(txt)]
        if hit_exc:
            keep.append(False)
            reason.append("excluded term: " + ", ".join(hit_exc[:3]))
        elif require_include and not hit_inc:
            keep.append(False)
            reason.append("no relevant (include) term found")
        else:
            keep.append(True)
            reason.append("")
        score.append(len(hit_inc))

    df = df.copy()
    df["relevance_score"] = score
    df["keep"] = keep
    df["drop_reason"] = reason
    return df


# ----------------------------------------------------------------------------
# 4. spaCy TOKENIZATION + LEMMATIZATION
# ----------------------------------------------------------------------------

def load_spacy():
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        sys.exit("[ERROR] spaCy model missing. Run:\n"
                 "        python -m spacy download en_core_web_sm")
    for w in EXTRA_STOPWORDS:
        nlp.vocab[w].is_stop = True
    return nlp


def tokenize_corpus(texts: list[str], nlp) -> list[str]:
    """Lemmatize, drop stopwords/punct/short tokens, optional POS filter."""
    out = []
    for doc in nlp.pipe(texts, batch_size=64):
        toks = []
        for tok in doc:
            if tok.is_stop or tok.is_punct or tok.is_space:
                continue
            lemma = tok.lemma_.lower().strip()
            if len(lemma) < MIN_TOKEN_LEN or lemma in EXTRA_STOPWORDS:
                continue
            if KEEP_POS is not None and tok.pos_ not in KEEP_POS:
                continue
            toks.append(lemma)
        out.append(" ".join(toks))
    return out


# ----------------------------------------------------------------------------
# 5. TF-IDF VECTORIZATION
# ----------------------------------------------------------------------------

def vectorize(tokenized: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    matrix = vec.fit_transform(tokenized)
    print(f"[tfidf] Matrix shape: {matrix.shape[0]} docs x "
          f"{matrix.shape[1]} terms")
    return vec, matrix


# ----------------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------------

def run(input_path: str, outdir: str, require_include: bool = False) -> None:
    os.makedirs(outdir, exist_ok=True)

    df_raw = load_data(input_path)
    meta = harmonize_columns(df_raw)
    meta["document_raw"] = meta.apply(build_document, axis=1)

    # drop rows with no usable text at all
    before = len(meta)
    meta = meta[meta["document_raw"].str.strip() != ""].reset_index(drop=True)
    print(f"[clean] Dropped {before - len(meta)} rows with no title/abstract/keywords")

    meta["document_clean"] = meta["document_raw"].map(clean_text)

    # relevance filter
    meta = relevance_filter(meta, "document_clean", require_include=require_include)
    dropped = meta[~meta["keep"]].copy()
    kept = meta[meta["keep"]].reset_index(drop=True)
    print(f"[filter] Kept {len(kept)} papers | Dropped {len(dropped)} off-topic")

    if kept.empty:
        sys.exit("[ERROR] No papers left after filtering - loosen your keyword lists.")

    # spaCy tokenization
    print("[spacy] Tokenizing + lemmatizing ...")
    nlp = load_spacy()
    kept["tokens"] = tokenize_corpus(kept["document_clean"].tolist(), nlp)
    kept = kept[kept["tokens"].str.strip() != ""].reset_index(drop=True)

    # TF-IDF
    vectorizer, matrix = vectorize(kept["tokens"].tolist())

    # ---- save everything ----
    from scipy import sparse
    kept.to_csv(os.path.join(outdir, "cleaned_corpus.csv"), index=False)
    dropped.to_csv(os.path.join(outdir, "dropped_papers.csv"), index=False)
    sparse.save_npz(os.path.join(outdir, "tfidf_matrix.npz"), matrix)
    with open(os.path.join(outdir, "tfidf_features.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(vectorizer.get_feature_names_out()))
    with open(os.path.join(outdir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    summary = (
        "MuTech Phase 3 - Text Mining run summary\n"
        f"  input file        : {input_path}\n"
        f"  rows read         : {len(df_raw)}\n"
        f"  papers kept       : {len(kept)}\n"
        f"  papers dropped    : {len(dropped)}\n"
        f"  TF-IDF matrix     : {matrix.shape[0]} docs x {matrix.shape[1]} terms\n"
        f"  require_include   : {require_include}\n"
        f"  ngram_range       : {TFIDF_PARAMS['ngram_range']}\n"
    )
    with open(os.path.join(outdir, "run_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    print("\n" + summary)
    print(f"[done] All outputs written to: {outdir}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="MuTech Phase 3 - NLP Text Mining & Preprocessing")
    p.add_argument("--input", "-i", required=True, help="path to papers CSV")
    p.add_argument("--outdir", "-o", default="phase3_output",
                   help="output directory (default: phase3_output)")
    p.add_argument("--require-include", action="store_true",
                   help="also drop papers that contain no INCLUDE keyword")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.outdir, require_include=args.require_include)
