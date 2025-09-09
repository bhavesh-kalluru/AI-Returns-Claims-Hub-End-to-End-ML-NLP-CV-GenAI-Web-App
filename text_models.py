# app/nlp/text_models.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

CATEGORIES = {
    "Audio Issue":    ["crackle", "anc", "noise", "sound", "mic", "earcup", "left side"],
    "Display Defect": ["dead pixel", "screen", "display", "ghosting", "stuck pixel", "flicker"],
    "Build Quality":  ["stitch", "stitching", "came off", "tear", "broken", "loose", "crack", "scratched", "defect"],
    "Battery/Power":  ["battery", "charge", "charging", "power", "drain", "won't turn on", "doesn't turn on"],
    "Shipping Damage":["arrived damaged", "box damaged", "dent", "shipping", "courier", "delivery", "package"],
    "Size/Fit":       ["size", "fit", "too large", "too small", "tight", "loose"],
}

def _norm(s: str) -> str:
    return (s or "").lower()

def classify_issue(text: str) -> Tuple[str, float]:
    """Simple keyword vote classifier -> (label, confidence)."""
    t = _norm(text)
    best_label, best_score = "Other", 0
    for label, kws in CATEGORIES.items():
        score = sum(1 for kw in kws if kw in t)
        if score > best_score:
            best_label, best_score = label, score
    conf = (0.4 + 0.2 * best_score) if best_score > 0 else 0.3
    return best_label, float(min(conf, 1.0))

def sentiment_compound(text: str) -> float:
    if not text:
        return 0.0
    return float(_analyzer.polarity_scores(text)["compound"])

def extract_keywords_tfidf(docs: List[str], top_k: int = 5) -> List[List[str]]:
    """Top TF-IDF unigrams/bigrams per doc."""
    corpus = [_norm(d) for d in docs]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=3000)
    X = vec.fit_transform(corpus)
    vocab = np.array(vec.get_feature_names_out())
    out: List[List[str]] = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            out.append([])
            continue
        arr = row.toarray()[0]
        idx = np.argsort(-arr)[:top_k]
        terms = [vocab[j] for j in idx if arr[j] > 0]
        out.append(terms)
    return out
