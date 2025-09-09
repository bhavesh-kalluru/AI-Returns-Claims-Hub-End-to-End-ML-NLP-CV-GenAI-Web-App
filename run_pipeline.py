# app/nlp/pipeline.py
from typing import List
from app.db.session import get_session
from app.models.schema import Claim
from app.db.migrations import ensure_claims_nlp_columns
from .text_models import classify_issue, sentiment_compound, extract_keywords_tfidf

def run_pipeline_all_claims() -> int:
    """Runs classification + sentiment + keyphrases over all claims, writes to DB."""
    ensure_claims_nlp_columns()

    s = get_session()
    try:
        claims: List[Claim] = s.query(Claim).order_by(Claim.id).all()
        if not claims:
            return 0

        texts = [c.description or "" for c in claims]
        kw_lists = extract_keywords_tfidf(texts, top_k=5)

        for c, kws in zip(claims, kw_lists):
            label, _ = classify_issue(c.description or "")
            sent = sentiment_compound(c.description or "")
            c.issue_label = label
            c.sentiment_score = sent
            c.key_phrases = ", ".join(kws)

        s.commit()
        return len(claims)
    finally:
        s.close()
