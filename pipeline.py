from __future__ import annotations
from typing import Tuple
from sqlalchemy import select
from app.db.session import get_session
from app.models.schema import Claim
from app.utils.logger import get_logger
from app.nlp.text_models import classify_issue, sentiment_label

logger = get_logger()

def analyze_text(text: str | None) -> Tuple[str, str]:
    """Return (issue_label, sentiment_label) for a claim description."""
    return classify_issue(text), sentiment_label(text)

def process_pending_claims(batch_size: int = 100) -> dict:
    """
    Process claims with status in ('new','investigating').
    - Always (re)compute sentiment_label if empty or different.
    - Update issue_category only if it's empty/unknown. If it differs, log a note.
    """
    sess = get_session()
    updated = 0
    skipped = 0
    examined = 0
    try:
        stmt = select(Claim).where(Claim.status.in_(["new", "investigating"]))
        claims = sess.scalars(stmt).all()
        for claim in claims[:batch_size]:
            examined += 1
            pred_issue, pred_sent = analyze_text(claim.description_text)

            changed = False

            # Update sentiment if missing or different
            if not claim.sentiment_label or claim.sentiment_label != pred_sent:
                claim.sentiment_label = pred_sent
                changed = True

            # Update issue_category only if it's unset/unknown
            curr_issue = (claim.issue_category or "unknown").lower()
            if curr_issue in ("", "unknown", None):
                if pred_issue and pred_issue != "unknown":
                    claim.issue_category = pred_issue
                    changed = True
            else:
                # If our prediction disagrees, just log for now (don’t overwrite)
                if pred_issue != "unknown" and pred_issue != curr_issue:
                    logger.info(
                        f"Claim#{claim.id}: kept issue_category='{curr_issue}', "
                        f"pred='{pred_issue}'"
                    )

            if changed:
                sess.add(claim)
                updated += 1
            else:
                skipped += 1

        sess.commit()
        logger.info(f"Processed claims: examined={examined}, updated={updated}, skipped={skipped}")
        return {"examined": examined, "updated": updated, "skipped": skipped}
    except Exception as e:
        sess.rollback()
        logger.exception(f"❌ NLP pipeline failed: {e}")
        raise
    finally:
        sess.close()
