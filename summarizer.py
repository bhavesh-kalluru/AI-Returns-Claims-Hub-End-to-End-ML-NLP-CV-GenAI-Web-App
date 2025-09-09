# app/genai/summarizer.py
from __future__ import annotations
import json
from typing import Tuple
from openai import OpenAI
from app.config import OPENAI_API_KEY, os

def _get_model() -> str:
    # allow override via .env, default to a compact, low-latency model
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _extract_json(text: str) -> dict:
    """
    Try to parse a JSON blob even if it’s inside a Markdown code block.
    Fallback to a minimal structure if parsing fails.
    """
    try:
        return json.loads(text)
    except Exception:
        # try code fence
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {"summary": text.strip()[:900], "reply": ""}

def draft_summary_and_reply(
    claim_text: str,
    customer_name: str,
    product_name: str,
    claim_status: str,
) -> Tuple[str, str, str]:
    """
    Returns (summary, reply, model_used)
    """
    if not OPENAI_API_KEY:
        # No key: deterministic, offline fallback
        summary = f"{customer_name} reports an issue with {product_name}. Details: {claim_text[:400]}"
        reply = (
            f"Hi {customer_name.split(' ')[0]},\n\n"
            "Thanks for contacting us. We’re sorry for the trouble with your item. "
            "We’ve logged your case and will follow up shortly.\n\n"
            "Regards,\nSupport Team"
        )
        return summary, reply, "offline-fallback"

    client = OpenAI(api_key=OPENAI_API_KEY)
    model = _get_model()

    system = (
        "You are an expert customer support assistant for e-commerce returns/claims. "
        "Read the claim text and produce a concise internal summary and a courteous customer reply. "
        "Keep the reply short, apologetic when appropriate, and ask for a photo if not already attached."
    )
    user = (
        f"Customer: {customer_name}\n"
        f"Product: {product_name}\n"
        f"Current Status: {claim_status}\n\n"
        f"Claim Text:\n{claim_text}\n\n"
        "Return a JSON object with keys exactly: summary, reply. "
        "summary = 2–4 bullet points. reply = short email (no HTML)."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        txt = resp.choices[0].message.content or ""
        data = _extract_json(txt)
        summary = data.get("summary", "")[:3000]
        reply = data.get("reply", "")[:3000]
        if not summary:
            summary = txt[:1000]
        return summary, reply, model
    except Exception as e:
        # Robust fallback
        summary = f"(fallback) {customer_name} issue with {product_name}. Details: {claim_text[:400]}"
        reply = (
            f"Hi {customer_name.split(' ')[0]},\n\n"
            "Thanks for reaching out. We’re looking into this and will update you soon.\n\n"
            "Regards,\nSupport Team"
        )
        return summary, reply, f"error-fallback: {type(e).__name__}"
