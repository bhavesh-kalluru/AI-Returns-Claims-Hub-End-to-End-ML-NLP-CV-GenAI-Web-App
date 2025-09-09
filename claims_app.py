# app/web/claims_app.py
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from sqlalchemy import text

from app.config import DATABASE_URL
from app.db.session import get_engine, get_session
from app.db.create_db import main as create_tables
from app.db.seed_data import main as seed_sample_data

# ---- Optional modules (safe fallbacks) ----
try:
    from app.nlp.run_pipeline import run_nlp_on_claims
except Exception:
    def run_nlp_on_claims() -> int: return 0

try:
    from app.cnn.image_checks import analyze_image
except Exception:
    analyze_image = None

try:
    from app.ann.refund_predictor import run_ann_predictor_on_claims
except Exception:
    def run_ann_predictor_on_claims() -> int: return 0

try:
    from app.genai.summarizer import draft_summary_and_reply
except Exception:
    draft_summary_and_reply = None

st.set_page_config(page_title="AI Returns & Claims Hub", layout="wide")
st.title("AI Returns & Claims Hub")

engine = get_engine()

# --- Quick DB ping ---
try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    st.success("DB connectivity OK ✅")
except Exception as e:
    st.error(f"DB connection failed: {e}")

# --- Inline migrations (idempotent) ---
def ensure_claims_all_columns() -> None:
    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info(claims)")).fetchall()
        cols = {r[1] for r in rows}
        def add(col, ddl):
            if col not in cols:
                conn.execute(text(f"ALTER TABLE claims ADD COLUMN {ddl}"))
                cols.add(col)
        # Previously added
        add("sentiment_score", "sentiment_score REAL")
        add("predicted_refund_prob", "predicted_refund_prob REAL")
        add("is_photo_attached", "is_photo_attached INTEGER DEFAULT 0")
        add("issue_label", "issue_label TEXT")
        add("key_phrases", "key_phrases TEXT")
        add("photo_path", "photo_path TEXT")
        add("photo_blur", "photo_blur REAL")
        add("photo_brightness", "photo_brightness REAL")
        add("photo_contrast", "photo_contrast REAL")
        add("damage_score", "damage_score REAL")
        # NEW — GenAI outputs
        add("ai_summary", "ai_summary TEXT")
        add("ai_reply",   "ai_reply TEXT")
        add("ai_model",   "ai_model TEXT")

# ---- Controls/top buttons ----
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
with c1:
    if st.button("Create tables"):
        try: create_tables(); st.success("Tables created.")
        except Exception as e: st.error(f"Create tables failed: {e}")
with c2:
    if st.button("Fix / Migrate DB"):
        try: ensure_claims_all_columns(); st.success("DB migrated.")
        except Exception as e: st.error(f"Migration failed: {e}")
with c3:
    if st.button("Seed sample data"):
        try: seed_sample_data(); st.success("Sample data seeded.")
        except Exception as e: st.error(f"Seeding failed: {e}")
with c4:
    if st.button("Run NLP"):
        try: ensure_claims_all_columns(); n = run_nlp_on_claims(); st.success(f"NLP updated {n} claims.")
        except Exception as e: st.error(f"NLP failed: {e}")
with c5:
    if st.button("Run ANN Predictor"):
        try: ensure_claims_all_columns(); n = run_ann_predictor_on_claims(); st.success(f"ANN predicted for {n} claims.")
        except Exception as e: st.error(f"ANN failed: {e}")
with c6:
    refresh = st.button("Refresh")
with c7:
    st.write("")  # spacer

st.divider()

# ---- Sidebar actions: Photo & GenAI ----
with st.sidebar:
    st.header("Attach & Analyze Photo")
    claim_id = st.number_input("Claim ID", min_value=1, step=1, value=1)
    photo = st.file_uploader("Photo (jpg/png)", type=["jpg","jpeg","png"])
    if st.button("Analyze & Save Photo"):
        if not photo:
            st.warning("Please upload a photo.")
        elif analyze_image is None:
            st.error("Vision module not installed (Step 4).")
        else:
            try:
                ensure_claims_all_columns()
                stats = analyze_image(photo.read())
                updir = ROOT / "app" / "data" / "uploads"
                updir.mkdir(parents=True, exist_ok=True)
                save_path = updir / f"claim_{int(claim_id)}_{photo.name}"
                with open(save_path,"wb") as f: f.write(photo.getbuffer())

                from app.db.session import get_session
                from app.models.schema import Claim
                s = get_session()
                try:
                    c = s.query(Claim).filter(Claim.id==int(claim_id)).one_or_none()
                    if not c: st.error(f"No claim with id={int(claim_id)}")
                    else:
                        c.is_photo_attached = True
                        c.photo_path = str(save_path)
                        c.photo_blur = stats["blur"]
                        c.photo_brightness = stats["brightness"]
                        c.photo_contrast = stats["contrast"]
                        c.damage_score = stats["damage_score"]
                        s.commit()
                        st.success(f"Saved. damage_score={stats['damage_score']:.2f}")
                finally: s.close()
            except Exception as e:
                st.error(f"Photo analysis failed: {e}")

    st.divider()
    st.header("GenAI: Summary + Reply")
    claim_id_ai = st.number_input("Claim ID for GenAI", min_value=1, step=1, value=1, key="genai_id")
    if st.button("Generate for Claim"):
        if draft_summary_and_reply is None:
            st.error("GenAI module not available.")
        else:
            try:
                ensure_claims_all_columns()
                from app.models.schema import Claim, Customer, Product
                s = get_session()
                try:
                    c = s.query(Claim).filter(Claim.id==int(claim_id_ai)).one_or_none()
                    if not c:
                        st.error(f"No claim with id={int(claim_id_ai)}")
                    else:
                        cust = s.query(Customer).filter(Customer.id==c.customer_id).one()
                        prod = s.query(Product).filter(Product.id==c.product_id).one()
                        summary, reply, used_model = draft_summary_and_reply(
                            claim_text=c.description,
                            customer_name=cust.name,
                            product_name=prod.name,
                            claim_status=c.status,
                        )
                        c.ai_summary = summary
                        c.ai_reply   = reply
                        c.ai_model   = used_model
                        s.commit()
                        st.success(f"GenAI updated claim {c.id} (model={used_model}).")
                        with st.expander("Preview Summary"):
                            st.write(summary or "(empty)")
                        with st.expander("Preview Reply"):
                            st.write(reply or "(empty)")
                finally:
                    s.close()
            except Exception as e:
                st.error(f"GenAI failed: {e}")

st.divider()

# ---- Helpers ----
def read_df(sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(text(sql), conn)

def table_exists(name: str) -> bool:
    with engine.connect() as conn:
        return bool(conn.execute(text(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=:n"
        ), {"n": name}).fetchone())

def get_columns(table: str):
    with engine.connect() as conn:
        return [r[1] for r in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()]

# ---- Tabs ----
tabs = st.tabs(["Customers", "Products", "Claims", "DB Status"])

with tabs[0]:
    if table_exists("customers"):
        st.dataframe(read_df("SELECT * FROM customers ORDER BY id"), use_container_width=True)
    else:
        st.info("Click **Create tables**, then **Seed sample data**.")

with tabs[1]:
    if table_exists("products"):
        st.dataframe(read_df("SELECT * FROM products ORDER BY id"), use_container_width=True)
    else:
        st.info("Click **Create tables**, then **Seed sample data**.")

with tabs[2]:
    if table_exists("claims"):
        cols = get_columns("claims")
        select_cols = ["c.id","cu.name AS customer","p.name AS product","c.status","c.created_at"]
        # Optional columns if present
        if "issue_label" in cols:           select_cols.insert(4, "c.issue_label")
        if "key_phrases" in cols:           select_cols.insert(5, "c.key_phrases")
        if "sentiment_score" in cols:       select_cols.append("c.sentiment_score")
        if "predicted_refund_prob" in cols: select_cols.append("c.predicted_refund_prob")
        if "is_photo_attached" in cols:     select_cols.append("c.is_photo_attached")
        if "damage_score" in cols:          select_cols.append("c.damage_score")
        if "photo_blur" in cols:            select_cols.append("c.photo_blur")
        if "photo_brightness" in cols:      select_cols.append("c.photo_brightness")
        if "photo_contrast" in cols:        select_cols.append("c.photo_contrast")
        if "ai_model" in cols:              select_cols.append("c.ai_model")
        if "ai_summary" in cols:            select_cols.append("c.ai_summary")
        if "ai_reply" in cols:              select_cols.append("c.ai_reply")

        sql = f"""
        SELECT {', '.join(select_cols)}
        FROM claims c
        JOIN customers cu ON cu.id=c.customer_id
        JOIN products  p  ON p.id=c.product_id
        ORDER BY c.id DESC
        """
        df = read_df(sql)
        if df.empty:
            st.warning("Claims table is empty. Click **Seed sample data**.")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Click **Create tables**, then **Seed sample data**.")

with tabs[3]:
    st.subheader("Database status")
    st.code(f"DATABASE_URL = {DATABASE_URL}", language="bash")
    try:
        with engine.connect() as conn:
            tables = [r[0] for r in conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )).fetchall()]
            counts = {}
            for t in tables:
                try:
                    counts[t] = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
                except Exception:
                    counts[t] = "n/a"
        st.write("**Tables & row counts:**")
        st.json(counts)
        if not tables:
            st.warning("No tables found. Click **Create tables** above.")
    except Exception as e:
        st.error(f"Status check failed: {e}")
