# app/web/main.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from sqlalchemy import text
from app.config import DATABASE_URL
from app.db.session import get_engine
from app.db.create_db import main as create_tables
from app.db.seed_data import main as seed_sample_data

st.set_page_config(page_title="AI Returns & Claims Hub", layout="wide")
st.title("AI Returns & Claims Hub — UI Skeleton")

# DB ping
try:
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    st.success("DB connectivity OK ✅")
except Exception as e:
    st.error(f"DB connection failed: {e}")

# --- Controls ---
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Create tables"):
        try:
            create_tables()
            st.success("Tables created.")
        except Exception as e:
            st.error(f"Create tables failed: {e}")

with col2:
    if st.button("Seed sample data"):
        try:
            seed_sample_data()
            st.success("Sample data seeded.")
        except Exception as e:
            st.error(f"Seeding failed: {e}")

with col3:
    refresh = st.button("Refresh view")

st.divider()

# --- Data viewers ---
def read_df(sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(text(sql), conn)

tabs = st.tabs(["Customers", "Products", "Claims"])

with tabs[0]:
    try:
        df = read_df("SELECT * FROM customers ORDER BY id")
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.info("Click 'Create tables' then 'Seed sample data' above.")

with tabs[1]:
    try:
        df = read_df("SELECT * FROM products ORDER BY id")
        st.dataframe(df, use_container_width=True)
    except Exception:
        st.info("Click 'Create tables' then 'Seed sample data' above.")

with tabs[2]:
    try:
        sql = """
        SELECT c.id, cu.name AS customer, p.name AS product, c.status,
               c.sentiment_score, c.predicted_refund_prob,
               c.is_photo_attached, c.created_at
        FROM claims c
        JOIN customers cu ON cu.id = c.customer_id
        JOIN products  p  ON p.id = c.product_id
        ORDER BY c.id DESC
        """
        df = read_df(sql)
        st.dataframe(df, use_container_width=True)
    except Exception:
        st.info("Click 'Create tables' then 'Seed sample data' above.")
