# app/utils/powerbi_export.py
from pathlib import Path
from typing import Dict
import pandas as pd
from sqlalchemy import text
from app.db.session import get_engine

EXPORT_DIR = Path(__file__).resolve().parents[2] / "app" / "data" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

def _read(sql: str) -> pd.DataFrame:
    eng = get_engine()
    with eng.connect() as conn:
        return pd.read_sql_query(text(sql), conn)

def export_powerbi_csvs() -> Dict[str, Path]:
    """
    Writes CSVs for customers, products, claims, and v_claims_facts (if present).
    Returns mapping {name: path}.
    """
    out: Dict[str, Path] = {}

    # base tables
    df_customers = _read("SELECT * FROM customers")
    df_products  = _read("SELECT * FROM products")
    df_claims    = _read("SELECT * FROM claims")

    paths = {
        "customers": EXPORT_DIR / "customers.csv",
        "products":  EXPORT_DIR / "products.csv",
        "claims":    EXPORT_DIR / "claims.csv",
    }
    df_customers.to_csv(paths["customers"], index=False)
    df_products.to_csv(paths["products"], index=False)
    df_claims.to_csv(paths["claims"], index=False)
    out.update(paths)

    # analytics view (if exists)
    try:
        df_v = _read("SELECT * FROM v_claims_facts")
        p = EXPORT_DIR / "v_claims_facts.csv"
        df_v.to_csv(p, index=False)
        out["v_claims_facts"] = p
    except Exception:
        # view missing—skip silently
        pass

    return out
