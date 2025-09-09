# app/ann/refund_predictor.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import json

import numpy as np
from joblib import dump, load
from sklearn.neural_network import MLPRegressor

from app.db.session import get_session
from app.models.schema import Claim

MODELS_DIR = Path(__file__).resolve().parents[2] / "app" / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "refund_mlp.joblib"

ISSUE_MAP = {
    "Audio Issue": 0,
    "Display Defect": 1,
    "Build Quality": 2,
    "Battery/Power": 3,
    "Shipping Damage": 4,
    "Size/Fit": 5,
    "Other": 6,
}

def _encode_issue(label: Optional[str]) -> int:
    return ISSUE_MAP.get(label or "Other", 6)

def _extract_features(c: Claim) -> np.ndarray:
    x = [
        float(c.sentiment_score or 0.0),
        1.0 if c.is_photo_attached else 0.0,
        float(c.damage_score or 0.0),
        float(_encode_issue(c.issue_label)),
    ]
    return np.array(x, dtype=float)

def _build_training_data(claims: List[Claim]) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for c in claims:
        if c.predicted_refund_prob is not None:
            X.append(_extract_features(c))
            y.append(float(c.predicted_refund_prob))
    return np.array(X, dtype=float), np.array(y, dtype=float)

def _synthetic_training_data(n: int = 300, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # sentiment [-1,1], photo {0,1}, damage [0,1], issue_cat {0..6}
    sent = rng.uniform(-1.0, 1.0, size=n)
    photo = rng.integers(0, 2, size=n).astype(float)
    damage = rng.beta(2, 4, size=n)  # skewed low
    issue = rng.integers(0, 7, size=n).astype(float)

    # Heuristic: defects + negative sentiment + photo + higher damage -> higher refund prob
    base = 0.25 + 0.25 * (issue == ISSUE_MAP["Display Defect"]) + 0.25 * (issue == ISSUE_MAP["Build Quality"]) \
           + 0.25 * (issue == ISSUE_MAP["Shipping Damage"])
    y = base + 0.2 * (damage) + 0.2 * photo + 0.15 * (-np.clip(sent, -1, 0))
    y = np.clip(y, 0.0, 1.0)

    X = np.vstack([sent, photo, damage, issue]).T
    return X, y

def train_or_load_model(claims: List[Claim]) -> MLPRegressor:
    if MODEL_PATH.exists():
        return load(MODEL_PATH)

    X_labeled, y_labeled = _build_training_data(claims)
    if X_labeled.shape[0] < 10:
        X, y = _synthetic_training_data()
    else:
        X, y = X_labeled, y_labeled

    model = MLPRegressor(hidden_layer_sizes=(16, 8), activation="relu", random_state=42, max_iter=600)
    model.fit(X, y)
    dump(model, MODEL_PATH)
    return model

def run_ann_predictor_on_claims() -> int:
    s = get_session()
    try:
        claims: List[Claim] = s.query(Claim).order_by(Claim.id).all()
        if not claims:
            return 0
        model = train_or_load_model(claims)
        updated = 0
        for c in claims:
            x = _extract_features(c).reshape(1, -1)
            yhat = float(model.predict(x)[0])
            c.predicted_refund_prob = float(np.clip(yhat, 0.0, 1.0))
            updated += 1
        s.commit()
        return updated
    finally:
        s.close()
