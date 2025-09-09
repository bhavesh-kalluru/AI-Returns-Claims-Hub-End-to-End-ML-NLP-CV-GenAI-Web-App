# app/cnn/image_checks.py
from io import BytesIO
from typing import Dict, Any
import numpy as np
from PIL import Image
import cv2

def _to_cv2(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)  # RGB
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def analyze_image(file_bytes: bytes) -> Dict[str, Any]:
    """
    Returns: dict with blur, brightness, contrast, edge_density, damage_score [0..1], quality_ok bool.
    """
    img = Image.open(BytesIO(file_bytes))
    cv = _to_cv2(img)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)

    # Blur (higher = sharper). Typical good > ~100; low => blurry.
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Brightness & contrast
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Edge density (rough damage cue if too many strong edges)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    # Heuristic damage score (0..1)
    # Normalize components roughly into [0..1], then combine
    blur_norm = min(blur / 200.0, 1.0)          # >200 considered sharp
    bright_norm = 1.0 - abs(brightness - 128) / 128.0  # centered around mid
    contrast_norm = min(contrast / 64.0, 1.0)
    edge_norm = min(edge_density / 0.15, 1.0)   # >0.15 is quite edgy

    # Good quality if reasonably sharp & visible
    quality_ok = (blur > 80) and (40 < brightness < 210)

    # Damage tends to correlate with high edges + high contrast
    damage_score = float(0.6 * edge_norm + 0.3 * contrast_norm + 0.1 * (1 - blur_norm))
    damage_score = max(0.0, min(1.0, damage_score))

    return {
        "width": int(img.width),
        "height": int(img.height),
        "blur": blur,
        "brightness": brightness,
        "contrast": contrast,
        "edge_density": edge_density,
        "damage_score": damage_score,
        "quality_ok": bool(quality_ok),
    }
