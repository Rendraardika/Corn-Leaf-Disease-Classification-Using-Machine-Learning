"""
Pipeline module for corn leaf disease classification.

This module combines preprocessing, segmentation, and feature extraction
into a single prediction pipeline using the trained XGBoost model.
"""

import os
import joblib
import numpy as np

from .preprocessing import preprocess_pil_image
from .segmentation import segment_otsu
from .feature_extraction import extract_features
from .utils import CLASS_MAP

# Cache model agar tidak load berulang
_model = None


def load_model():
    """Load trained XGBoost model."""
    global _model
    if _model is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "..", "model", "xgb_best_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        _model = joblib.load(model_path)

    return _model


def predict_image(pil_image):
    """
    Complete prediction pipeline.

    Returns:
        pred_class (str)
        probabilities (np.ndarray)
        segmentation (np.ndarray)
        confidence (float)
    """

    # 1. Preprocessing
    _, gray = preprocess_pil_image(pil_image)

    # 2. Segmentasi Otsu
    segmentation = segment_otsu(gray)

    # 3. Ekstraksi fitur (313 dimensi)
    features = extract_features(gray).reshape(1, -1)

    # 4. Prediksi
    model = load_model()
    probabilities = model.predict_proba(features)[0]
    pred_idx = int(np.argmax(probabilities))

    pred_class = CLASS_MAP[pred_idx]
    confidence = float(np.max(probabilities))

    return pred_class, probabilities, segmentation, confidence


def get_class_names():
    """Return list of class names."""
    return CLASS_MAP.copy()
