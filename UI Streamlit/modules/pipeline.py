"""
Pipeline module for corn leaf disease classification.

This module combines preprocessing, segmentation, and feature extraction
into a single prediction pipeline using the trained XGBoost model.
"""

import os
import joblib
import numpy as np
from PIL import Image

from .preprocessing import preprocess_pil_image
from .segmentation import segment_otsu
from .feature_extraction import extract_features
from .utils import CLASS_MAP


# Global model cache
_model = None


def load_model():
    """
    Load the trained XGBoost model from the model directory.

    Returns:
        Trained XGBoost model
    """
    global _model
    if _model is None:
        # Get the path to the model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "..", "model", "xgb_best_model.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        _model = joblib.load(model_path)
    
    return _model


def predict_image(pil_image):
    """
    Complete prediction pipeline for a single image.

    Pipeline steps:
    1. Preprocess image (resize, normalize, grayscale)
    2. Segment using Otsu thresholding
    3. Extract features (Fine + Coarse + DOR = 313 dimensions)
    4. Predict using trained XGBoost model

    Args:
        pil_image: PIL Image object

    Returns:
        tuple: (pred_class, probabilities, segmentation_image)
            - pred_class: Predicted class name (string)
            - probabilities: Probability for each class (numpy array, shape [4,])
            - segmentation_image: Otsu segmentation result (numpy array)
    """
    # Step 1: Preprocess
    img_rgb_norm, gray = preprocess_pil_image(pil_image)
    
    # Step 2: Segment
    segmentation = segment_otsu(gray)
    
    # Step 3: Extract features
    features = extract_features(gray)
    
    # Step 4: Load model and predict
    model = load_model()
    
    # Reshape features for prediction (1, 313)
    features = features.reshape(1, -1)
    
    # Get prediction and probabilities
    pred_idx = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Map prediction index to class name
    pred_class = CLASS_MAP[pred_idx]
    
    return pred_class, probabilities, segmentation


def get_class_names():
    """
    Get the list of class names.

    Returns:
        list: List of class names in order
    """
    return CLASS_MAP.copy()
