"""
Preprocessing module for corn leaf disease classification.

This module contains functions for image preprocessing including:
- Resize to target size (256x256)
- BGR to RGB conversion
- Normalization to [0, 1]
- Grayscale conversion
"""

import cv2
import numpy as np


def preprocess_image(img, target_size=(256, 256)):
    """
    Preprocessing citra daun jagung.

    Tahapan:
    1. Resize ke ukuran target (default 256x256, sejalan dengan PlantVillage).
    2. BGR (OpenCV) -> RGB.
    3. Normalisasi piksel ke [0, 1].
    4. Konversi ke grayscale.

    Args:
        img: Input image in BGR format (from cv2.imread or similar)
        target_size: Target size tuple (width, height)

    Returns:
        tuple: (img_rgb_norm, gray)
            - img_rgb_norm: Normalized RGB image (float32, range [0, 1])
            - gray: Grayscale image (uint8, range [0, 255])
    """
    img_resized = cv2.resize(img, target_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_rgb_norm = img_rgb.astype("float32") / 255.0
    gray = cv2.cvtColor((img_rgb_norm * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
    return img_rgb_norm, gray


def preprocess_pil_image(pil_image, target_size=(256, 256)):
    """
    Preprocessing PIL Image daun jagung.

    Args:
        pil_image: PIL Image object
        target_size: Target size tuple (width, height)

    Returns:
        tuple: (img_rgb_norm, gray)
    """
    # Convert PIL to numpy array (RGB format)
    img_rgb = np.array(pil_image.convert("RGB"))
    
    # Resize
    img_resized = cv2.resize(img_rgb, target_size)
    
    # Normalize
    img_rgb_norm = img_resized.astype("float32") / 255.0
    
    # Convert to grayscale
    gray = cv2.cvtColor((img_rgb_norm * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
    
    return img_rgb_norm, gray
