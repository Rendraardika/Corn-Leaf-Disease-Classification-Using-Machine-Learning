"""
Segmentation module for corn leaf disease classification.

This module contains the Otsu thresholding segmentation function.
"""

import cv2


def segment_otsu(gray):
    """
    Segmentasi Otsu berbasis intensitas.

    Tahapan:
    1. Gaussian blur untuk mereduksi noise.
    2. Thresholding Otsu -> citra biner.

    Args:
        gray: Grayscale image (uint8)

    Returns:
        otsu_binary: Binary segmentation mask (uint8, values 0 or 255)
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu_binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_binary
