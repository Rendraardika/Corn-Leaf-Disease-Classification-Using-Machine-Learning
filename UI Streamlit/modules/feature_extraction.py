"""
Feature extraction module for corn leaf disease classification.

This module contains three types of texture feature extraction:
1. Fine Features: LBP-like rotation invariant (256 bins)
2. Coarse Features: Gradient magnitude histogram (32 bins)
3. DOR Features: Directional Order Relation (25 bins)

Total feature vector size: 256 + 32 + 25 = 313 dimensions
"""

import cv2
import numpy as np


def extract_fine_features(gray, radius=1, neighbors=8, step=2):
    """
    Extract Fine texture features using LBP-like rotation invariant method.

    Args:
        gray: Grayscale image (uint8)
        radius: Radius for neighbor sampling (default: 1)
        neighbors: Number of neighbors to sample (default: 8)
        step: Step size for pixel sampling (default: 2)

    Returns:
        hist: Normalized histogram of LBP codes (256 bins)
    """
    h, w = gray.shape
    codes = []

    for y in range(radius, h - radius, step):
        for x in range(radius, w - radius, step):
            center = gray[y, x]
            binary = []
            for n in range(neighbors):
                theta = 2 * np.pi * n / neighbors
                yy = int(round(y + radius * np.sin(theta)))
                xx = int(round(x + radius * np.cos(theta)))
                binary.append(1 if gray[yy, xx] >= center else 0)

            # Rotation invariant: take minimum rotation
            rotations = [
                int("".join(map(str, binary[i:] + binary[:i])), 2)
                for i in range(neighbors)
            ]
            codes.append(min(rotations))

    hist, _ = np.histogram(codes, bins=256, range=(0, 256))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-8)
    return hist


def extract_coarse_features(gray, num_bins=32):
    """
    Extract Coarse texture features using gradient magnitude histogram.

    Args:
        gray: Grayscale image (uint8)
        num_bins: Number of histogram bins (default: 32)

    Returns:
        hist: Normalized histogram of gradient magnitudes
    """
    sobelx = cv2.Sobel(gray.astype("float32"), cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray.astype("float32"), cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    hist, _ = np.histogram(magnitude, bins=num_bins, range=(0, magnitude.max() + 1e-8))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-8)
    return hist


def extract_dor_features(gray, window_size=5):
    """
    Extract DOR (Directional Order Relation) features.

    Args:
        gray: Grayscale image (uint8)
        window_size: Size of the window for DOR computation (must be odd, default: 5)

    Returns:
        hist: Normalized histogram of dominant indices (window_size^2 bins)
    """
    assert window_size % 2 == 1, "window_size harus ganjil"

    pad = window_size // 2
    padded = np.pad(gray.astype("float32"), pad, mode="reflect")
    h, w = gray.shape
    dom_idx = []

    for y in range(h):
        for x in range(w):
            region = padded[y:y+window_size, x:x+window_size]
            center = padded[y+pad, x+pad]

            diffs = np.abs(region - center).flatten()
            dom_idx.append(np.argmax(diffs))

    num_pos = window_size * window_size
    hist, _ = np.histogram(dom_idx, bins=num_pos, range=(0, num_pos))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-8)
    return hist


def extract_features(gray):
    """
    Extract all features (Fine + Coarse + DOR) and concatenate them.

    Args:
        gray: Grayscale image (uint8)

    Returns:
        features: Feature vector of 313 dimensions (256 + 32 + 25)
    """
    fine = extract_fine_features(gray)
    coarse = extract_coarse_features(gray)
    dor = extract_dor_features(gray)

    # Total fitur = 256 + 32 + 25 = 313 dimensi
    return np.concatenate([fine, coarse, dor]).astype("float32")
