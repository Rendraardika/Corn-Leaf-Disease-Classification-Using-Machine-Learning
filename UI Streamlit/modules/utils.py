"""
Utility constants and helper functions for Corn Leaf Disease Classifier.

IMPORTANT: The CLASS_MAP order must match the LabelEncoder order used during training.
LabelEncoder sorts labels alphabetically, so:
  - index 0: daun rusak
  - index 1: daun sehat  
  - index 2: hawar daun
  - index 3: karat daun
"""

# This matches the alphabetical order from sklearn's LabelEncoder
CLASS_MAP = ["Daun Rusak", "Daun Sehat", "Hawar Daun", "Karat Daun"]

# Color themes for each class
CLASS_COLORS = {
    "Daun Sehat": "#22c55e",    # Green
    "Daun Rusak": "#ef4444",    # Red
    "Hawar Daun": "#f59e0b",    # Amber
    "Karat Daun": "#a855f7"     # Purple
}

# Disease descriptions
CLASS_DESCRIPTIONS = {
    "Daun Sehat": "Daun dalam kondisi sehat tanpa tanda-tanda penyakit.",
    "Daun Rusak": "Daun mengalami kerusakan fisik atau mekanis.",
    "Hawar Daun": "Hawar daun (Northern Leaf Blight) disebabkan oleh jamur Exserohilum turcicum.",
    "Karat Daun": "Karat daun (Rust) disebabkan oleh jamur Puccinia sorghi."
}
