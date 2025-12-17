import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import time

from skimage.measure import shannon_entropy

# ==============================
# IMPORT MODULE ML
# ==============================
from modules.pipeline import predict_image, get_class_names
from modules.utils import CLASS_MAP, CLASS_COLORS, CLASS_DESCRIPTIONS

# ==============================
# KONFIGURASI HALAMAN
# ==============================
st.set_page_config(
    page_title="CornShield - Deteksi Penyakit Daun Jagung",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# PARAMETER VALIDASI (FINAL)
# ==============================
CONF_THRESHOLD = 0.55
MIN_AREA_RATIO = 0.05
MIN_GREEN_RATIO = 0.15
MIN_ENTROPY = 3.0
MAX_ENTROPY = 8.8
MAX_GRAY_RATIO_ON_LEAF = 0.85
MIN_CONF_MARGIN = 0.10

# ==============================
# FUNGSI VALIDASI
# ==============================
def is_green_dominant_hsv(image):
    img_rgb = np.array(image.convert("RGB"))
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(img_hsv, lower_green, upper_green)
    return (np.sum(mask > 0) / mask.size) >= MIN_GREEN_RATIO


def is_valid_texture(image):
    gray = np.array(image.convert("L"))
    entropy = shannon_entropy(gray)
    return MIN_ENTROPY <= entropy <= MAX_ENTROPY


def is_valid_segmentation(segmentation):
    total_pixels = segmentation.size
    leaf_pixels = np.sum(segmentation > 0)
    return (leaf_pixels / total_pixels) >= MIN_AREA_RATIO


def is_mostly_grayscale_on_leaf(image, segmentation):
    """
    FIX UTAMA:
    - Resize image ke ukuran segmentation
    - Hindari mismatch (480x640 vs 256x256)
    """
    seg_h, seg_w = segmentation.shape
    image_resized = image.resize((seg_w, seg_h))

    img_rgb = np.array(image_resized.convert("RGB"))
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s_channel = img_hsv[:, :, 1]

    leaf_mask = segmentation > 0
    if np.sum(leaf_mask) == 0:
        return True

    gray_leaf_pixels = np.sum((s_channel < 30) & leaf_mask)
    total_leaf_pixels = np.sum(leaf_mask)

    ratio = gray_leaf_pixels / total_leaf_pixels
    return ratio > MAX_GRAY_RATIO_ON_LEAF


def is_prediction_confident(probabilities):
    probs_sorted = np.sort(probabilities)
    top = probs_sorted[-1]
    second = probs_sorted[-2]
    return top >= CONF_THRESHOLD and (top - second) >= MIN_CONF_MARGIN

# ==============================
# CSS
# ==============================
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.stApp { background: #F4F7F6; }
.hero-container {
    border-radius: 20px;
    background: linear-gradient(135deg, #2E7D32, #1b5e20);
    color: white;
    padding: 3rem 2rem;
    margin-bottom: 2rem;
}
.custom-card {
    background: white;
    padding: 1.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 16px rgba(0,0,0,0.06);
}
.result-box {
    background: white;
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    border: 2px solid #2E7D32;
}
.result-conf {
    font-size: 0.9rem;
    padding: 6px 14px;
    border-radius: 50px;
    background: #e8f5e9;
    display: inline-block;
}
.main-footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    background: white;
    padding: 1rem;
    border-top: 1px solid #eee;
    font-size: 0.85rem;
    text-align: center;
}
.content-spacer { height: 100px; }
</style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.markdown("## 🌽 CornShield ML")
    st.caption("Final Project Machine Learning 2025")
    st.markdown("---")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(base_dir, "assets", "sample_images")

    samples = ["Tidak Ada"]
    if os.path.exists(sample_dir):
        samples += [f for f in os.listdir(sample_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    selected_sample = st.selectbox("🖼️ Coba Gambar Sampel", samples)

    st.markdown("---")
    st.markdown("### 🏷️ Keterangan Kelas")
    for cls in CLASS_MAP:
        st.markdown(f"- {cls}")

# ==============================
# HERO
# ==============================
st.markdown("""
<div class="hero-container">
    <h1>Deteksi Penyakit Daun Jagung</h1>
    <p>Sistem klasifikasi penyakit daun jagung berbasis Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# INPUT
# ==============================
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("📤 Unggah citra daun jagung",
                                     type=["jpg", "jpeg", "png"])

with col2:
    st.info("""
    **Tips Pengambilan Gambar**
    - Fokus satu helai daun
    - Background putih diperbolehkan
    - Cahaya cukup
    """)

image = None
if uploaded_file:
    image = Image.open(uploaded_file)
elif selected_sample != "Tidak Ada":
    image = Image.open(os.path.join(sample_dir, selected_sample))

# ==============================
# PROSES
# ==============================
if image:
    with st.spinner("🔬 Menganalisis citra..."):
        time.sleep(0.5)
        pred_class, probs, segmentation, confidence = predict_image(image)

    # VALIDASI BERURUTAN (FINAL)
    if not is_green_dominant_hsv(image):
        st.error("❌ Objek tidak dikenali (warna hijau tidak dominan)")
        st.stop()

    if not is_valid_texture(image):
        st.error("❌ Tekstur tidak menyerupai daun alami")
        st.stop()

    if not is_valid_segmentation(segmentation):
        st.error("❌ Segmentasi daun gagal")
        st.stop()

    if is_mostly_grayscale_on_leaf(image, segmentation):
        st.error("❌ Daun terlalu pucat / rusak parah")
        st.stop()

    if not is_prediction_confident(probs):
        st.warning("""
        ⚠️ Objek hijau terdeteksi, namun tidak dapat
        dipastikan sebagai daun jagung.
        """)
        st.stop()

    # ==============================
    # OUTPUT
    # ==============================
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        t1, t2 = st.tabs(["Citra Asli", "Segmentasi"])
        with t1:
            st.image(image, use_container_width=True)
        with t2:
            st.image(segmentation, clamp=True, channels="GRAY",
                     use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        conf_pct = np.max(probs) * 100
        color = CLASS_COLORS[pred_class]

        st.markdown(f"""
        <div class="result-box" style="border-color:{color}">
            <h2 style="color:{color}">{pred_class}</h2>
            <div class="result-conf">Keyakinan: {conf_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📋 Deskripsi Penyakit")
        st.write(CLASS_DESCRIPTIONS[pred_class])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📊 Distribusi Probabilitas")
        for cls, p in zip(get_class_names(), probs):
            st.progress(float(p), text=f"{cls} – {p*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Silakan unggah citra daun jagung untuk memulai analisis.")

st.markdown('<div class="content-spacer"></div>', unsafe_allow_html=True)

# ==============================
# FOOTER
# ==============================
st.markdown("""
<div class="main-footer">
    Dikembangkan untuk Tugas Akhir Machine Learning<br>
    Informatika – UPN "Veteran" Jawa Timur | 2025
</div>
""", unsafe_allow_html=True)
