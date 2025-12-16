import streamlit as st
import numpy as np
from PIL import Image
import os
import time

from modules.pipeline import predict_image, get_class_names, load_model
from modules.utils import CLASS_MAP, CLASS_COLORS, CLASS_DESCRIPTIONS

st.set_page_config(
    page_title="CornShield - Disease Classifier",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    /* VARIABLE DEFINITIONS */
    :root {
        --primary-green: #2E7D32;
        --light-green: #A5D6A7;
        --accent-yellow: #FFC107;
        --dark-bg: #0f1b2b;
        --card-bg: #ffffff;
        --text-dark: #1f2937;
        --text-light: #6b7280;
        --shadow-light: 4px 4px 10px rgba(0,0,0,0.05), -4px -4px 10px rgba(255,255,255,0.8);
        --shadow-hover: 6px 6px 15px rgba(46, 125, 50, 0.15), -6px -6px 15px rgba(255,255,255,1);
    }

    /* GLOBAL STYLES */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: #F4F7F6;
    }
    
    /* REMOVE DEFAULT STREAMLIT PADDING */
    .clean-block.css-13ln4jf {
        padding-top: 2rem;
    }

    /* CUSTOM COMPONENTS */
    
    /* Hero Section */
    .hero-container {
        border-radius: 20px;
        background: linear-gradient(135deg, var(--primary-green) 0%, #1b5e20 100%);
        color: white;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(46, 125, 50, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    /* Abstract corn leaf pattern overlay */
    .hero-container::after {
        content: "🌽";
        font-size: 15rem;
        position: absolute;
        right: -2rem;
        bottom: -4rem;
        opacity: 0.1;
        transform: rotate(-15deg);
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        max-width: 600px;
        opacity: 0.9;
    }

    /* Card Styling */
    .custom-card {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: var(--shadow-light);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(0,0,0,0.02);
        margin-bottom: 1.5rem;
        height: 100%;
    }

    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-hover);
    }
    
    .card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-dark);
        margin-bottom: 1rem;
        border-bottom: 2px solid #f3f4f6;
        padding-bottom: 0.5rem;
    }

    /* Prediction Result Box */
    .result-box {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 2px solid var(--primary-green);
        box-shadow: 0 10px 30px rgba(46, 125, 50, 0.1);
        margin-bottom: 1.5rem;
    }

    .result-label {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--primary-green);
        margin: 0.5rem 0;
        line-height: 1.2;
    }

    .result-conf {
        font-size: 0.9rem;
        color: var(--text-light);
        background: #e8f5e9;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        display: inline-block;
        margin-top: 0.5rem;
        font-weight: 600;
    }
    
    /* Metrics Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }

    .metric-item {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        border-bottom: 3px solid var(--accent-yellow);
    }
    
    .metric-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-dark);
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: var(--text-light);
        text-transform: uppercase;
        margin-top: 0.2rem;
    }

    /* Footer */
    .main-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: white;
        text-align: center;
        padding: 1rem;
        box-shadow: 0 -4px 10px rgba(0,0,0,0.05);
        z-index: 999;
        font-size: 0.85rem;
        color: var(--text-light);
        border-top: 1px solid #f3f4f6;
    }
    
    /* Fix for content being hidden behind footer */
    .content-spacer {
        height: 80px;
    }
    
    /* File Uploader styling */
    div[data-testid="stFileUploader"] {
        width: 100%;
    }
    
    div[data-testid="stFileUploader"] section {
        background-color: #f0fdf4;
        border: 2px dashed var(--primary-green);
        border-radius: 12px;
    }

</style>
""", unsafe_allow_html=True)

# === SIDEBAR ===

with st.sidebar:
    st.markdown("## 🌽 CornShield ML")
    st.caption("v1.0.0 | XGBoost Model")
    st.markdown("---")
    
    # Simple navigation feel
    st.markdown("**📊 Dashboard Controls**")
    
    sample_dir = os.path.join(os.path.dirname(__file__), "assets", "sample_images")
    sample_files = ["None"]
    if os.path.exists(sample_dir):
        sample_files += [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    selected_sample = st.selectbox("📝 Using Sample Image", sample_files)
    
    st.markdown("---")
    
    st.markdown("### 🧬 Architecture")
    with st.expander("Feature Extraction Step"):
        st.markdown("""
        **1. Preprocessing**
        - Resize (256x256)
        - Normalization
        
        **2. Texture Analysis**
        - Fine LBP (256 dims)
        - Coarse Gradient (32 dims)
        - DOR (25 dims)
        
        **3. Classification**
        - XGBoost Inference
        """)
    
    st.markdown("### 🏷️ Legend")
    for cls in CLASS_MAP:
        color = CLASS_COLORS[cls]
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; font-size: 0.9rem;">
            <span>{cls}</span>
            <span style="width: 12px; height: 12px; background-color: {color}; border-radius: 50%;"></span>
        </div>
        """, unsafe_allow_html=True)


# === HERO SECTION ===

st.markdown("""
<div class="hero-container">
    <div class="hero-title">Corn Leaf Disease Classifier</div>
    <div class="hero-subtitle">
        Accelerating plant pathology diagnosis with computer vision and machine learning.
        Detects Healthy, Damaged, Blight, and Rust conditions.
    </div>
</div>
""", unsafe_allow_html=True)


# === MAIN APPLICATION ===
col_input, col_tips = st.columns([2, 1])

with col_input:
    st.markdown("##### 📤 Upload Image Analysis")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="main_uploader")
    
with col_tips:
    st.info("""
    **💡 Tip:** For best results, ensure the image:
    - Focuses on a single leaf
    - Has good lighting
    - Is not blurry
    """)


image_to_process = None
source_type = "Upload"

if uploaded_file is not None:
    image_to_process = Image.open(uploaded_file)
elif selected_sample != "None":
    sample_path = os.path.join(sample_dir, selected_sample)
    image_to_process = Image.open(sample_path)
    source_type = "Sample"


if image_to_process:
    st.write("") # Spacer
    
    with st.spinner("🔬 Running texture analysis pipeline..."):
        time.sleep(0.5) # UX Delay
        
        try:
            pred_class, probabilities, segmentation = predict_image(image_to_process)
            
            d_col1, d_col2 = st.columns([1.2, 1])
            
            with d_col1:
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-header">👁️ Visual Analysis</div>', unsafe_allow_html=True)
                
                # Tabs for different views
                tab1, tab2 = st.tabs(["Original Image", "Otsu Segmentation"])
                
                with tab1:
                    st.image(image_to_process, use_container_width=True)
                    
                with tab2:
                    st.image(segmentation, use_container_width=True, clamp=True, channels="GRAY")
                    st.caption("Otsu thresholding isolates the leaf structure from background/noise.")
                
                st.markdown('</div>', unsafe_allow_html=True)
                

                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-header">📊 Image Metrics</div>', unsafe_allow_html=True)
                
                w, h = image_to_process.size
                
                st.markdown(f"""
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-value">{w}x{h}</div>
                        <div class="metric-label">Resolution</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">313</div>
                        <div class="metric-label">Features</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{source_type}</div>
                        <div class="metric-label">Source</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with d_col2:
                confidence = np.max(probabilities) * 100
                pred_color = CLASS_COLORS[pred_class]
                

                icons = {
                    "Daun Sehat": "🌿",
                    "Daun Rusak": "🍂",
                    "Hawar Daun": "🦠",
                    "Karat Daun": "🟤"
                }
                icon = icons.get(pred_class, "🍃")
                
                st.markdown(f"""
                <div class="result-box" style="border-color: {pred_color};">
                    <div style="font-size: 3.5rem; margin-bottom: 0.5rem; animation: pulse 2s infinite;">{icon}</div>
                    <div style="color: #64748b; font-size: 0.9rem; letter-spacing: 1px; text-transform: uppercase;">Prediction Model Result</div>
                    <div class="result-label" style="color: {pred_color};">{pred_class}</div>
                    <div class="result-conf" style="color: {pred_color}; background: {pred_color}15;">
                        Confidence Score: {confidence:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                

                desc = CLASS_DESCRIPTIONS[pred_class]
                st.markdown(f"""
                <div class="custom-card" style="border-left: 4px solid {pred_color}; padding: 1.2rem;">
                    <strong>📋 Diagnosis Details</strong>
                    <p style="margin-top: 0.5rem; color: #4b5563; font-size: 0.95rem; line-height: 1.5;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
                

                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-header">📉 Probability Distribution</div>', unsafe_allow_html=True)
                
                class_names = get_class_names()
                
                for cls, prob in zip(class_names, probabilities):
                    color = CLASS_COLORS[cls]
                    pct = prob * 100
                    is_winner = (cls == pred_class)
                    font_weight = "700" if is_winner else "400"
                    opacity = "1" if is_winner else "0.6"
                    
                    st.markdown(f"""
                    <div style="margin-bottom: 15px; opacity: {opacity};">
                        <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                            <span style="font-weight:{font_weight}; color:#1f2937;">{cls}</span>
                            <span style="font-weight:{font_weight}; color:#1f2937;">{pct:.1f}%</span>
                        </div>
                        <div style="width:100%; background-color:#f3f4f6; border-radius:8px; height:8px;">
                            <div style="width:{pct}%; background-color:{color}; height:8px; border-radius:8px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Something went wrong during analysis. Please check the model files.")
            st.error(str(e))

else:

    st.markdown("""
    <div style="
        display: flex; 
        flex-direction: column; 
        align-items: center; 
        justify-content: center; 
        padding: 4rem; 
        background: white; 
        border-radius: 20px; 
        border: 2px dashed #e5e7eb;
        margin-top: 2rem;
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.5;">🖼️</div>
        <h3 style="color: #9ca3af; margin: 0;">No Image Selected</h3>
        <p style="color: #9ca3af; margin-top: 0.5rem;">Upload an image or select a sample from the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div class="content-spacer"></div>', unsafe_allow_html=True)


st.markdown("""
<div class="main-footer">
    <div style="max-width: 1200px; margin: 0 auto; display: flex; justify-content: space-between; align-items: center;">
        <div style="font-weight: 600; color: var(--primary-green);">CornShield ML</div>
        <div>developed for Machine Learning Final Project</div>
        <div style="font-size: 0.8rem;">2025</div>
    </div>
</div>
""", unsafe_allow_html=True)
