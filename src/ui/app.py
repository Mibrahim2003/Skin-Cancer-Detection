"""
DermaOps Streamlit Dashboard

A user-friendly web interface for skin lesion classification using the
Fine-Tuned ResNet50 model served via FastAPI.

Usage:
    streamlit run src/ui/app.py
    
    Or with Docker Compose:
    docker-compose up
"""

import os
import io
import streamlit as st
import requests
from PIL import Image
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# API URL Configuration
# In Docker Compose, the API hostname is 'api'. Locally, it's 'localhost'.
API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")

# For local development without Docker:
# API_URL = "http://localhost:8000/predict"

# Class descriptions for better UX
CLASS_DESCRIPTIONS = {
    "akiec": {
        "name": "Actinic Keratoses (Solar Keratoses)",
        "description": "Pre-cancerous scaly patches caused by sun exposure. Early treatment recommended.",
        "severity": "⚠️ Pre-cancerous",
        "color": "orange"
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "description": "Most common type of skin cancer. Rarely spreads but should be treated.",
        "severity": "🔴 Cancerous",
        "color": "red"
    },
    "bkl": {
        "name": "Benign Keratosis",
        "description": "Non-cancerous skin growth. Usually harmless but monitor for changes.",
        "severity": "✅ Benign",
        "color": "green"
    },
    "df": {
        "name": "Dermatofibroma",
        "description": "Benign fibrous skin nodule. Harmless, treatment usually not required.",
        "severity": "✅ Benign",
        "color": "green"
    },
    "mel": {
        "name": "Melanoma",
        "description": "Most dangerous form of skin cancer. Immediate medical attention required.",
        "severity": "🚨 Malignant",
        "color": "darkred"
    },
    "nv": {
        "name": "Melanocytic Nevi (Moles)",
        "description": "Common benign moles. Monitor for ABCDE warning signs.",
        "severity": "✅ Benign",
        "color": "green"
    },
    "vasc": {
        "name": "Vascular Lesions",
        "description": "Blood vessel-related skin conditions. Usually benign.",
        "severity": "✅ Benign",
        "color": "blue"
    }
}

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="DermaOps AI Diagnostic",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .severity-warning {
        background-color: #e65100;
        border-left: 5px solid #ff9800;
        color: white;
    }
    .severity-danger {
        background-color: #b71c1c;
        border-left: 5px solid #f44336;
        color: white;
    }
    .severity-safe {
        background-color: #1b5e20;
        border-left: 5px solid #4caf50;
        color: white;
    }
    .disclaimer {
        background-color: #031b2d;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/dermatology.png", width=80)
    st.title("DermaOps")
    st.markdown("---")
    
    st.subheader("ℹ️ About")
    st.markdown("""
    **DermaOps** is an AI-powered skin lesion classifier using:
    - 🧠 Fine-tuned ResNet50
    - 📊 HAM10000 Dataset
    - 🎯 7 Diagnostic Categories
    """)
    
    st.markdown("---")
    st.subheader("📋 Supported Diagnoses")
    for code, info in CLASS_DESCRIPTIONS.items():
        st.markdown(f"**{code}**: {info['name']}")
    
    st.markdown("---")
    st.subheader("⚙️ Settings")
    
    # API URL override for local testing
    custom_api = st.text_input(
        "API Endpoint",
        value=API_URL,
        help="Change this to http://localhost:8000/predict for local testing"
    )
    if custom_api != API_URL:
        API_URL = custom_api
    
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit & FastAPI")
    st.caption("Model: ResNet50 | Accuracy: 72.85%")

# =============================================================================
# MAIN CONTENT
# =============================================================================

# Header
st.markdown('<p class="main-header">🩺 DermaOps: AI Skin Lesion Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a dermoscopic image to get a real-time AI-powered diagnosis</p>', unsafe_allow_html=True)

st.markdown("---")

# Two-column layout
col1, col2 = st.columns([1, 1])

# =============================================================================
# COLUMN 1: IMAGE UPLOAD
# =============================================================================

with col1:
    st.header("📤 1. Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose a skin lesion image...",
        type=["jpg", "png", "jpeg"],
        help="Supported formats: JPG, PNG, JPEG. For best results, use dermoscopic images."
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width="stretch")
        
        # Image info
        st.caption(f"📁 File: {uploaded_file.name} | 📐 Size: {image.size[0]}x{image.size[1]}")
        
        # Analyze button
        if st.button("🔬 Analyze Lesion", type="primary", width="stretch"):
            with st.spinner("🧠 Running inference on ResNet50 model..."):
                try:
                    # Prepare image bytes for API
                    img_bytes = io.BytesIO()
                    # Convert to RGB if necessary (handle RGBA, etc.)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image.save(img_bytes, format='JPEG')
                    img_bytes.seek(0)
                    
                    # Prepare multipart form data
                    files = {
                        "file": (uploaded_file.name, img_bytes.getvalue(), "image/jpeg")
                    }
                    
                    # Send POST request to FastAPI backend
                    response = requests.post(API_URL, files=files, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state['result'] = result
                        st.success("✅ Analysis complete!")
                    else:
                        st.error(f"❌ API Error ({response.status_code}): {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("""
                    🔌 **Connection Error**
                    
                    Could not connect to the FastAPI backend. Please ensure:
                    1. The API server is running (`uvicorn src.api.main:app`)
                    2. The API URL is correct (check sidebar settings)
                    3. If using Docker, both containers are on the same network
                    """)
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. The model may be loading. Please try again.")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
    else:
        # Placeholder when no image uploaded
        st.info("👆 Upload a dermoscopic image to begin analysis")
        
        # Sample images hint
        with st.expander("💡 Tips for best results"):
            st.markdown("""
            - Use **dermoscopic images** (close-up, well-lit)
            - Ensure the **lesion is centered** in the image
            - Avoid blurry or low-resolution images
            - The model works best with **224x224** pixel images
            """)

# =============================================================================
# COLUMN 2: RESULTS
# =============================================================================

with col2:
    st.header("📊 2. Diagnostic Results")
    
    if 'result' in st.session_state:
        res = st.session_state['result']
        
        # Get prediction details
        prediction = res.get('prediction', 'Unknown')
        confidence = float(res.get('confidence', 0))
        probabilities = res.get('probabilities', {})
        
        # Get class info
        class_info = CLASS_DESCRIPTIONS.get(prediction, {
            "name": prediction,
            "description": "Unknown classification",
            "severity": "❓ Unknown",
            "color": "gray"
        })
        
        # Severity-based styling
        severity_class = "severity-safe"
        if prediction in ["mel", "bcc"]:
            severity_class = "severity-danger"
        elif prediction in ["akiec"]:
            severity_class = "severity-warning"
        
        # Main prediction display
        st.markdown(f"""
        <div class="result-box {severity_class}">
            <h2>{class_info['severity']}</h2>
            <h3>{class_info['name']}</h3>
            <p><strong>Code:</strong> {prediction.upper()}</p>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Description
        st.info(f"📋 **Description:** {class_info['description']}")
        
        # Confidence meter
        st.subheader("🎯 Confidence Score")
        st.progress(confidence)
        
        if confidence < 0.5:
            st.warning("⚠️ Low confidence prediction. Consider getting a second opinion.")
        elif confidence > 0.8:
            st.success("✅ High confidence prediction.")
        
        # Probability distribution chart
        st.subheader("📈 Class Probabilities")
        
        if probabilities:
            # Create DataFrame for visualization
            df_probs = pd.DataFrame([
                {"Class": k.upper(), "Probability": float(v) * 100}
                for k, v in probabilities.items()
            ])
            df_probs = df_probs.sort_values("Probability", ascending=True)
            
            # Horizontal bar chart
            st.bar_chart(
                df_probs.set_index("Class"),
                horizontal=True
            )
            
            # Top 3 predictions table
            st.subheader("🏆 Top Predictions")
            top3 = df_probs.nlargest(3, "Probability")
            top3["Probability"] = top3["Probability"].apply(lambda x: f"{x:.2f}%")
            st.table(top3.reset_index(drop=True))
        
        # Action recommendations
        st.subheader("📋 Recommended Actions")
        if prediction == "mel":
            st.error("""
            🚨 **URGENT: Potential Melanoma Detected**
            - Seek immediate consultation with a dermatologist
            - Do not delay medical evaluation
            - This is an AI screening tool, not a diagnosis
            """)
        elif prediction == "bcc":
            st.warning("""
            ⚠️ **Potential Basal Cell Carcinoma**
            - Schedule an appointment with a dermatologist
            - BCC rarely spreads but should be treated
            - Early treatment leads to better outcomes
            """)
        elif prediction == "akiec":
            st.warning("""
            ⚠️ **Potential Pre-cancerous Lesion**
            - Consult a dermatologist for evaluation
            - Actinic keratoses can progress to cancer
            - Regular skin checks recommended
            """)
        else:
            st.success("""
            ✅ **Likely Benign Lesion**
            - Continue regular skin monitoring
            - Watch for any changes (size, color, shape)
            - Annual skin check recommended
            """)
        
        # Clear results button
        if st.button("🔄 Clear Results"):
            del st.session_state['result']
            st.rerun()
            
    else:
        # No results yet
        st.info("👈 Upload an image and click 'Analyze Lesion' to see results here")
        
        # Show sample output
        with st.expander("📊 What to expect"):
            st.markdown("""
            After analysis, you'll see:
            1. **Primary Diagnosis** - The most likely classification
            2. **Confidence Score** - How certain the model is
            3. **Probability Distribution** - Scores for all 7 classes
            4. **Recommended Actions** - Next steps based on the diagnosis
            """)

# =============================================================================
# FOOTER / DISCLAIMER
# =============================================================================

st.markdown("---")

st.markdown("""
<div class="disclaimer">
    <h4>⚕️ Medical Disclaimer</h4>
    <p>
        <strong>This AI tool is for educational and screening purposes only.</strong>
        It is NOT a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified dermatologist for proper evaluation of skin lesions.
        The model has an accuracy of ~73% and may produce incorrect results.
    </p>
</div>
""", unsafe_allow_html=True)

# Footer info
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("🧠 Model: Fine-tuned ResNet50")
with col_f2:
    st.caption("📊 Dataset: HAM10000 (10,015 images)")
with col_f3:
    st.caption("🎯 Test F1 Score: 0.7486")
