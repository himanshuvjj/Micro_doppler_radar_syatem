import streamlit as st
import numpy as np
import pickle
import os

# ------ PAGE CONFIG -----
st.set_page_config(
    page_title="Micro-Doppler Radar",
    page_icon="📡",
    layout="wide"
)

#------ CUSTOM CSS ------
st.markdown("""
    <style>
    .title {
        color: #00C9A7;
        font-size: 40px;
        font-weight: bold;
    }
    .card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        margin-top: 10px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ---- TITLE ------
st.markdown('<p class="title">📡 Micro-Doppler Radar Detection</p>', unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
model_path = os.path.join(os.path.dirname(__file__), "../model/model.pkl")

@st.cache_resource
def load_model():
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

model = load_model()

# ----- FILE UPLOAD ----
st.subheader("📂 Upload Signal File (CSV with 5000 values)")

uploaded_file = st.file_uploader("Upload your signal file", type=["csv", "txt"])

# ---- PROCESS FILE ----
if uploaded_file is not None:
    try:
        # Load data
        signal = np.loadtxt(uploaded_file, delimiter=",")

        # Flatten if needed
        signal = signal.flatten()

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f"📏 Signal Length: {len(signal)}")

        # Validate input size
        if len(signal) != 5000:
            st.error("❌ Invalid input! File must contain exactly 5000 values.")
        else: 
            # Show preview
            st.subheader("📊 Signal Preview")
            st.line_chart(signal[:200])  # show first 200 points

            # Prediction
            if model is None:
                st.error("⚠️ Model not loaded.")
            else:
                prediction = model.predict([signal])[0]

                st.subheader("🎯 Prediction Result")

                if prediction == 0:
                    st.success("🟢 Normal Object Detected")
                else:
                    st.error("🔴 Suspicious Activity Detected")

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")  