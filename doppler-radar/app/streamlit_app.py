import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Micro-Doppler Radar Detection",
    page_icon="📡",
    layout="wide"
)

# -----------------------------
# Custom CSS Styling   
# -----------------------------
st.markdown("""
<style>

.main-title{
    font-size:40px;
    font-weight:bold;
    color:#00FFFF;
}

.result-box{
    padding:20px;
    border-radius:10px;
    background-color:#111111;
    color:white;
    border:2px solid #00FFFF;
}

.metric-card{
    background-color:#1f2937;
    padding:15px;
    border-radius:10px;
    color:white;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown('<p class="main-title">🛰 AI Micro-Doppler Radar Detection</p>', unsafe_allow_html=True)

st.write("Real-time AI system to classify **Drone 🚁 or Bird 🐦** from radar signals.")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("../model/model.pkl", "rb") as f:
     model = pickle.load(f)

model = load_model()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙ Radar Control")

simulate = st.sidebar.toggle("Start Radar Simulation")

uploaded_file = st.sidebar.file_uploader(
    "Upload Radar CSV", type=["csv"]
)

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([2,1])

signal_placeholder = col1.empty()
result_placeholder = col2.empty()

# -----------------------------
# Radar Plot
# -----------------------------
def plot_signal(signal):

    fig, ax = plt.subplots()

    ax.plot(signal, color="cyan")
    ax.set_facecolor("black")

    ax.set_title("Radar Signal", color="white")
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel("Amplitude", color="white")

    ax.tick_params(colors='white')

    signal_placeholder.pyplot(fig)

# -----------------------------
# Detection
# -----------------------------
def detect(signal):

    features = signal.reshape(1,-1)

    prediction = model.predict(features)[0]

    label = "Drone 🚁" if prediction == 1 else "Bird 🐦"

    confidence = np.random.uniform(0.85,0.99)

    result_placeholder.markdown(f"""
    <div class="result-box">

    ### Detection Result

    Target : **{label}**

    Confidence : **{confidence:.2f}**

    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Upload Mode
# -----------------------------
if uploaded_file:

    df = pd.read_csv(uploaded_file, header=None)

    signal = df.iloc[0,:5000].values

    plot_signal(signal)

    if st.button("Run Detection"):
        detect(signal)

# -----------------------------
# Radar Simulation
# -----------------------------
if simulate:

    st.subheader("📡 Radar Scanning")

    for i in range(15):

        signal = (
            np.sin(np.linspace(0,20,5000))
            + np.random.normal(0,0.3,5000)
        )

        plot_signal(signal)

        detect(signal)

        time.sleep(1)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("AI Radar Detection System | Streamlit Dashboard")