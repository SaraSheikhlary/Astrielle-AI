import streamlit as st
import librosa
import numpy as np
from transformers import pipeline
from PIL import Image

# --- 1. CORE CONFIG ---
st.set_page_config(page_title="ASTRIELLE AI", layout="wide")

# Force Dark Mode CSS immediately
st.markdown("""
    <style>
    [data-theme="light"], [data-theme="dark"], .stApp {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), 
                    url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
        background-size: cover;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🛰️ ASTRIELLE AI")
    st.write("Autonomous HSI Edge Intelligence")
    st.divider()
    page = st.radio("Navigation", ["Landing Page", "Vocal Biomarkers", "Structural Health", "Mission Summary"])
    st.divider()
    st.info("System: Edge Computing\nLatency: 0.004ms")

# --- 3. PAGE ROUTING ---
if page == "Landing Page":
    st.title("WELCOME TO ASTRIELLE AI")
    st.subheader("Deep Space Human-Systems Integration")
    st.write("Bypassing the 22-minute Mars-Earth delay with localized AI diagnostics.")
    if st.button("Access Command Center"):
        st.info("Select a module from the sidebar to begin.")

elif page == "Vocal Biomarkers":
    st.header("🎙️ Vocal Biomarker Monitor")
    st.write("Analyzing telemetry for mission-critical stressors.")
    up = st.file_uploader("Upload Audio (.wav)", type=["wav"])
    live = st.audio_input("Live Audio Stream")
    if up or live:
        st.success("Data stream active. Analyzing...")

elif page == "Structural Health":
    st.header("🛰️ Structural Health")
    vibe = st.slider("Vibration (Hz)", 0, 5000, 1100)
    st.line_chart(np.random.randn(
