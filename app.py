import streamlit as st
import librosa
import numpy as np
from transformers import pipeline
from PIL import Image

# --- 1. SETUP ---
st.set_page_config(page_title="ASTRIELLE AI", layout="wide")

# Force Dark Mode CSS
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

# --- 2. NAVIGATION ---
with st.sidebar:
    st.title("🛰️ ASTRIELLE AI")
    st.write("Edge Intelligence Dashboard")
    st.divider()
    # This radio button replaces the landing page button to ensure it always 'bites'
    choice = st.radio("Mission Control", ["Home", "Vocal Biomarkers", "Structural Health", "Summary"])
    st.divider()
    st.caption("Status: Local Edge Active")

# --- 3. PAGE LOGIC ---
if choice == "Home":
    st.title("ASTRIELLE AI")
    st.subheader("Autonomous Human-Systems Integration")
    st.write("Localized AI diagnostics for Deep Space missions.")
    st.info("Please select a module from the sidebar to begin analysis.")

elif choice == "Vocal Biomarkers":
    st.header("🎙️ Vocal Biomarker Monitor")
    st.write("Upload telemetry or use the live feed to analyze psychological stressors.")
    up = st.file_uploader("Upload .WAV", type=["wav"])
    live = st.audio_input("Live Stream")
    if up or live:
        st.success("Data
