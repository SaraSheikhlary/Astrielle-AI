import streamlit as st
import librosa
import numpy as np
from transformers import pipeline
from PIL import Image

# --- 1. SETTINGS ---
try:
    favicon = Image.open("astrielle_favicon_square.png")
    logo = Image.open("astrielle_logo_full.png")
except:
    favicon = None
    logo = None

st.set_page_config(page_title="ASTRIELLE AI", page_icon=favicon, layout="wide")

# --- 2. THE SWITCH LOGIC ---
if 'entered' not in st.session_state:
    st.session_state.entered = False

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

# --- 3. PAGE ROUTING ---
if not st.session_state.entered:
    # --- LANDING PAGE ---
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.title("🛰️ ASTRIELLE AI")
        st.subheader("Autonomous Edge Intelligence")
        st.write("Deep Space Human-Systems Integration.")
        if st.button("INITIALIZE MISSION CONTROL", use_container_width=True):
            st.session_state.entered = True
            st.rerun() # This forces the app to 'bite' the next page
else:
    # --- MAIN DASHBOARD ---
    with st.sidebar
