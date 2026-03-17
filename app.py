import streamlit as st
import librosa
import pandas as pd
import numpy as np
from transformers import pipeline

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Astrielle AI | HSI")

if 'entered' not in st.session_state:
    st.session_state.entered = False

# --- 2. THE SPLASH SCREEN (Now with High Contrast & Color) ---
if not st.session_state.entered:
    st.markdown("""
        <style>
            .stApp {
                /* Vibrant Nebula Background with a lighter gradient so colors pop */
                background: linear-gradient(rgba(0, 5, 20, 0.3), rgba(0, 0, 0, 0.6)), 
                            url('https://images.unsplash.com/photo-1506703719100-a0f3a48c0f41?auto=format&fit=crop&q=80&w=2000');
                background-size: cover; display: flex; align-items: center; justify-content: center;
            }
            .landing-card {
                text-align: center; color: white; padding: 60px;
                background: rgba(10, 15, 30, 0.6); border-radius: 30px;
                backdrop-filter: blur(12px); border: 2px solid rgba(0, 242, 255, 0.3);
                box-shadow: 0 0 40px rgba(0, 242, 255, 0.2);
            }
            .title-text { 
                font-size: 85px; font-weight: 900; letter-spacing: 12px; 
                text-shadow: 0px 0px 20px rgba(0, 242, 255, 0.8); /* Neon Glow */
            }
            .subtitle-text { font-size: 22px; color: #00f2ff; letter-spacing: 3px; margin-bottom: 20px; font-weight: bold; }
        </style>
        <div class="landing-card">
            <div class="title-text">ASTRIELLE</div>
            <div class="subtitle-text">Autonomous Edge Intelligence</div>
            <p style="font-size:18px; opacity:0.9;">Advanced Human-Systems Integration for Deep Space</p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("INITIALIZE MISSION CONTROL", use_container_width=True):
        st.session_state.entered = True
        st.rerun()
    st.stop() 

# --- 3. THE MAIN DASHBOARD ---
else:
    with st.sidebar:
        st.title("🛰️ Command Center")
        if st.button("Log Out / Reset"):
            st.session_state.entered = False
            st.rerun()
        st.divider()
        st.write("**Local Latency:** 0.004ms")

    # THEME & BACKGROUND (Kept dark for readability)
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(rgba(14, 17, 23, 0.85), rgba(14, 17, 23, 0.85)), 
                            url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
                background-size: cover; background-attachment: fixed;
            }
            .stTabs [data-baseweb="tab-panel"] {
                padding: 30px; border-radius: 20px; backdrop-filter: blur(20px);
                background: rgba(30, 30, 30, 0.75); border: 1px solid rgba(128, 128, 128, 0.2);
            }
        </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🎙️ Vocal AI", "🛰️ Structural", "🧠 HSI Synergy", "📑 Summary"])

    with tab1:
        st.title("✨ Vocal Biomarker Monitor")
        
        @st.cache_resource
        def load_voice_model():
            return pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
        
        classifier = load_voice_model()
