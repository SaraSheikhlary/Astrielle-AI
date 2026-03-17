import streamlit as st
import librosa
import pandas as pd
import numpy as np
from transformers import pipeline

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Astrielle AI | HSI")

# --- 2. SESSION STATE ---
if 'entered' not in st.session_state:
    st.session_state.entered = False

# --- 3. THE SPLASH SCREEN (Vibrant & Colorful) ---
if not st.session_state.entered:
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.5)), 
                            url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
                background-size: cover;
                display: flex; align-items: center; justify-content: center;
            }
            .landing-card {
                text-align: center; color: white; padding: 60px;
                background: rgba(0, 0, 0, 0.2); 
                border-radius: 30px;
                backdrop-filter: blur(8px); 
                border: 1px solid rgba(255,255,255,0.2);
            }
            .title-text { font-size: 85px; font-weight: 800; letter-spacing: 12px; margin-bottom: 0px; text-shadow: 2px 2px 15px rgba(0,0,0,0.5); }
            .subtitle-text { font-size: 22px; color: #00f2ff; letter-spacing: 3px; margin-bottom: 30px; text-transform: uppercase; }
        </style>
        <div class="landing-card">
            <div class="title-text">ASTRIELLE</div>
            <div class="subtitle-text">Autonomous Edge Intelligence</div>
            <p style="max-width:600px; margin:0 auto; font-size:18px; opacity:1.0; font-weight:500;">
                Advanced <b>Human-Systems Integration</b> for Deep Space. 
                Localized AI diagnostics to bypass the 20-minute communication lag.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("INITIALIZE MISSION CONTROL", use_container_width=True):
        st.session_state.entered = True
        st.rerun()
    st.stop() 

# --- 4. THE MAIN DASHBOARD (Dark & Muted) ---
else:
    with st.sidebar:
        st.title("🛰️ Command Center")
        if st.button("Log Out / Reset View"):
            st.session_state.entered = False
            st.rerun()
        st.divider()
        st.write("**System Status:** Nominal")
        st.write("**Local Latency:** 0.004ms")

    # DASHBOARD CSS
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(rgba(10, 10, 15, 0.92), rgba(10, 10, 15, 0.92)), 
                            url('
