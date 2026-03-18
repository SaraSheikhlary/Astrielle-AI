import streamlit as st
import librosa
import pandas as pd
import numpy as np
from transformers import pipeline
from PIL import Image

# --- 1. CONFIGURATION ---
try:
    sidebar_logo_full = Image.open("astrielle_logo_full.png")
    favicon_icon_square = Image.open("astrielle_favicon_square.png")
except Exception:
    sidebar_logo_full = None
    favicon_icon_square = None

st.set_page_config(
    layout="wide", 
    page_title="ASTRIELLE AI | HSI",
    page_icon=favicon_icon_square, 
    initial_sidebar_state="expanded"
)

# --- 1.5 SEO & META TAGS ---
st.markdown("""
    <style>
        /* Hidden SEO text for search engine crawlers */
        .seo-hide { display: none; }
    </style>
    <div class="seo-hide">
        <h1>Astrielle AI - Deep Space Edge Intelligence</h1>
        <h2>Autonomous Human-Systems Integration for Mars Missions</h2>
        <p>Astrielle AI provides localized AI diagnostics, vocal biomarker monitoring, and structural health tracking to bypass the 20-minute Mars-Earth communication lag.</p>
    </div>
    <meta name="description" content="Astrielle AI: Autonomous Edge Intelligence and Human-Systems Integration for Deep Space Exploration.">
    <meta name="keywords" content="Astrielle AI, Space Technology, Edge Computing, Human-Systems Integration, Vocal Biomarkers, Mars Mission">
    <meta name="author" content="Astrielle AI">
""", unsafe_allow_html=True)


# --- 2. SESSION STATE ---
if 'entered' not in st.session_state:
    st.session_state.entered = False

# --- 3. THE SPLASH SCREEN ---
if not st.session_state.entered:
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                            url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
                background-size: cover;
                display: flex; align-items: center; justify-content: center;
            }
            .landing-card {
                text-align: center; color: white; padding: 60px;
                background: rgba(255, 255, 255, 0.05); 
                border-radius: 30px;
                backdrop-filter: blur(15px); 
                border: 1px solid rgba(255,255,255,0.1);
            }
            .title-text { font-size: 85px; font-weight: 800; letter-spacing: 12px; margin-bottom: 0px; }
            .subtitle-text { font-size: 22px; color: #00f2ff; letter-spacing: 3px; margin-bottom: 30px; }
        </style>
        <div class="landing-card">
            <div class="title-text">ASTRIELLE AI</div>
            <div class="subtitle-text">Autonomous Edge Intelligence</div>
            <p style="max-width:600px; margin:0 auto; font-size:18px; opacity:0.8;">
                Advanced <b>Human-Systems Integration</b> for Deep Space. 
                Localized AI diagnostics to bypass the 20-minute Mars-Earth communication lag.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("INITIALIZE MISSION CONTROL", use_container_width=True):
        st.session_state.entered = True
        st.rerun()
    st.stop()

# --- 4. THE MAIN DASHBOARD ---
else:
    with st.sidebar:
        if sidebar_logo_full:
            st.image(sidebar_logo_full, use_container_width=True)
        st.markdown("""
            <div style='text-align: center; color: #4F8BF9; font-size: 20px; font-weight: bold;'>
                ASTRIELLE AI
                <br>
                <span style='font-size: 14px; font-weight: normal; color: #AFAFAF;'>
                Autonomous HSI Edge Intelligence
                </span>
            </div>
            <br>
        """, unsafe_allow_html=True)
        st.divider()

        st.title("🛰️ Command Center")
        if st.button("Log Out / Reset View"):
            st.session_state.entered = False
            st.rerun()
        st.divider()
        st.write("**System:** Edge Computing")
        st.write("**Local Latency:** 0.004ms")
        st.write("**
