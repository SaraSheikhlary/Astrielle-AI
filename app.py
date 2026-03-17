import streamlit as st
import librosa
import pandas as pd
import numpy as np
from transformers import pipeline

# --- 1. SETTINGS & AUTHENTICATION STATE ---
st.set_page_config(layout="wide", page_title="Astrielle AI | Mission Control")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# --- 2. LOGIN & LANDING PAGE ---
if not st.session_state.authenticated:
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.7)), 
                            url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
                background-size: cover;
                display: flex; align-items: center; justify-content: center;
            }
            .auth-card {
                text-align: center; color: white; padding: 50px;
                background: rgba(0, 0, 0, 0.4); border-radius: 30px;
                backdrop-filter: blur(15px); border: 1px solid rgba(255,255,255,0.1);
            }
            .title-text { font-size: 60px; font-weight: 800; letter-spacing: 10px; margin-bottom: 0px; }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div class="title-text">ASTRIELLE</div>', unsafe_allow_html=True)
        st.write("### Autonomous Edge Intelligence")
        
        tab_login, tab_signup = st.tabs(["Login", "Create Mission Account"])
        
        with tab_login:
            user = st.text_input("Astronaut ID")
            pw = st.text_input("Access Key", type="password")
            if st.button("INITIALIZE UPLINK", use_container_width=True):
                st.session_state.authenticated = True
                st.rerun()
        
        with tab_signup:
            st.text_input("Full Name")
            st.text_input("Email")
            st.selectbox("Affiliation", ["NASA", "SpaceX", "ESA", "Private Sector"])
            st.button("Request Credentials")
        
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop() 

# --- 3. MAIN DASHBOARD ---
else:
    with st.sidebar:
        st.title("🛰️ Command")
        st.write("User: **Astro_01**")
        if st.button("🔴 TERMINATE SESSION"):
            st.session_state.authenticated = False
            st.rerun()
        st.divider()
        st.info("System Status: **Edge-Autonomous**")

    # Muted Background for Dashboard
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(rgba(10,10,15,0.94), rgba(10,10,15,0.94)), 
                            url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
                background-size: cover;
            }
            .stTabs [data-baseweb="tab-panel"] {
                background: rgba(255, 255, 255, 0.03); padding: 30px; border-radius: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🎙️ Vocal AI", "🛰️ Structural AI", "🧠 HSI Synergy", "📑 Mission Summary"])

    with tab1:
        st.title("✨ Vocal Biomarker Monitor")
        @st.cache_resource
        def load_voice_model():
            return pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
        
        classifier = load_voice_model()
        up = st.file_uploader("Upload Telemetry (.wav)", type="wav")
        if up:
            speech, sr = librosa.load(up, sr=16000)
            res = classifier(speech)
            for r in res:
                st.write(f"**{r['label']}**")
                st.progress(r['score'])

    with tab2:
        st.title("🛰️ Structural Health Monitoring")
        c1, c2 = st.columns([1, 2])
        with c1:
            vibe = st.slider("Vibration (Hz)", 0, 5000, 1100)
            strn = st
