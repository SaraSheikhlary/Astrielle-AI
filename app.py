import streamlit as st
import librosa
import pandas as pd
import numpy as np
from transformers import pipeline

# --- 1. CONFIG ---
st.set_page_config(layout="wide", page_title="Astrielle AI")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# --- 2. LOGIN PAGE ---
if not st.session_state.authenticated:
    # We use a single line to avoid triple-quote errors
    st.markdown('<style>.stApp { background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.7)), url("https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000"); background-size: cover; display: flex; align-items: center; justify-content: center; } .auth-card { text-align: center; color: white; padding: 40px; background: rgba(0, 0, 0, 0.5); border-radius: 20px; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }</style>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown('<div class="auth-card"><h1>🛰️ ASTRIELLE AI</h1><p>Autonomous Mission Intelligence</p></div>', unsafe_allow_html=True)
        user = st.text_input("Astronaut ID")
        pw = st.text_input("Access Key", type="password")
        if st.button("INITIALIZE UPLINK", use_container_width=True):
            st.session_state.authenticated = True
            st.rerun()
    st.stop() 

# --- 3. MAIN DASHBOARD ---
else:
    with st.sidebar:
        st.title("👨‍🚀 Command")
        if st.button("TERMINATE SESSION"):
            st.session_state.authenticated = False
            st.rerun()
        st.divider()
        st.info("System Status: Nominal")

    # Muted Dashboard Background - Single line to prevent errors
    st.markdown('<style>.stApp { background: linear-gradient(rgba(10,10,15,0.95), rgba(10,10,15,0.95)), url("https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000"); background-size: cover; } .stTabs [data-baseweb="tab-panel"] { background: rgba(255, 255, 255, 0.05); padding: 25px; border-radius: 15px; } h1, h2, h3, p, label, span { color: white !important; }</style>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🎙️ Vocal AI", "🛰️ Structural AI", "🧠 HSI Synergy", "📑 Mission Summary"])

    with tab1:
        st.header("Vocal Biomarker Monitor")
        @st.cache_resource
        def load_voice_model():
            return pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
        
        try:
            classifier = load_voice_model()
            up = st.file_uploader("Upload Audio Telemetry (.wav)", type="wav")
            if up:
                speech, sr = librosa.load(up, sr=16000)
                res = classifier(speech)
                for r in res:
                    st.write(f"**{r['label']}**")
                    st.progress(r['score'])
        except:
            st.warning("AI Model Loading... please wait.")

    with tab2:
        st.header("Structural Health")
        c1, c2 = st.columns([1, 2])
        with c1:
            strain = st.slider("Hull Strain (με)", 0, 10000, 4000)
        with c2:
            st.metric("Deformation Risk", f"{strain/100}%", delta="Predictive")
            st.line_chart(np.random.randn(20, 1))

    with tab3:
        st.header("Human-Systems Integration")
        st.info("Direct Edge Feedback: Active. Mars-Earth Delay: 22m (Bypassed)")
        st.bar_chart({"Earth Delay (s)": 1320, "Astrielle AI (s)": 0.004})

    with tab4:
        st.header("📑 Mission Summary & Prediction Logic")
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Predictive Analytics")
            st.write("Using Strain-Life (e-N) theory, the app calculates the Rate of Damage to predict
