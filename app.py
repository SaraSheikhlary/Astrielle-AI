import streamlit as st
import librosa
import pandas as pd
import numpy as np
from transformers import pipeline

# --- 1. CONFIG ---
st.set_page_config(layout="wide", page_title="Astrielle AI | HSI")

# --- 2. THE CSS STYLES (Kept separate to prevent syntax errors) ---
LANDING_STYLE = """
<style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.2), rgba(0,0,0,0.4)), 
                    url("https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000");
        background-size: cover;
        display: flex; align-items: center; justify-content: center;
    }
    .landing-card {
        text-align: center; color: white; padding: 50px;
        background: rgba(0, 0, 0, 0.3); border-radius: 25px; 
        backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2);
    }
    .title-text { font-size: 70px; font-weight: 800; letter-spacing: 10px; margin: 0; }
    .subtitle-text { font-size: 18px; color: #00f2ff; letter-spacing: 4px; margin-bottom: 30px; }
</style>
"""

DASHBOARD_STYLE = """
<style>
    .stApp {
        background: linear-gradient(rgba(10, 10, 15, 0.94), rgba(10, 10, 15, 0.94)), 
                    url("https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000");
        background-size: cover; background-attachment: fixed;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 25px; border-radius: 15px; background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px); border: 1px solid rgba(255, 255, 255, 0.1); margin-top: 15px;
    }
    h1, h2, h3, p, span, label, .stMarkdown { color: white !important; }
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%; text-align: center; 
        font-size: 0.8em; padding: 10px; background: rgba(0,0,0,0.9); color: white;
    }
</style>
"""

# --- 3. LOGIC ---
if 'entered' not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:
    st.markdown(LANDING_STYLE, unsafe_allow_html=True)
    st.markdown('<div class="landing-card"><div class="title-text">ASTRIELLE</div><div class="subtitle-text">AUTONOMOUS EDGE INTELLIGENCE</div><p style="opacity:0.9;">Localized Human-Systems Integration for Deep Space.</p></div>', unsafe_allow_html=True)
    
    if st.button("INITIALIZE MISSION CONTROL", use_container_width=True):
        st.session_state.entered = True
        st.rerun()
    st.stop()

else:
    st.markdown(DASHBOARD_STYLE, unsafe_allow_html=True)
    
    with st.sidebar:
        st.title("🛰️ Command")
        if st.button("Log Out"):
            st.session_state.entered = False
            st.rerun()
        st.divider()
        st.write("**Latency:** 0.004ms")

    tab1, tab2, tab3 = st.tabs(["🎙️ Vocal AI", "🛰️ Structural AI", "🧠 Synergy"])

    with tab1:
        st.header("Vocal Biomarker Monitor")
        @st.cache_resource
        def load_voice_model():
            return pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
        
        classifier = load_voice_model()
        up = st.file_uploader("Upload Telemetry", type="wav")
        if up:
            speech, sr = librosa.load(up, sr=16000)
            res = classifier(speech)
            for r in res:
                st.write(f"**{r['label']}**")
                st.progress(r['score'])

    with tab2:
        st.header("Structural Health")
        strain = st.slider("Strain (με)", 0, 10000, 4000)
        st.metric("Deformation Risk", f"{(strain/100):.1f}%")
        st.line_chart(np.random.randn(15, 1))

    with tab3:
        st.header("HSI Synergy")
        st.info("Edge Feedback: Active")
        st.bar_chart({"Earth Delay": 1320, "Astrielle AI": 0.004})

    st.markdown('<div class="footer">© 2026 Astrielle AI | Confidential Mission Telemetry</div>', unsafe_allow_html=True)
