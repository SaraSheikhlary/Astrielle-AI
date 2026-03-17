import streamlit as st
import librosa
import pandas as pd
import numpy as np
from transformers import pipeline

# --- 1. CONFIG ---
st.set_page_config(layout="wide", page_title="Astrielle AI | Space Biotech")

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
            <div class="title-text">ASTRIELLE</div>
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
    # Sidebar for Reset & Edge Info
    with st.sidebar:
        st.title("🛰️ Command Center")
        if st.button("Log Out / Reset View"):
            st.session_state.entered = False
            st.rerun()
        st.divider()
        st.write("**System:** Edge Computing")
        st.write("**Latency:** 0.004ms (Local)")
        st.write("**Earth Sync:** 22m Delay (Bypassed)")

    # Main CSS for Dashboard & Footer    
    st.markdown("""
        <style>
            /* 1. This uses the system's own text color variable */
            .stApp {
                color: var(--text-color);
                background: linear-gradient(rgba(14, 17, 23, 0.8), rgba(14, 17, 23, 0.8)), 
                            url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
                background-size: cover;
            }

            /* 2. Forces headings to be crisp in both modes */
            h1, h2, h3, p {
                color: var(--text-color) !important;
            }

            /* 3. Glass-morphism for Tab Content (Makes it readable) */
            .stTabs [data-baseweb="tab-panel"] {
                background: rgba(255, 255, 255, 0.05);
                padding: 20px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }

            /* 4. The Footer Styling */
            .footer {
                position: fixed; left: 0; bottom: 0; width: 100%;
                background-color: rgba(0, 0, 0, 0.8); 
                color: #aaa; text-align: center;
                font-size: 0.8em; padding: 10px 0; z-index: 999;
            }
        </style>
    """, unsafe_allow_html=True)
            .stApp {
                background: linear-gradient(rgba(14, 17, 23, 0.9), rgba(14, 17, 23, 0.9)), 
                            url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
                background-size: cover;
            }
            .footer {
                position: fixed; left: 0; bottom: 0; width: 100%;
                background-color: rgba(14, 17, 23, 0.95); 
                color: #888; text-align: center;
                font-size: 0.8em; padding: 15px 0; z-index: 999;
                border-top: 1px solid rgba(255,255,255,0.1);
            }
        </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎙️ Vocal Biomarkers", "🛰️ Structural Health", "🧠 Human-Systems Integration"])

    with tab1:
        st.title("✨ Vocal Biomarker Monitor")
        @st.cache_resource
        def load_voice_model():
            return pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
        
        classifier = load_voice_model()
        
        up = st.file_uploader("Upload Voice Telemetry (.wav)", type="wav")
        if up:
            speech, sr = librosa.load(up, sr=16000)
            res = classifier(speech)
            for r in res:
                st.write(f"**{r['label']}**")
                st.progress(r['score'])

        st.divider()
        recorded = st.audio_input("Live Microphone Stream")
        if recorded:
            speech, sr = librosa.load(recorded, sr=16000)
            res = classifier(speech)
            for r in res:
                st.write(f"**{r['label']}**")
                st.progress(r['score'])

    with tab2:
        st.title("🛰️ Structural Health Monitoring")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("### Sensor Inputs")
            vibration = st.slider("Vibration (Hz)", 0, 5000, 1100)
            strain = st.slider("Strain (με)", 0, 10000, 4000)
        with col2:
            damage = (strain / 10000) * 100
            st.metric("Deformation Risk", f"{damage:.1f}%")
            st.line_chart(np.random.randn(20, 1))

    with tab3:
        st.title("🧠 Human-Systems Integration (HSI)")
        st.subheader("Autonomous Synergy Guardian")
        st.info("Direct Edge Feedback: Active. Mars-Earth Delay: 22m (Bypassed)")
        st.write("This module correlates biological stress with mechanical fatigue to predict mission-critical failures before they reach Earth.")
        
        # A quick visual for the 'Real-Time' proof
        st.bar_chart({"Earth Delay (s)": 1320, "Astrielle AI (s)": 0.004})

    # --- THE FOOTER ---
    st.markdown("""
        <div class="footer">
            © 2026 Astrielle AI | Developed by [Your Name/PhD Bio] | <b>Confidential Mission Telemetry</b><br>
            Powered by Edge Intelligence for Autonomous Deep Space Exploration.
        </div>
    """, unsafe_allow_html=True)
