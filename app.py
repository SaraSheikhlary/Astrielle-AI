import streamlit as st
import librosa
import pandas as pd
import numpy as np
from transformers import pipeline

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Astrielle AI | HSI")

if 'entered' not in st.session_state:
    st.session_state.entered = False

# --- 2. THE SPLASH SCREEN ---
if not st.session_state.entered:
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                            url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
                background-size: cover; display: flex; align-items: center; justify-content: center;
            }
            .landing-card {
                text-align: center; color: white; padding: 60px;
                background: rgba(255, 255, 255, 0.05); border-radius: 30px;
                backdrop-filter: blur(15px); border: 1px solid rgba(255,255,255,0.1);
            }
            .title-text { font-size: 85px; font-weight: 800; letter-spacing: 12px; }
        </style>
        <div class="landing-card">
            <div class="title-text">ASTRIELLE</div>
            <p>Advanced Human-Systems Integration for Deep Space</p>
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

    # THEME & BACKGROUND
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
        
        # FIXED: Using the 3-letter codes the AI actually outputs
        emo_icons = {"ang": "😡", "sad": "😢", "hap": "😊", "neu": "😐", "fea": "😨"}

        # Logic for both Upload and Record
        source = st.file_uploader("Upload Telemetry (.wav)", type="wav")
        rec = st.audio_input("Or Live Stream Audio")
        
        active_file = source if source is not None else rec

        if active_file:
            speech, sr = librosa.load(active_file, sr=16000)
            results = classifier(speech)
            
            # Extract the raw 3-letter label and make sure it's lowercase
            top_emo = results[0]['label'].lower()
            
            # Show the top result with the correct Emoji
            st.subheader(f"AI Detected: {top_emo.upper()} {emo_icons.get(top_emo, '🛰️')}")
            
            # AI Feedback block updated to match 3-letter codes
            if top_emo in ["ang", "fea"]:
                st.error("⚠️ AI ALERT: Stress detected. Suggest immediate rest cycle.")
            elif top_emo == "hap":
                st.success("✅ AI STATUS: Optimal crew morale detected.")
            else:
                st.info("📡 AI STATUS: Crew biomarkers nominal.")

            # Show details with emojis mapped to every bar
            for r in results:
                lbl = r['label'].lower()
                st.write(f"**{lbl.upper()}** {emo_icons.get(lbl, '')}")
                st.progress(r['score'])

    with tab2:
        st.title("🛰️ Structural Health Monitoring")
        strn = st.slider("Hull Strain (με)", 0, 10000, 4000)
        st.metric("Deformation Risk", f"{strn/100}%", delta="Predictive")
        st.line_chart(np.random.randn(20, 1))

    with tab3:
        st.title("🧠 Human-Systems Integration")
        st.info("Direct Edge Feedback: Active. Mars-Earth Delay: 22m (Bypassed)")
        st.bar_chart({"Earth Delay (s)": 1320, "Astrielle AI (s)": 0.004})

    with tab4:
        st.title("📑 Mission Summary")
        st.subheader("What is Astrielle AI?")
        st.write("Astrielle is an **Autonomous Edge Intelligence** system. It processes data on the ship to bypass the 20-minute communication lag between Mars and Earth.")
        st.subheader("Safety Analytics")
        st.write("Current Safety Index: **98.4%**")
        st.write("The system uses structural telemetry and crew vocal biomarkers to predict mission risks before they become emergencies.")

    st.markdown('<div style="text-align:center; padding:10px; opacity:0.5;">© 2026 Astrielle AI</div>', unsafe_allow_html=True)
