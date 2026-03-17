import streamlit as st
import librosa
import pandas as pd
import numpy as np
from transformers import pipeline

# --- 1. CONFIG ---
st.set_page_config(layout="wide", page_title="Astrielle AI")

# --- 2. INITIALIZE SESSION STATE ---
if 'entered' not in st.session_state:
    st.session_state.entered = False

# --- 3. THE SPLASH SCREEN ---
if not st.session_state.entered:
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
                            url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
                background-size: cover;
                display: flex; align-items: center; justify-content: center;
            }
            .landing-card {
                text-align: center; color: white; padding: 50px;
                background: rgba(0,0,0,0.4); border-radius: 20px;
                backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1);
            }
            h1 { font-size: 70px; letter-spacing: 8px; margin-bottom: 10px; }
            p { font-size: 20px; opacity: 0.8; margin-bottom: 40px; }
        </style>
        <div class="landing-card">
            <h1>ASTRIELLE AI</h1>
            <p>Autonomous Edge Intelligence for Deep Space Missions</p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("ENTER MISSION CONTROL", use_container_width=True):
        st.session_state.entered = True
        st.rerun()

else:
    # --- 4. THE MAIN DASHBOARD ---
    st.markdown("""
        <style>
            [data-theme="light"] { color: #000000 !important; }
            [data-theme="dark"] { color: #ffffff !important; }
            .stApp {
                background: linear-gradient(rgba(14, 17, 23, 0.85), rgba(14, 17, 23, 0.85)), 
                            url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
                background-size: cover;
            }
        </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "🎙️ Vocal Biomarker Monitor", 
        "🛰️ Structural Health Monitoring", 
        "🧠 Human-Systems Integration"
    ])

    # --- TAB 1: VOCAL AI ---
    with tab1:
        st.title("✨ Vocal Biomarker Monitor")
        
        @st.cache_resource
        def load_voice_model():
            return pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")

        classifier = load_voice_model()

        uploaded_file = st.file_uploader("Upload Voice Clip (.wav)", type="wav")
        if uploaded_file:
            speech, sr = librosa.load(uploaded_file, sr=16000)
            result = classifier(speech)
            for r in result:
                st.write(f"**{r['label']}**: {r['score']:.2%}")
                st.progress(r['score'])

        st.divider()
        recorded_voice = st.audio_input("Record live voice analysis")
        if recorded_voice:
            speech, sr = librosa.load(recorded_voice, sr=16000)
            result = classifier(speech)
            for r in result:
                st.write(f"**{r['label']}**: {r['score']:.2%}")
                st.progress(r['score'])

    # --- TAB 2: STRUCTURAL AI ---
    with tab2:
        st.title("🛰️ Asset Integrity Analysis")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("### Sensor Telemetry")
            vibration = st.slider("Vibration Frequency (Hz)", 0, 5000, 1200)
            strain = st.slider("Micro-strain (με)", 0, 10000, 4500)
            cycles = st.number_input("Cumulative Stress Cycles", 1000, 1000000, 50000)

        with col2:
            damage_score = (strain / 10000) * (np.log10(cycles) / 6) * 100
            st.write("### Damage Probability Analysis")
            if damage_score < 30:
                st.success(f"Nominal: {damage_score:.2f}% Damage")
            elif damage_score < 70:
                st.warning(f"Caution: {damage_score:.2f}% Damage")
            else:
                st.error(f"CRITICAL: {damage_score:.2f}% Failure Probability")
            
            st.line_chart(pd.DataFrame(np.random.randn(20, 2), columns=['X-Axis', 'Y-Axis']))

    # --- TAB 3: HSI (THE DELAY SOLUTION) ---
    with tab3:
        st.title("🧠 Human-Systems Integration")
        st.subheader("Autonomous Synergy Guardian")
        
        # This explains your PhD "Why"
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Earth Signal Latency", "1,200,000 ms", "Mars Delay")
        with col_b:
            st.metric("Astrielle AI Latency", "4.2 ms", "Instant Action", delta_color="inverse")
            
        st.info("Local Edge Computing Mode Active: No data is sent to Earth for processing.")
        st.write("By integrating biometric state with structural integrity, we predict human-error risks before they occur.")
