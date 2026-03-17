import streamlit as st
import librosa
import pandas as pd
import numpy as np
import time
from transformers import pipeline

# --- 1. CONFIG & COSMIC THEME ---
st.set_page_config(layout="wide", page_title="Astrielle AI | Space Biotech")

st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(rgba(14, 17, 23, 0.8), rgba(14, 17, 23, 0.8)), 
                        url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
            background-size: cover;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stMetric { background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; }
        .footer {
            position: fixed; left: 0; bottom: 0; width: 100%;
            background-color: rgba(14, 17, 23, 0.9); color: grey;
            text-align: center; font-size: 0.8em; padding: 15px 0; z-index: 999;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. TABS NAVIGATION ---
tab1, tab2, tab3 = st.tabs([
    "🎙️ Vocal Biomarker Monitor", 
    "🛰️ Structural Health Monitoring", 
    "🧠 Human-Systems Integration"
])

# --- 3. TAB 1: VOCAL BIOMARKER MONITOR ---
with tab1:
    st.title("✨ Vocal Biomarker Monitor")
    st.write("Real-time cognitive and physiological state analysis via voice.")

    @st.cache_resource
    def load_voice_model():
        return pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")

    classifier = load_voice_model()

    st.sidebar.header("✨ Fusion Weights")
    v_weight = st.sidebar.slider("Voice Importance", 0.0, 1.0, 0.7)
    f_weight = 1.0 - v_weight

    uploaded_file = st.file_uploader("Upload Voice Clip (.wav)", type="wav")
    
    # Initialize variables to avoid "not defined" errors
    final_fusion_score = 0.0

    if uploaded_file:
        speech, sr = librosa.load(uploaded_file, sr=16000)
        result = classifier(speech)

        v_score = 0.0
        for r in result:
            if r['label'] == 'ang':
                v_score = r['score']

        f_score = st.slider("Simulated Face Stress (Adjust me!)", 0.0, 1.0, 0.4)
        final_fusion_score = (v_score * v_weight) + (f_score * f_weight)

        st.subheader("Temporal Analysis")
        time_points = np.linspace(0, 10, 20)
        stress_trend = [final_fusion_score + np.random.normal(0, 0.05) for _ in time_points]
        df = pd.DataFrame({"Time (s)": time_points, "Stress Level": stress_trend})
        st.line_chart(df.set_index("Time (s)"))

        if final_fusion_score > 0.6:
            st.error(f"SYSTEM ALERT: High Stress Detected ({final_fusion_score:.2%})")
        else:
            st.success(f"SYSTEM STATUS: Calm/Stable ({final_fusion_score:.2%})")

    # MICROPHONE SECTION
    recorded_voice = st.audio_input("Record your voice for AI analysis")
    if recorded_voice:
        with st.spinner("Astrielle AI is analyzing your live voice..."):
            speech, sr = librosa.load(recorded_voice, sr=16000)
            result = classifier(speech)
            
            label_map = {
                'ang': '🔥 High Stress (Angry)', 'hap': '😊 Happy / Positive',
                'sad': '😢 Sad / Low Energy', 'neu': '😐 Neutral / Calm', 'fea': '😨 Fearful / Anxious'
            }
            
            st.subheader("Live Analysis Results:")
            for r in result:
                clean_name = label_map.get(r['label'], r['label']) 
                st.write(f"**{clean_name}**")
                st.progress(r['score'])

# --- 4. TAB 2: STRUCTURAL HEALTH MONITORING (SHM) ---
with tab2:
    st.title("🛰️ Structural Health Monitoring")
    st.subheader("Predictive Damage & Deformation AI")
    
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
            st.success(f"Nominal: {damage_score:.2f}% Damage Probability")
        elif damage_score < 70:
            st.warning(f"Caution: {damage_score:.2f}% Damage Probability")
        else:
            st.error(f"CRITICAL: {damage_score:.2f}% Probability of Failure")
            
        chart_data = pd.DataFrame(np.random.randn(20, 2), columns=['Stress', 'Strain'])
        st.line_chart(chart_data)

# --- 5. TAB 3: HUMAN-SYSTEMS INTEGRATION (HSI) ---
with tab3:
    st.title("🧠 Human-Systems Integration")
    st.write("Autonomous Synergy Guardian: Human-Machine Feedback Loop.")
    st.info("System initializing... Building Cross-Domain correlation engine.")

# --- FOOTER ---
st.markdown("""
    <div class="footer">
        © 2026 Astrielle AI | <b>Privacy & Terms</b><br>
        This app uses AI-generated data for monitoring. We do not store personal user data.
    </div>
""", unsafe_allow_html=True)
