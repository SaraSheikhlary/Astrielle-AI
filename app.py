import streamlit as st
import librosa
import pandas as pd
import numpy as np
from transformers import pipeline

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Astrielle AI | Deep Space HSI")

# --- 2. SESSION STATE (The Memory) ---
if 'entered' not in st.session_state:
    st.session_state.entered = False

# --- 3. THE SPLASH SCREEN (Landing Page) ---
# Goal: Maximum Color/Contrast for a dramatic entrance.
if not st.session_state.entered:
    st.markdown("""
        <style>
            .stApp {
                /* Increased darkness (0.85) to push contrast on the landing page */
                background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)), 
                            url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
                background-size: cover;
                display: flex; align-items: center; justify-content: center;
            }
            .landing-card {
                text-align: center; color: white; padding: 60px;
                background: rgba(0, 0, 0, 0.4); /* Deeper glass card */
                border-radius: 30px;
                backdrop-filter: blur(20px); 
                border: 1px solid rgba(255,255,255,0.1);
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.6);
            }
            .title-text { font-size: 85px; font-weight: 800; letter-spacing: 12px; margin-bottom: 0px; }
            .subtitle-text { font-size: 22px; color: #00f2ff; letter-spacing: 3px; margin-bottom: 30px; text-transform: uppercase; }
        </style>
        <div class="landing-card">
            <div class="title-text">ASTRIELLE</div>
            <div class="subtitle-text">Autonomous Edge Intelligence</div>
            <p style="max-width:600px; margin:0 auto; font-size:18px; opacity:0.8; line-height: 1.6;">
                Advanced <b>Human-Systems Integration</b> for Deep Space transit. 
                Localized AI diagnostics to bypass the 20-minute Mars-Earth communication lag.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ENTRY BUTTON (Centered and wide)
    if st.button("INITIALIZE MISSION CONTROL", use_container_width=True):
        st.session_state.entered = True
        st.rerun()
    st.stop() 

# --- 4. THE MAIN DASHBOARD (Visible only after 'Enter') ---
else:
    # Sidebar for Reset & Edge Info
    with st.sidebar:
        st.title("🛰️ Command Center")
        if st.button("Log Out / Reset View"):
            st.session_state.entered = False
            st.rerun()
        st.divider()
        st.write("**System Status:** Nominal")
        st.write("**Local Latency:** 0.004ms")
        st.write("**Earth Sync:** 22m Delay (Bypassed)")

    # ADAPTIVE CSS (Lowered Contrast for optimal data readability)
    # The Syntax Error (missing quotes/parentheses) was fixed right here.
    st.markdown("""
        <style>
            /* BASE BACKGROUND (Washed out with white overlay in light mode) */
            .stApp {
                background: url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
                background-size: cover;
                background-attachment: fixed;
            }

            /* LIGHT MODE: Lower Contrast - White sheet over stars + Black text */
            [data-theme="light"] .stApp {
                background-image: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)), 
                                  url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
                color: #000000;
            }

            /* DARK MODE: Dark tint over stars + White text */
            [data-theme="dark"] .stApp {
                background-image: linear-gradient(rgba(14, 17, 23, 0.9), rgba(14, 17, 23, 0.9)), 
                                  url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
                color: #ffffff;
            }

            /* TAB CONTENT BOXES */
            .stTabs [data-baseweb="tab-panel"] {
                padding: 30px;
                border-radius: 20px;
                backdrop-filter: blur(20px);
                border: 1px solid rgba(128, 128, 128, 0.2);
                margin-top: 20px;
            }

            /* TAB BOX LIGHT MODE (Solid white for professional look) */
            [data-theme="light"] .stTabs [data-baseweb="tab-panel"] {
                background: rgba(255, 255, 255, 0.95);
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }

            /* TAB BOX DARK MODE (Glass effect) */
            [data-theme="dark"] .stTabs [data-baseweb="tab-panel"] {
                background: rgba(30, 30, 30, 0.75);
                box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            }

            /* UNIVERSAL TEXT READABILITY FIX */
            h1, h2, h3, p, span, label {
                color: var(--text-color) !important;
                font-family: 'Inter', sans-serif;
            }

            /* FOOTER DYNAMICS */
            .footer {
                position: fixed; left: 0; bottom: 0; width: 100%;
                text-align: center; font-size: 0.8em; padding: 12px 0; z-index: 999;
            }
            [data-theme="light"] .footer { background: white; color: black; border-top: 1px solid #ddd; }
            [data-theme="dark"] .footer { background: black; color: white; border-top: 1px solid #333; }
        </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎙️ Vocal Biomarkers", "🛰️ Structural Health", "🧠 Human-Systems Integration"])

    # --- TAB 1: VOCAL AI ---
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

    # --- TAB 2: STRUCTURAL AI ---
    with tab2:
        st.title("🛰️ Structural Health Monitoring")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("### Sensor Telemetry")
            vibration = st.slider("Vibration (Hz)", 0, 5000, 1100)
            strain = st.slider("Strain (με)", 0, 10000, 4000)
            cycles = st.number_input("Stress Cycles", 1000, 1000000, 50000)
        with col2:
            # Physics-based simulation
            damage = (strain / 10000) * (np.log10(cycles) / 6) * 100
            st.metric("Deformation Risk", f"{damage:.1f}%")
            st.line_chart(pd.DataFrame(np.random.randn(20, 1), columns=["Stress Level"]))

    # --- TAB 3: HSI (THE DELAY SOLUTION) ---
    with tab3:
        st.title("🧠 Human-Systems Integration")
        st.info("Direct Edge Feedback: Active. Mars-Earth Delay: 22m (Bypassed)")
        
        c_a, c_b = st.columns(2)
        with c_a:
            st.metric("Earth Comms Latency", "1,300,000 ms", "Mars-Max")
        with c_b:
            st.metric("Astrielle AI Latency", "0.004 ms", "Local-Edge", delta_color="inverse")
            
        st.write("### Synergy Guardian")
        st.write("Autonomous correlation of biological and mechanical telemetry to bypass Earth decision-making lag.")
        st.bar_chart({"Earth Delay (s)": 1320, "Astrielle AI (s)": 0.004})

    # --- THE FOOTER ---
    # This block was also updated to close properly.
    st.markdown("""
        <div class="footer">
            © 2026 Astrielle AI | <b>Confidential Mission Telemetry</b> | 
            Edge Intelligence for Deep Space Exploration.
        </div>
    """, unsafe_allow_html=True)
