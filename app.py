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
        <p>Astrielle AI provides localized AI diagnostics, vocal biomarker monitoring, and structural health tracking.</p>
    </div>
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
        st.write("**Earth Sync:** 22m Delay (Bypassed)")

    # THEME GUARD CSS 
    st.markdown("""
        <style>
            .stApp {
                background: url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
                background-size: cover;
                background-attachment: fixed;
            }
            [data-theme="light"] .stApp {
                background-image: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)), 
                                  url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
            }
            [data-theme="dark"] .stApp {
                background-image: linear-gradient(rgba(14, 17, 23, 0.85), rgba(14, 17, 23, 0.85)), 
                                  url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
            }
            .stTabs [data-baseweb="tab-panel"] {
                padding: 30px; border-radius: 20px; backdrop-filter: blur(20px);
                border: 1px solid rgba(128, 128, 128, 0.2); margin-top: 20px;
            }
            [data-theme="light"] .stTabs [data-baseweb="tab-panel"] { background: rgba(255, 255, 255, 0.95); }
            [data-theme="dark"] .stTabs [data-baseweb="tab-panel"] { background: rgba(30, 30, 30, 0.75); }

            .footer {
                position: fixed; left: 0; bottom: 0; width: 100%;
                text-align: center; font-size: 0.8em; padding: 12px 0; z-index: 999;
            }
            [data-theme="light"] .footer { background: white; color: black; border-top: 1px solid #ddd; }
            [data-theme="dark"] .footer { background: black; color: white; border-top: 1px solid #333; }

            /* --- PURE BLACK SIDEBAR --- */
            [data-testid="stSidebar"] { background-color: #000000 !important; }
            /* --- BLACK UPLOAD & AUDIO WIDGETS --- */
            [data-testid="stFileUploadDropzone"] { background-color: rgba(0, 0, 0, 0.7) !important; border: 1px solid rgba(255, 255, 255, 0.2) !important; }
            [data-testid="stAudioInput"] { background-color: rgba(0, 0, 0, 0.7) !important; border: 1px solid rgba(255, 255, 255, 0.2) !important; border-radius: 8px; padding: 10px; }
        </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎙️ Vocal Biomarkers", 
        "🗣️ Crew Synergy", 
        "🛰️ Structural Health", 
        "🧠 Human-Systems Integration", 
        "📑 Summary"
    ])

    with tab1:
        st.title("✨ Vocal Biomarker Monitor")

        @st.cache_resource
        def load_voice_model():
            return pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")

        classifier = load_voice_model()
        
        emo_icons = {"ang": "😬", "sad": "😢", "hap": "😊", "neu": "😐", "fea": "😨", "sur": "😲"}
        emo_names = {"ang": "Angry (high stress)", "sad": "Sad", "hap": "Happy", "neu": "Neutral", "fea": "Fear", "sur": "Surprise"}

        source = st.file_uploader("Upload Telemetry (.wav)", type="wav")
        rec = st.audio_input("Live Stream")
        
        active_file = source if source is not None else rec

        if active_file:
            speech, sr = librosa.load(active_file, sr=16000)
            results = classifier(speech)
            
            top_emo = results[0]['label'].lower()
            detected_text = "AI Detected: " + emo_names.get(top_emo, top_emo.upper()).upper() + " " + emo_icons.get(top_emo, "🛰️")
            st.subheader(detected_text)
            
            if top_emo in ["ang", "fea"]:
                st.error("⚠️ AI ALERT: Stress detected. Suggest immediate rest cycle.")
            elif top_emo == "hap":
                st.success("✅ AI STATUS: Optimal crew morale detected.")
            else:
                st.info("📡 AI STATUS: Crew biomarkers nominal.")

            for r in results:
                lbl = r['label'].lower()
                full_lbl = emo_names.get(lbl, lbl.upper()).upper()
                icon = emo_icons.get
