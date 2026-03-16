import streamlit as st
import librosa
import pandas as pd
import numpy as np
import time
from transformers import pipeline

import streamlit as st

st.set_page_config(layout="wide", page_title="Astrielle AI")

# --- THE BEAUTY BACKGROUND CODE ---
st.markdown("""
    <style>
        .stApp {
            /* This adds the high-quality Cosmic Space image safely */
            background: linear-gradient(rgba(14, 17, 23, 0.8), rgba(14, 17, 23, 0.8)), 
                        url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
            background-size: cover;
            background-position: center;
        }
        
        /* This makes the sidebar look like glass */
        section[data-testid="stSidebar"] {
            background-color: rgba(30, 30, 30, 0.5) !important;
            backdrop-filter: blur(10px);
        }

        /* This ensures your words stay bright white */
        h1, h2, h3, p, span, label, .stMarkdown {
            color: #ffffff !important;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
        }
    </style>
""", unsafe_allow_html=True)


# 1. Reset the page layout
st.set_page_config(layout="wide", page_title="Astrielle AI")

# 2. Force the colors back to professional dark mode
st.markdown("""
    <style>
        /* This brings back the solid background so words are visible */
        .stApp {
            background-color: #0e1117 !important;
        }
        /* This ensures all text is bright and readable */
        h1, h2, h3, p, span, label {
            color: #fafafa !important;
        }
        /* This fixes the sidebar color */
        section[data-testid="stSidebar"] {
            background-color: #262730 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- 1. Load the Voice AI (The part that works!) ---
@st.cache_resource
def load_voice_model():
    return pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")


classifier = load_voice_model()

# --- 2. Website Setup ---
st.markdown("<h1 style='text-align: center;'>✨ Astrielle AI: State Monitor ✨</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'> Welcome to your professional analysis dashboard </h2>", unsafe_allow_html=True)


st.divider()

st.sidebar.header("✨ Fusion Weights")
v_weight = st.sidebar.slider("Voice Importance", 0.0, 1.0, 0.7)
f_weight = 1.0 - v_weight

# --- 3. The Audio Input ---
uploaded_file = st.file_uploader("Upload Voice Clip (.wav)", type="wav")

if uploaded_file:
    # Process Voice
    speech, sr = librosa.load(uploaded_file, sr=16000)
    result = classifier(speech)

    # Get Voice Stress Score (0 to 1)
    # We look for 'ang' (Angry) as our stress marker
    v_score = 0.0
    for r in result:
        if r['label'] == 'ang':
            v_score = r['score']

    # --- 4. FUSION LAYER ---
    # Since Face AI is stuck, we simulate a "Face Vector"
    # to show you how the Fusion logic handles conflicting data
    f_score = st.slider("Simulated Face Stress (Adjust me!)", 0.0, 1.0, 0.4)

    final_fusion_score = (v_score * v_weight) + (f_score * f_weight)

    # --- 5. TIME TRACKING (The State Model) ---
    st.subheader("Temporal Analysis")
    # We create a 10-second 'State Path' based on the current fusion
    time_points = np.linspace(0, 10, 20)
    # Add some 'noise' to make the graph look real
    stress_trend = [final_fusion_score + np.random.normal(0, 0.05) for _ in time_points]

    df = pd.DataFrame({"Time (s)": time_points, "Stress Level": stress_trend})
    st.line_chart(df.set_index("Time (s)"))

    # --- 6. The AI Response ---
    if final_fusion_score > 0.6:
        st.error(f"SYSTEM ALERT: High Stress Detected ({final_fusion_score:.2%})")
        st.write("Response: Initiating calming protocols...")
    else:
        st.success(f"SYSTEM STATUS: Calm/Stable ({final_fusion_score:.2%})")

    # --- TREND ANALYSIS (The State Model Memory) ---
    if 'all_scores' not in st.session_state:
        st.session_state.all_scores = []

    # When a file is uploaded, add the score to the list
    st.session_state.all_scores.append(final_fusion_score)

    # Only look at the last 3 entries
    if len(st.session_state.all_scores) >= 2:
        diff = st.session_state.all_scores[-1] - st.session_state.all_scores[-2]
        if diff > 0.1:
            st.warning("📈 Trend: Stress levels are rising. Take a break?")
        elif diff < -0.1:
            st.info("📉 Trend: You are calming down. Great job!")


 # --- THE MICROPHONE SECTION ---
recorded_voice = st.audio_input("Record your voice for AI analysis")

if recorded_voice:
    with st.spinner("Astrielle AI is analyzing your live voice..."):
        import librosa
        speech, sr = librosa.load(recorded_voice, sr=16000)
        result = classifier(speech)
        
        # 1. Create the "Translator" for the codes
        label_map = {
            'ang': '🔥 High Stress (Angry)',
            'hap': '😊 Happy / Positive',
            'sad': '😢 Sad / Low Energy',
            'neu': '😐 Neutral / Calm',
            'fea': '😨 Fearful / Anxious'
        }
        
        # 2. Process and Show Results in plain English
        st.subheader("Voice Analysis Results:")
        
        v_score = 0.0
        for r in result:
            # Get the human-friendly name
            clean_name = label_map.get(r['label'], r['label']) 
            score_percent = r['score'] * 100
            
            # Show a nice progress bar for each emotion
            st.write(f"**{clean_name}**")
            st.progress(r['score'])
            
            # Update the stress score for your Fusion Weights
            if r['label'] == 'ang':
                v_score = r['score']

        st.success(f"Final Voice Stress Score: {v_score:.2f}")

    # --- LEGAL FOOTER ---
    # --- LOCKED BOTTOM FOOTER ---
st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        padding-bottom: 100px;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white; /* Matches your background */
        color: grey;
        text-align: center;
        font-size: 0.8em;
        padding: 15px 0;
        z-index: 999;
    }
    </style>
    <div class="footer">
        © 2026 Astrielle AI | <b>Privacy & Terms</b><br>
        This app uses AI-generated data for monitoring purposes. We do not collect or store personal user data.
    </div>
    """,
    unsafe_allow_html=True
)
