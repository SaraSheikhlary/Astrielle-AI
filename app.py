import streamlit as st
import librosa
import pandas as pd
import numpy as np
import time
from transformers import pipeline


# --- 1. Load the Voice AI (The part that works!) ---
@st.cache_resource
def load_voice_model():
    return pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")


classifier = load_voice_model()

# --- 2. Website Setup ---
st.markdown("<h1 style='text-align: center;'>✨ Astrielle AI: State Monitor ✨</h1>", unsafe_allow_html=True)
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


    st.subheader("🎤 Live Voice Input")
audio_value = st.audio_input("Record your voice to analyze")

if audio_value:
    st.audio(audio_value)
    st.success("Recording captured! Processing...")
    # This is where your AI analysis code will go

    # --- 4. FUSION LAYER ---
    # Since Face AI is stuck, we simulate a "Face Vector"
    # to show us how the Fusion logic handles conflicting data
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

#Voice Record
import streamlit as st

# 1. SETUP (Always at the top)
st.set_page_config(page_title="Astrielle AI", layout="centered")

# 2. TITLE & HEADER
st.title("Astrielle AI: State Monitor")
st.write("Welcome to your professional analysis dashboard.")

# 3. THE MICROPHONE (This is where the code goes!)
st.subheader("🎤 Voice Input")
audio_input = st.audio_input("Record a test clip")

if audio_input:
    st.audio(audio_input)
    st.download_button(
        label="📥 Download Recording",
        data=audio_input,
        file_name="recording.wav",
        mime="audio/wav"
    )


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
