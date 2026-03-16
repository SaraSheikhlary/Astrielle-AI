import librosa
from transformers import pipeline

# 1. Load the AI
print("Loading AI Model (this may take a moment)...")
classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er"
)

# 2. Use Capital letters to match your 'Audio' folder and '.wav' files
audio_files = [
    "Audio/Angry.wav",
    "Audio/Happy.wav",
    "Audio/Neutral.wav",
    "Audio/Sad.wav"
]

for file_path in audio_files:
    print(f"\n--- Analyzing: {file_path} ---")
    try:
        # PLAN B: Load audio with librosa (This avoids the FFmpeg error!)
        audio_input, sampling_rate = librosa.load(file_path, sr=16000)

        # Pass the raw audio numbers to the AI
        results = classifier(audio_input)

        # Show the top result
        top = results[0]
        print(f"Result: {top['label'].upper()} (Confidence: {top['score']:.2%})")

    except FileNotFoundError:
        print(f"❌ Error: Could not find '{file_path}'. Check your folder name spelling!")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\nDone!")

import librosa
from transformers import pipeline

# 1. Initialize the AI
print("Loading AI Model...")
classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er"
)

# 2. List of all the Audio files to check
audio_files = [
    "Audio/Angry.wav",
    "Audio/Happy.wav",
    "Audio/Neutral.wav",
    "Audio/Sad.wav"
]

# 3. The Loop: This tells Python to do the next steps for each file
for file_path in audio_files:
    print(f"\n--- Analyzing: {file_path} ---")
    try:
        # Load the audio (Plan B method)
        speech, sr = librosa.load(file_path, sr=16000)

        # Get the prediction
        result = classifier(speech)
        top_label = result[0]['label']

        # 4. Your Logic: Stress vs. Calm
        # 'ang' = Angry, 'hap' = Happy, 'neu' = Neutral, 'sad' = Sad
        if top_label == "ang":
            emotion = "STRESS"
            response = "I hear tension in your voice. Try taking a slow breath with me."
        else:
            emotion = "CALM"
            response = "You sound calm. How can I help you today?"

        print(f"Result: {top_label.upper()} -> classified as {emotion}")
        print(f"AI Response: {response}")

    except Exception as e:
        print(f"Could not process {file_path}: {e}")

print("\n--- All files analyzed! ---")

import time
import numpy as np

# 1. Setup the State Tracking
emotion_history = []


def get_fused_emotion(voice_score, face_score):
    # This is a simple 'Late Fusion' logic
    # It averages the scores from both sensors
    combined_score = (voice_score + face_score) / 2
    return "STRESS" if combined_score > 0.5 else "CALM"


# 2. The "Time Tracker" Loop
for second in range(10):  # Imagine tracking for 10 seconds
    # Get the vectors (simplified for this example)
    v_score = 0.8  # AI says Voice is stressed
    f_score = 0.4  # AI says Face looks calm

    current_state = get_fused_emotion(v_score, f_score)
    emotion_history.append(current_state)

    # Track "State Model"
    if emotion_history[-3:] == ["STRESS", "STRESS", "STRESS"]:
        print("⚠️ Warning: Sustained Stress Detected Over Time!")

    time.sleep(1)  # Wait for the next second