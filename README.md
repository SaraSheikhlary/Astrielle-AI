# 🌌 Astrielle AI | Autonomous Edge Intelligence

**Advanced Human-Systems Integration (HSI) for Deep Space Exploration.**

Current Mars missions face a **22-minute delay** for signals to reach Earth. In a mission-critical emergency, waiting 44 minutes for a round-trip response from ground control is not an option. 

**Astrielle AI** solves this by moving intelligence to the *Edge*. By analyzing structural telemetry and crew vocal biomarkers locally on the spacecraft, Astrielle predicts and prevents mission failure in milliseconds, completely bypassing the Earth-Mars communication lag.

---

## ✨ Features

* **🎙️ Vocal Biomarker Monitor:** Utilizes a state-of-the-art Wav2Vec2 AI model to analyze crew voice telemetry (via `.wav` upload or live microphone). It detects psychological states (Stress, Fear, Morale) to ensure optimal human performance.
* **🛰️ Structural Health Monitoring:** Real-time predictive analytics tracking hull vibration and strain to calculate structural deformation risks before microscopic cracks become catastrophic failures.
* **🧠 HSI Synergy Dashboards:** Live comparisons of edge-computing latency (0.004ms) versus Earth-Sync delays (22m), proving the necessity of autonomous localized AI.
* **📑 Mission Intelligence Summary:** High-level metrics, including the current Autonomous Safety Index and system reliability scores (ROC-AUC).

---

## 🛠️ Technology Stack

* **Frontend & Framework:** [Streamlit](https://streamlit.io/)
* **Audio Processing:** [Librosa](https://librosa.org/)
* **Machine Learning / AI:** [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) (`pipeline`)
* **AI Model:** `superb/wav2vec2-base-superb-er` (Emotion Recognition)
* **Data Handling:** Pandas, NumPy

---

## 🚀 Installation & Setup

**1. Clone the repository or download the files:**
Ensure `app.py` is in your working directory.

**2. Install the required Python dependencies:**
You will need to install the core libraries and PyTorch (required for Hugging Face models).
```bash
pip install streamlit librosa pandas numpy transformers torch torchaudio
