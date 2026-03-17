import streamlit as st
import librosa
import pandas as pd
import numpy as np
from transformers import pipeline

# --- 1. CONFIG ---
st.set_page_config(layout="wide", page_title="Astrielle AI")

# --- 2. SESSION STATE ---
if 'entered' not in st.session_state:
    st.session_state.entered = False

# --- 3. LANDING PAGE (Vibrant Color) ---
if not st.session_state.entered:
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.5)), 
                            url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
                background-size: cover;
                display: flex; align-items: center; justify-content: center;
            }
            .landing-card {
                text-align: center; color: white; padding: 60px;
                background: rgba(0, 0, 0, 0.4); 
                border-radius: 30px; backdrop-filter: blur(10px); 
                border: 1px solid rgba(255,255,255,0.2);
            }
            .title-text { font-size: 70px; font-weight: 800; letter-spacing: 10px; margin-bottom: 0px; }
            .subtitle-text { font-size: 20px; color: #00f2ff; letter-spacing: 3px; margin-bottom: 30px; }
        </style>
        <div class="landing-card">
            <div class="title-text">ASTRIELLE</div>
            <div class="subtitle-text">Autonomous Edge Intelligence</div>
            <p style="max-width:550px; margin:0 auto; font-size:18px; opacity:0.9;">
                Advanced <b>Human-Systems Integration</b> for Deep Space missions.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("INITIALIZE MISSION CONTROL", use_container_width=True):
        st.session_state.entered = True
        st.rerun()
    st.stop() 

# --- 4. DASHBOARD (Muted & Dark) ---
else:
    with st.sidebar:
        st.title("🛰️ Command")
        if st.button("Log Out"):
            st.session_state.entered = False
            st.rerun()
        st.divider()
        st.write("**Latency:** 0.004ms")

    # This CSS block is simplified to avoid the Syntax Error
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(rgba(10, 10, 15, 0.9), rgba(10, 10, 15, 0.9)), 
                            url('https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&q=80&w=2000');
                background-size: cover;
                background-attachment: fixed;
            }
            .stTabs [data-baseweb="tab-panel"] {
                padding: 30px; border-radius: 20px;
                background: rgba(255, 255, 255, 0.05);
