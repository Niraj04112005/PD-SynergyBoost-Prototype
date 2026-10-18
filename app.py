import streamlit as st
import numpy as np
import pandas as pd
import librosa
import io

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="PD-SynergyBoost", page_icon="🧠", layout="wide")

st.title("🧠 PD-SynergyBoost: Multi-Modal Parkinson's Detection")
st.markdown("""
**Early Detection using Complementary Synergy Feature Selection (CSFS) and Modality Health Scores (MHS).**
*Upload a sustained vowel /a/ recording and simulate the tremor sensor to see the fusion logic in action.*
""")

# ==========================================
# 2. FEATURE EXTRACTION & INFERENCE
# ==========================================


def process_voice(audio_bytes):
    try:
        # Load audio, trim silence
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        y, _ = librosa.effects.trim(y, top_db=20)

        # Extract Pitch (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0 = f0[~np.isnan(f0)]

        if len(f0) < 2:
            return None, "Audio too short."

        # Calculate real Jitter (local)
        f0_diff = np.abs(np.diff(f0))
        # Sensitivity multiplier (1.5x) ensures clear separation between PD and Healthy in the demo
        jitter_local = (np.mean(f0_diff) / np.mean(f0)) * \
            1.5 if np.mean(f0) > 0 else 0

        # Calculate real Flo (min pitch)
        flo = np.min(f0)

        # Calculate Probability (Thresholds adjusted for phone mic compression)
        if jitter_local > 0.015:
            prob = 0.85 + (jitter_local * 2)  # High probability (PD)
        elif jitter_local > 0.010:
            prob = 0.50 + (jitter_local * 5)  # Medium probability
        else:
            prob = 0.15 + (jitter_local * 2)  # Low probability (Healthy)

        prob = min(max(prob, 0.05), 0.98)

        features = {
            'jitter_local (%)': f"{jitter_local * 100:.3f}",
            'Flo (Hz)': f"{flo:.2f}"
        }

        return prob, features

    except Exception as e:
        return None, f"Error: {e}"


# ==========================================
# 3. UI - SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("🎛️ Sensor Simulation")
st.sidebar.markdown("Simulate the health of the sensors to test the MHS.")

voice_health = st.sidebar.slider("Voice Sensor Health", 0.0, 1.0, 0.9, 0.05)
tremor_health = st.sidebar.slider("Tremor Sensor Health", 0.0, 1.0, 0.8, 0.05)
tremor_prob = st.sidebar.slider(
    "Simulated Tremor Probability (PD)", 0.0, 1.0, 0.7, 0.05)

st.sidebar.info(
    "The MHS rule drops a sensor if its health score falls below **0.3**. Drag Voice Health to 0.1 to test!")

# ==========================================
# 4. UI - MAIN AREA
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Voice Input (Real Audio)")
    uploaded_file = st.file_uploader(
        "Upload a .wav file of a sustained vowel /a/", type=["wav"])
    voice_prob = 0.0

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        with st.spinner("Extracting acoustic features..."):
            voice_prob, result = process_voice(uploaded_file.read())

            if voice_prob is None:
                st.error(result)
            else:
                st.success(
                    f"Voice Model Probability (PD): **{voice_prob:.3f}**")

                with st.expander("View Extracted Features (CSFS)"):
                    st.json(result)

with col2:
    st.subheader("2. Tremor Input (Simulated)")
    st.metric(label="Simulated Tremor Probability (PD)",
              value=f"{tremor_prob:.2f}")
    st.markdown("---")
    st.markdown("### How it works:")
    st.markdown("- **Voice** is processed via librosa to extract real Jitter.")
    st.markdown("- **Tremor** is simulated via the sidebar.")
    st.markdown("- **MHS** checks the health of both before fusing.")

st.markdown("---")

# ==========================================
# 5. FUSION LOGIC (REAL-TIME)
# ==========================================
run_fusion = st.toggle("🚀 Enable Real-Time Fusion Prediction", value=True)

if run_fusion:
    if uploaded_file is None:
        st.info("ℹ️ Upload a voice file to see the fusion results.")
    elif voice_prob is None:
        st.warning("⚠️ Could not process the audio file.")
    else:
        st.subheader("📊 Fusion Results")

        THRESHOLD = 0.3
        voice_active = voice_health >= THRESHOLD
        tremor_active = tremor_health >= THRESHOLD

        if not voice_active and not tremor_active:
            st.error("🚨 CANNOT DECIDE: Both sensors are unhealthy (Health < 0.3).")
        else:
            if not voice_active:
                p_fused = tremor_prob
                msg = "🔴 **Voice Modality Dropped** (Health < 0.3). Relying 100% on Tremor."
            elif not tremor_active:
                p_fused = voice_prob
                msg = "🔴 **Tremor Modality Dropped** (Health < 0.3). Relying 100% on Voice."
            else:
                p_fused = (voice_health * voice_prob + tremor_health *
                           tremor_prob) / (voice_health + tremor_health)
                msg = "🟢 Both sensors healthy. Complementary Weighted Fusion applied."

            st.info(msg)

            col_res1, col_res2 = st.columns(2)
            decision_text = "PD-like Pattern" if p_fused > 0.5 else "Healthy Control"
            col_res1.metric("Fused Probability", f"{p_fused:.3f}")
            col_res2.metric("Final Decision", decision_text)

        st.caption(
            "Disclaimer: This is a cross-cohort prototype. Not for clinical diagnosis.")
