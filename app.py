"""
PD-SynergyBoost - Streamlit demo (standalone: no .pkl, no pd_common.py)
=======================================================================

Install (Windows / VS Code terminal):
    pip install streamlit librosa praat-parselmouth
    pip install nolds          # optional: enables the D2 feature (shows n/a without it)

Run:
    streamlit run app.py

WHAT THIS DEMO IS (and is not) - this is shown inside the app as well:
  * The voice probability comes from a simple JITTER RULE (illustrative thresholds), NOT from the
    trained stacking model. The thresholds are not fitted or validated on data.
  * The tremor probability and both health scores are manual slider inputs (simulated).
  * Voice and tremor come from unrelated cohorts: this demonstrates the fusion LOGIC only.
  * Not a diagnostic tool.
"""
from __future__ import annotations

import io
from typing import Any, Dict, Optional

import librosa
import numpy as np
import pandas as pd
import parselmouth
import streamlit as st
from parselmouth.praat import call

# ----------------------------------------------------------------------------- constants
F_MIN, F_MAX = 60.0, 500.0         # Hz, same pitch range as training
MAX_SECONDS = 8.0                  # same crop as training
MIN_SR = 6000                      # telephone audio is 8 or 16 kHz
F0_REF = 127.14                    # Hz, semitone reference used in training for D2
HEALTH_THRESHOLD = 0.3
CSFS_FEATURES = ["jitter_local", "Flo", "D2", "jitter_ddp", "jitter_rap", "pr_shimmer_apq11"]

DISCLOSURE = (
    "**Demo mode.** The voice probability is computed by a simple **jitter rule with illustrative "
    "thresholds** (not the trained model, not validated). Tremor probability and health scores are "
    "manual inputs. Voice and tremor come from **unrelated cohorts**, so this shows the fusion "
    "logic only. **Not a diagnosis.**")


# ----------------------------------------------------------------------------- audio analysis
class AudioError(ValueError):
    """Recording cannot be analysed (too short, silent, unreadable, no voiced pulses)."""


def _finite(v: Any) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


@st.cache_data(show_spinner="Analysing voice ...")
def analyze_audio(audio_bytes: bytes) -> Dict[str, Any]:
    """Uploaded WAV bytes -> {'features': {...}} or {'error': msg}.

    Cached on the file bytes, so moving a slider (which reruns the script) does not redo Praat.
    librosa decodes the bytes into a numpy array; Praat receives ONLY that array (never BytesIO).
    """
    try:
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        except Exception as e:
            raise AudioError(f"Cannot read this file as WAV audio ({type(e).__name__}).") from e
        if y.size == 0:
            raise AudioError("The audio file is empty.")
        if sr < MIN_SR:
            raise AudioError(f"Sampling rate {sr} Hz is too low (need 8 kHz or higher).")
        y = y - np.mean(y)
        y, _ = librosa.effects.trim(y, top_db=30)
        if y.size / sr < 0.5:
            raise AudioError("Recording is shorter than 0.5 s after removing silence.")
        peak = float(np.max(np.abs(y)))
        if peak < 1e-6:
            raise AudioError("Recording is silent.")
        y = (0.9 * y / peak)[: int(MAX_SECONDS * sr)].astype(np.float64)

        snd = parselmouth.Sound(y, sampling_frequency=float(sr))   # numpy array, not BytesIO

        feats = {k: float("nan") for k in CSFS_FEATURES}
        pp = call(snd, "To PointProcess (periodic, cc)", F_MIN, F_MAX)
        if call(pp, "Get number of points") < 5:
            raise AudioError("Too few voiced pulses found: use a steady sustained vowel /a/.")
        a = (0, 0, 0.0001, 0.02, 1.3)
        feats["jitter_local"] = _finite(call(pp, "Get jitter (local)", *a))
        feats["jitter_rap"] = _finite(call(pp, "Get jitter (rap)", *a))
        feats["jitter_ddp"] = _finite(call(pp, "Get jitter (ddp)", *a))      # = 3 x RAP by definition
        feats["pr_shimmer_apq11"] = _finite(
            call([snd, pp], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6))

        pitch = snd.to_pitch_ac(pitch_floor=F_MIN, pitch_ceiling=F_MAX)
        f0 = pitch.selected_array["frequency"]
        f0 = f0[f0 > 0]
        if f0.size:
            feats["Flo"] = float(np.min(f0))

        try:                                    # D2 needs the optional 'nolds' package
            import nolds
            if f0.size >= 64:
                st_series = 12.0 * np.log2(f0 / F0_REF)
                if np.std(st_series) > 1e-9:
                    feats["D2"] = _finite(nolds.corr_dim(
                        (st_series - st_series.mean()) / st_series.std(), emb_dim=4, fit="poly"))
        except Exception:
            pass

        if not np.isfinite(feats["jitter_local"]):
            raise AudioError("Praat could not measure jitter on this recording.")
        return {"features": feats, "duration_s": float(y.size / sr), "sr": int(sr)}
    except AudioError as e:
        return {"error": str(e)}
    except Exception as e:                      # unexpected library failure: show, don't crash
        return {"error": f"Analysis failed: {type(e).__name__}: {e}"}


# ----------------------------------------------------------------------------- rule-based voice score
def voice_probability(jitter_local: float) -> Optional[float]:
    """ILLUSTRATIVE rule (not a trained model). Jitter is a fraction (0.01 = 1 %).
        > 0.012          -> 0.75 .. 0.95  (reaches 0.95 at 0.024)
        0.008 .. 0.012   -> 0.45 .. 0.75
        < 0.008          -> 0.05 .. 0.30
    Linear inside each band. Returns None if jitter is NaN."""
    j = _finite(jitter_local)
    if not np.isfinite(j):
        return None
    if j > 0.012:
        return float(0.75 + 0.20 * min(1.0, (j - 0.012) / 0.012))
    if j > 0.008:
        return float(0.45 + 0.30 * (j - 0.008) / 0.004)
    return float(0.05 + 0.25 * max(0.0, j) / 0.008)


# ----------------------------------------------------------------------------- MHS fusion
def fuse(p_voice: Optional[float], p_tremor: float, h_voice: float, h_tremor: float,
         use_mhs: bool = True, threshold: float = HEALTH_THRESHOLD) -> Dict[str, Any]:
    """MHS rule. A modality with no voice probability (no/invalid audio) counts as unavailable.
        both healthy : (h_v*p_v + h_t*p_t) / (h_v + h_t)
        one < 0.3    : 100 % of the other
        both < 0.3   : cannot decide
    use_mhs=False    : plain average of the available modalities (ignores health)."""
    v_ok = p_voice is not None and (h_voice >= threshold or not use_mhs)
    t_ok = h_tremor >= threshold or not use_mhs
    if not use_mhs:
        vals = [p for p in (p_voice, p_tremor) if p is not None]
        p = float(np.mean(vals))
        return _result(p, "No MHS: plain average" if len(vals) == 2 else "No MHS: tremor only",
                       0.5 if len(vals) == 2 else 0.0, 0.5 if len(vals) == 2 else 1.0, None)
    if v_ok and t_ok:
        s = h_voice + h_tremor
        wv, wt = (h_voice / s, h_tremor / s) if s > 0 else (0.5, 0.5)
        return _result(wv * p_voice + wt * p_tremor, "Both healthy: weighted average", wv, wt, None)
    if v_ok:
        return _result(p_voice, "Tremor unhealthy: dropped", 1.0, 0.0, "tremor")
    if t_ok:
        why = "Voice unhealthy: dropped" if p_voice is not None else "No usable voice: dropped"
        return _result(p_tremor, why, 0.0, 1.0, "voice")
    return {"decision": None, "p": None, "mode": "Cannot decide - both modalities unusable",
            "w_voice": 0.0, "w_tremor": 0.0, "dropped": "both"}


def _result(p: float, mode: str, wv: float, wt: float, dropped: Optional[str]) -> Dict[str, Any]:
    return {"decision": "PD-like Pattern" if p >= 0.5 else "Healthy Control", "p": float(p),
            "mode": mode, "w_voice": wv, "w_tremor": wt, "dropped": dropped}


# ----------------------------------------------------------------------------- UI
def main() -> None:
    st.set_page_config(page_title="PD-SynergyBoost demo", page_icon="🧠", layout="wide")
    st.title("🧠 PD-SynergyBoost: late fusion demo")
    st.warning(DISCLOSURE)

    left, right = st.columns(2, gap="large")

    # ---------------- left: voice
    p_voice: Optional[float] = None
    with left:
        st.subheader("Voice (sustained /a/, .wav)")
        up = st.file_uploader("Upload a WAV recording (8 or 16 kHz telephone audio works)",
                              type=["wav"])
        if up is None:
            st.info("No audio yet: voice will be treated as unavailable in the fusion.")
        else:
            audio_bytes = up.getvalue()
            st.audio(audio_bytes, format="audio/wav")
            res = analyze_audio(audio_bytes)
            if "error" in res:
                st.error(res["error"] + " Voice is treated as unavailable.")
            else:
                feats = res["features"]
                p_voice = voice_probability(feats["jitter_local"])
                c1, c2 = st.columns(2)
                c1.metric("Jitter (local, Praat)", f"{100 * feats['jitter_local']:.2f} %")
                c2.metric("Rule-based voice probability",
                          "n/a" if p_voice is None else f"{p_voice:.2f}")
                with st.expander("Extracted CSFS features", expanded=False):
                    df = pd.DataFrame({"feature": CSFS_FEATURES,
                                       "value": [feats[k] for k in CSFS_FEATURES]})
                    df["value"] = df["value"].map(lambda v: "n/a" if not np.isfinite(v) else f"{v:.5g}")
                    st.dataframe(df, hide_index=True, use_container_width=True)
                    st.caption(
                        f"{res['duration_s']:.1f} s analysed at {res['sr']} Hz. All values measured with "
                        "Praat (jitter_ddp = 3 x jitter_rap by definition; Flo = lowest voiced F0; "
                        "D2 needs the optional `nolds` package). Note: `jitter_local` here is "
                        "Praat's cycle-level jitter, which is not the same quantity (or scale) as the "
                        "frame-level jitter the research model was trained on.")

    # ---------------- right: tremor / health sliders + fusion
    with right:
        st.subheader("Tremor and sensor health (simulated inputs)")
        p_tremor = st.slider("Tremor probability", 0.0, 1.0, 0.60, 0.01)
        h_voice = st.slider("Voice health", 0.0, 1.0, 0.80, 0.01)
        h_tremor = st.slider("Tremor health", 0.0, 1.0, 0.80, 0.01)
        use_mhs = st.toggle("Apply MHS health rule (live)", value=True,
                            help="On: modalities with health < 0.3 are dropped, otherwise a "
                                 "health-weighted average. Off: plain average, health ignored.")

        out = fuse(p_voice, p_tremor, h_voice, h_tremor, use_mhs)
        st.divider()
        if out["decision"] is None:
            st.error("⚠️ Cannot decide: both modalities are unhealthy or unavailable.")
        else:
            (st.error if out["decision"] == "PD-like Pattern" else st.success)(
                f"### {out['decision']}")
            m1, m2 = st.columns(2)
            m1.metric("Fused probability", f"{out['p']:.2f}")
            m2.metric("Confidence", f"{max(out['p'], 1 - out['p']):.2f}")
        st.caption(
            f"Rule applied: **{out['mode']}**"
            + (f" | weights voice/tremor = {out['w_voice']:.2f}/{out['w_tremor']:.2f}"
               if out["decision"] is not None else "")
            + (f" | dropped: {out['dropped']}" if out["dropped"] else ""))
        st.caption("Demo output only: a screening-style illustration, not a medical result.")


if __name__ == "__main__":
    main()
