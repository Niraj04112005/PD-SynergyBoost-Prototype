"""
PD-SynergyBoost - Streamlit demo (Frontend)
"""
from __future__ import annotations

import html as _html
import io
import json
import os
from typing import Any, Dict, Optional

import librosa
import numpy as np
import pandas as pd
import parselmouth
import requests
import streamlit as st
from parselmouth.praat import call

# ============ API CONFIGURATION ============
API_URL = "http://127.0.0.1:8000"
API_KEY = "change-me-pd-demo"
API_TIMEOUT = (4, 20)
# ===========================================

# =============================================================================================
# 1. BACKEND LOGIC (UNCHANGED)
# =============================================================================================
F_MIN, F_MAX = 60.0, 500.0
MAX_SECONDS = 8.0
MIN_SR = 6000
F0_REF = 127.14
HEALTH_THRESHOLD = 0.3
CSFS_FEATURES = ["jitter_local", "Flo", "D2",
                 "jitter_ddp", "jitter_rap", "pr_shimmer_apq11"]


class AudioError(ValueError):
    pass


def _finite(v: Any) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


@st.cache_data(show_spinner="Analysing voice ...")
def analyze_audio(audio_bytes: bytes) -> Dict[str, Any]:
    try:
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        except Exception as e:
            raise AudioError(
                f"Cannot read this file as WAV audio ({type(e).__name__}).") from e
        if y.size == 0:
            raise AudioError("The audio file is empty.")
        if sr < MIN_SR:
            raise AudioError(
                f"Sampling rate {sr} Hz is too low (need 8 kHz or higher).")
        y = y - np.mean(y)
        y, _ = librosa.effects.trim(y, top_db=30)
        if y.size / sr < 0.5:
            raise AudioError(
                "Recording is shorter than 0.5 s after removing silence.")
        peak = float(np.max(np.abs(y)))
        if peak < 1e-6:
            raise AudioError("Recording is silent.")
        y = (0.9 * y / peak)[: int(MAX_SECONDS * sr)].astype(np.float64)

        snd = parselmouth.Sound(y, sampling_frequency=float(sr))
        feats = {k: float("nan") for k in CSFS_FEATURES}
        pp = call(snd, "To PointProcess (periodic, cc)", F_MIN, F_MAX)
        if call(pp, "Get number of points") < 5:
            raise AudioError(
                "Too few voiced pulses found: use a steady sustained vowel /a/.")
        a = (0, 0, 0.0001, 0.02, 1.3)
        feats["jitter_local"] = _finite(call(pp, "Get jitter (local)", *a))
        feats["jitter_rap"] = _finite(call(pp, "Get jitter (rap)", *a))
        feats["jitter_ddp"] = _finite(call(pp, "Get jitter (ddp)", *a))
        feats["pr_shimmer_apq11"] = _finite(
            call([snd, pp], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6))

        pitch = snd.to_pitch_ac(pitch_floor=F_MIN, pitch_ceiling=F_MAX)
        f0 = pitch.selected_array["frequency"]
        f0 = f0[f0 > 0]
        if f0.size:
            feats["Flo"] = float(np.min(f0))

        try:
            import nolds
            if f0.size >= 64:
                st_series = 12.0 * np.log2(f0 / F0_REF)
                if np.std(st_series) > 1e-9:
                    feats["D2"] = _finite(nolds.corr_dim(
                        (st_series - st_series.mean()) / st_series.std(), emb_dim=4, fit="poly"))
        except Exception:
            pass

        if not np.isfinite(feats["jitter_local"]):
            raise AudioError("Praat could not measure jitter.")
        return {"features": feats, "duration_s": float(y.size / sr), "sr": int(sr)}
    except AudioError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Analysis failed: {type(e).__name__}: {e}"}


def query_voice_api(feats: Dict[str, float]) -> Dict[str, Any]:
    if "PASTE-YOUR" in API_URL or not API_URL.startswith("http"):
        return {"error": "API_URL is not set.", "kind": "config"}
    payload = {k: (float(feats[k]) if np.isfinite(
        feats[k]) else None) for k in CSFS_FEATURES}
    key = (API_URL, json.dumps(payload, sort_keys=True))
    cache = st.session_state.setdefault("_api_cache", {})
    if key in cache:
        return cache[key]
    try:
        r = requests.post(API_URL.rstrip("/") + "/predict", json=payload, timeout=API_TIMEOUT,
                          headers={"X-API-Key": API_KEY, "ngrok-skip-browser-warning": "1"})
    except requests.exceptions.Timeout:
        return {"error": "The model API timed out.", "kind": "offline"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Cannot reach the model API ({type(e).__name__}).", "kind": "offline"}
    if r.status_code == 401:
        return {"error": "API rejected key.", "kind": "auth"}
    if r.status_code != 200:
        return {"error": f"API returned HTTP {r.status_code}", "kind": "http"}
    try:
        body = r.json()
        p = float(body["p_pd"])
    except (ValueError, KeyError, TypeError):
        return {"error": "Unexpected reply from the API.", "kind": "bad"}
    out = {"p": p, "imputed": list(body.get("imputed_features", []))}
    cache[key] = out
    return out


def fuse(p_voice: Optional[float], p_tremor: float, h_voice: float, h_tremor: float, use_mhs: bool = True) -> Dict[str, Any]:
    v_ok = p_voice is not None and (h_voice >= HEALTH_THRESHOLD or not use_mhs)
    t_ok = h_tremor >= HEALTH_THRESHOLD or not use_mhs
    if not use_mhs:
        vals = [p for p in (p_voice, p_tremor) if p is not None]
        p = float(np.mean(vals))
        return _result(p, "No MHS: plain average", 0.5, 0.5, None)
    if v_ok and t_ok:
        s = h_voice + h_tremor
        wv, wt = (h_voice / s, h_tremor / s) if s > 0 else (0.5, 0.5)
        return _result(wv * p_voice + wt * p_tremor, "Both healthy: weighted average", wv, wt, None)
    if v_ok:
        return _result(p_voice, "Tremor unhealthy: dropped", 1.0, 0.0, "tremor")
    if t_ok:
        return _result(p_tremor, "Voice unhealthy: dropped", 0.0, 1.0, "voice")
    return {"decision": None, "p": None, "mode": "Cannot decide", "w_voice": 0.0, "w_tremor": 0.0, "dropped": "both"}


def _result(p: float, mode: str, wv: float, wt: float, dropped: Optional[str]) -> Dict[str, Any]:
    return {"decision": "PD-like Pattern" if p >= 0.5 else "Healthy Control", "p": float(p),
            "mode": mode, "w_voice": wv, "w_tremor": wt, "dropped": dropped}


# =============================================================================================
# 2. UI HELPERS & PREMIUM CSS
# =============================================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

:root{
  --bg:#0b1220; --panel:rgba(17,26,46,.85); --panel-2:rgba(255,255,255,.05);
  --line:rgba(148,163,184,.2); --text:#e6edf7; --muted:#93a4bd;
  --blue:#38bdf8; --teal:#2dd4bf; --red:#f87171; --green:#34d399;
}
.stApp{
  background: radial-gradient(1000px 500px at 10% 0%, rgba(56,189,248,.15), transparent 60%),
              radial-gradient(900px 500px at 90% 10%, rgba(45,212,191,.1), transparent 50%), var(--bg);
  color:var(--text); font-family:"Inter", sans-serif;
}
#MainMenu, footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2rem; padding-bottom:3rem; max-width:1200px;}

/* Fade in animation */
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

/* Cards */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--panel); border:1px solid var(--line)!important; border-radius:20px;
  padding:1.5rem; backdrop-filter:blur(10px); box-shadow:0 10px 30px rgba(0,0,0,0.4);
  animation: fadeIn 0.5s ease-out;
}
.card-title{display:flex; align-items:center; gap:.8rem; margin-bottom:1.2rem;
  font-weight:800; font-size:1.25rem; color:var(--text);}
.card-title .dot{width:40px; height:40px; border-radius:12px; display:grid; place-items:center;
  background:linear-gradient(135deg, rgba(56,189,248,.2), rgba(45,212,191,.2));
  border:1px solid rgba(56,189,248,.4); font-size:1.2rem;}

/* Header */
.hero{display:flex; align-items:center; gap:1.5rem; padding:1.8rem 2rem; margin-bottom:1.5rem;
  border-radius:24px; border:1px solid var(--line);
  background:linear-gradient(120deg, rgba(56,189,248,.15), rgba(45,212,191,.08) 50%, rgba(17,26,46,.8));
  box-shadow:0 15px 40px rgba(0,0,0,0.5);}
.hero h1{margin:0; font-size:2.8rem; line-height:1.1; font-weight:800; letter-spacing:-0.5px;
  background:linear-gradient(90deg,#e6f6ff,#7dd3fc 40%,#5eead4); -webkit-background-clip:text;
  background-clip:text; -webkit-text-fill-color:transparent; filter:drop-shadow(0 0 10px rgba(56,189,248,0.3));}
.hero p{margin:.4rem 0 0; color:var(--muted); font-size:1.1rem;}

/* Tiles */
.tiles{display:grid; grid-template-columns:repeat(2,1fr); gap:1rem; margin:1rem 0;}
.tile{padding:1.2rem; border-radius:16px; background:var(--panel-2); border:1px solid var(--line);}
.tile .k{font-size:.8rem; letter-spacing:1px; text-transform:uppercase; color:var(--muted);}
.tile .v{font-size:2rem; font-weight:800; margin-top:.3rem; color:var(--text);}

/* Feature Grid */
.fgrid{display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin:1rem 0;}
.fcard{padding:1rem; border-radius:16px; background:rgba(56,189,248,.08); border:1px solid rgba(56,189,248,.3);}
.fcard .n{font-size:.85rem; color:#7dd3fc; font-weight:600;}
.fcard .x{font-size:1.4rem; font-weight:800; margin:.3rem 0; color:var(--text);}

/* Result Banners */
.rgrid{display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-top:1rem;}
.rcard{padding:2rem; border-radius:20px; text-align:center; border:1.5px solid var(--line);}
.rcard.pd{border-color:var(--red); background:linear-gradient(180deg, rgba(248,113,113,0.15), rgba(11,18,32,0.8)); box-shadow:0 0 30px rgba(248,113,113,0.2);}
.rcard.hc{border-color:var(--green); background:linear-gradient(180deg, rgba(52,211,153,0.15), rgba(11,18,32,0.8)); box-shadow:0 0 30px rgba(52,211,153,0.2);}
.rcard.na{border-color:var(--amber);}
.rcard .k{font-size:.85rem; text-transform:uppercase; letter-spacing:1px; color:var(--muted);}
.rcard .big{font-size:3.5rem; font-weight:800; margin:.5rem 0;}
.rcard.pd .big{color:var(--red);}
.rcard.hc .big{color:var(--green);}

/* Sidebar */
section[data-testid="stSidebar"]{background:#070b14; border-right:1px solid var(--line);}
.sb-title{font-size:1.3rem; font-weight:800; color:var(--text); margin-bottom:1rem; display:flex; align-items:center; gap:.5rem;}
</style>
"""
BRAIN_SVG = """<svg viewBox="0 0 64 64" fill="none" stroke="url(#pdg)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><defs><linearGradient id="pdg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#38bdf8"/><stop offset="1" stop-color="#2dd4bf"/></linearGradient></defs><path d="M32 12c-3-4-10-4-13 1-5 0-9 4-8 9-3 3-3 8 0 11-1 5 3 10 8 10 2 5 9 6 13 2"/><path d="M32 12c3-4 10-4 13 1 5 0 9 4 8 9 3 3 3 8 0 11 1 5-3 10-8 10-2 5-9 6-13 2"/><path d="M32 12v40"/><path d="M22 24c3 0 5 2 5 5M42 24c-3 0-5 2-5 5M22 40c3 0 5-2 5-4M42 40c-3 0-5-2-5-4"/></svg>"""

DISCLOSURE_HTML = ("<b>Demo mode.</b> Voice probability comes from the trained stacking model via local API. "
                   "Validation is limited (age confound), and Praat features may differ in scale. "
                   "Tremor is simulated. <b>Not a diagnosis.</b>")
FEATURE_INFO = {
    "jitter_local": ("Jitter (local)", "Pitch variation"),
    "Flo": ("Flo (Hz)", "Lowest pitch"),
    "D2": ("D2", "Correlation dim."),
    "jitter_ddp": ("Jitter (DDP)", "3 x RAP"),
    "jitter_rap": ("Jitter (RAP)", "Relative avg. pert."),
    "pr_shimmer_apq11": ("Shimmer (APQ11)", "Amplitude variation"),
}


def md(s: str) -> None: st.markdown(s, unsafe_allow_html=True)
def inject_css() -> None: md(CSS)


def render_header() -> None:
    md(f'<div class="hero">{BRAIN_SVG}<div><h1>PD-SynergyBoost</h1><p>Health-aware late fusion of voice and wearable-tremor signals.</p></div></div>')


def notice(kind: str, text_html: str) -> None:
    icon = {"warn": "⚠️", "err": "❌", "info": "ℹ️"}.get(kind, "")
    md(f'<div class="notice {kind}" style="background:var(--panel-2); padding:1rem; border-radius:12px; margin-bottom:1rem; border-left:4px solid var(--{kind if kind != "info" else "blue"});"><b>{icon}</b> {text_html}</div>')


def card_title(icon: str, title: str) -> None:
    md(f'<div class="card-title"><span class="dot">{icon}</span>{_html.escape(title)}</div>')


def tile(label: str, value: str) -> str:
    return f'<div class="tile"><div class="k">{_html.escape(label)}</div><div class="v">{value}</div></div>'


def health_pill(h: float, use_mhs: bool) -> str:
    if not use_mhs:
        return '<span style="color:var(--muted)">ignored</span>'
    return '<span style="color:var(--green)">healthy</span>' if h >= HEALTH_THRESHOLD else '<span style="color:var(--red)">unhealthy</span>'


def feature_grid(feats: Dict[str, float]) -> str:
    cells = []
    for k in CSFS_FEATURES:
        name, desc = FEATURE_INFO[k]
        v = feats[k]
        val = "n/a" if not np.isfinite(v) else f"{v:.5g}"
        cells.append(
            f'<div class="fcard"><div class="n">{k}</div><div class="x">{val}</div><div style="font-size:0.75rem; color:var(--muted);">{_html.escape(name)}</div></div>')
    return '<div class="fgrid">' + "".join(cells) + "</div>"


def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        md('<div class="sb-title">🎛️ Sensor Simulation</div>')
        p_tremor = st.slider("📊 Tremor Probability", 0.0, 1.0, 0.60, 0.01)
        h_voice = st.slider("🎙️ Voice Health", 0.0, 1.0, 0.80, 0.01)
        h_tremor = st.slider("🖐️ Tremor Health", 0.0, 1.0, 0.80, 0.01)
        use_mhs = st.toggle("Apply MHS Health Rule", value=True,
                            help="Drops modalities below 0.3 health")
    return {"p_tremor": p_tremor, "h_voice": h_voice, "h_tremor": h_tremor, "use_mhs": use_mhs}


def result_cards(out: Dict[str, Any]) -> str:
    if out["decision"] is None:
        cls, prob, dec = "na", "—", "Cannot Decide"
    else:
        cls = "pd" if out["decision"] == "PD-like Pattern" else "hc"
        prob, dec = f"{out['p']:.2f}", _html.escape(out["decision"])
    return f'<div class="rgrid"><div class="rcard {cls}"><div class="k">Fused Probability</div><div class="big">{prob}</div></div><div class="rcard {cls}"><div class="k">Final Decision</div><div class="big" style="font-size:2.5rem; padding-top:1rem;">{dec}</div></div></div>'

# =============================================================================================
# 3. PAGE LAYOUT
# =============================================================================================


def main() -> None:
    st.set_page_config(page_title="PD-SynergyBoost",
                       page_icon="🧠", layout="wide")
    inject_css()
    ctrl = render_sidebar()
    render_header()
    notice("info", DISCLOSURE_HTML)

    col_v, col_t = st.columns(2, gap="large")
    p_voice: Optional[float] = None

    with col_v:
        with st.container(border=True):
            card_title("🎙️", "Voice Input")
            up = st.file_uploader("Upload WAV", type=[
                                  "wav"], label_visibility="collapsed")
            if up is None:
                notice("info", "Voice unavailable in fusion.")
            else:
                audio_bytes = up.getvalue()
                st.audio(audio_bytes, format="audio/wav")
                res = analyze_audio(audio_bytes)
                if "error" in res:
                    notice("err", res["error"])
                else:
                    feats = res["features"]
                    api = query_voice_api(feats)
                    if "error" in api:
                        notice("warn", "API Offline. Voice dropped.")
                    else:
                        p_voice = api["p"]

                    md('<div class="tiles">'
                       + tile("Jitter (Praat)",
                              f"{100 * feats['jitter_local']:.2f} %")
                       + tile("Voice Probability",
                              "n/a" if p_voice is None else f"{p_voice:.2f}")
                       + "</div>")
                    with st.expander("Extracted Features", expanded=True):
                        md(feature_grid(feats))
            md(f'<div style="margin-top:1rem; font-size:0.9rem;">Voice Health: {health_pill(ctrl["h_voice"], ctrl["use_mhs"])}</div>')

    with col_t:
        with st.container(border=True):
            card_title("🖐️", "Tremor Input")
            md('<div class="tiles">'
               + tile("Tremor Probability", f"{ctrl['p_tremor']:.2f}")
               + tile("Tremor Health", f"{ctrl['h_tremor']:.2f}")
               + "</div>")
            md(f'<div style="margin-top:1rem; font-size:0.9rem;">Tremor Health: {health_pill(ctrl["h_tremor"], ctrl["use_mhs"])}</div>')
            st.caption("Simulated wearable sensor output.")

    out = fuse(p_voice, ctrl["p_tremor"], ctrl["h_voice"],
               ctrl["h_tremor"], ctrl["use_mhs"])
    with st.container(border=True):
        card_title("🧬", "Fusion Results")
        md(result_cards(out))
        md(f'<div style="text-align:center; color:var(--muted); margin-top:1rem;">Rule: {out["mode"]}</div>')


if __name__ == "__main__":
    main()
