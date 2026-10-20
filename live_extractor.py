#!/usr/bin/env python3
"""
live_extractor.py -- PD-SynergyBoost: fast extraction of the 6 CSFS-selected VOICE features
==========================================================================================

    from live_extractor import extract_csfs_features
    df = extract_csfs_features("recording.wav")      # 1-row DataFrame, 6 columns, nothing printed

Columns (exact order): jitter_local, Flo, D2, jitter_ddp, jitter_rap, pr_shimmer_apq11

SCALE-MATCHING GUARANTEE: this module does NOT re-implement the feature maths. It imports the
constants and the helper functions (`perturbation_quotient`, `praat_perturbation`) from
feature_extraction_voice.py, the exact code that produced features_voice.csv, and repeats that
file's preprocessing step for step (DC removal, 30 dB trim, 0.9 peak normalisation, first 8 s,
pyin with the same frame/hop/resolution, F0 capped at 3400 Hz, F0 floor/ceiling 60-500 Hz).
Only RPDE/DFA/PPE/spread1/2 (and unused jitter/shimmer variants) are skipped for speed, so a live
call is much cheaper than a training-time extraction.

Needs feature_extraction_voice.py in the same folder (or BASE_DIR) and its dependencies
(librosa, praat-parselmouth, nolds, numpy, pandas, scipy, tqdm, openpyxl).

Errors: audio that cannot be used raises `AudioQualityError` (a ValueError subclass) with a
clear message: unreadable file, sampling rate < 6 kHz, shorter than 0.5 s after silence trim,
silent, or fewer than 20 voiced frames. Pass raise_on_error=False to get a row of NaN
instead (the model's imputer would then fill training medians, which is NOT a real
measurement; prefer handling the exception in your app).
A valid recording can still give NaN for D2 if the voiced F0 series is too short (< 64 frames)
or degenerate; this mirrors training, where such NaNs were median-imputed.

Version pins: see pd_common.py header (use the same librosa / parselmouth / nolds as training).
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict

BASE_DIR = os.environ.get("PD_BASE_DIR", "/content/drive/MyDrive/PD_SynergyBoost/")
for _p in (str(Path(__file__).resolve().parent) if "__file__" in globals() else ".", BASE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

import librosa
import nolds
import parselmouth

# Single source of truth for constants + feature maths (identical to training).
from feature_extraction_voice import (  # noqa: E402
    F0_REF, F_MAX, F_MIN, MIN_VOICED_FRAMES, RESOLUTION, TEL_FHI_CAP,
    perturbation_quotient, praat_perturbation)

warnings.filterwarnings("ignore")

CSFS_FEATURES = ["jitter_local", "Flo", "D2", "jitter_ddp", "jitter_rap", "pr_shimmer_apq11"]
MAX_SECONDS = 8.0      # same default as training (--max-seconds)
MIN_SR = 6000          # same rule as training


class AudioQualityError(ValueError):
    """The recording cannot be analysed (unreadable, too short, silent, unvoiced...)."""


def _load_and_prepare(wav_path: str) -> "tuple[np.ndarray, int]":
    """Training preprocessing, step for step. Raises AudioQualityError."""
    try:
        y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    except Exception as e:                                   # corrupt / not a wav / missing
        raise AudioQualityError(f"cannot read audio file: {type(e).__name__}: {e}") from e
    if y.size == 0:
        raise AudioQualityError("empty audio")
    if sr < MIN_SR:
        raise AudioQualityError(f"sampling rate too low ({sr} Hz; need >= {MIN_SR})")
    y = y - np.mean(y)
    y, _ = librosa.effects.trim(y, top_db=30)
    if y.size / sr < 0.5:
        raise AudioQualityError("recording shorter than 0.5 s after trimming silence")
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < 1e-6:
        raise AudioQualityError("silent recording")
    y = (0.9 * y / peak)[: int(MAX_SECONDS * sr)].astype(np.float64)
    return y, int(sr)


def extract_csfs_features(wav_path: str, raise_on_error: bool = True) -> pd.DataFrame:
    """Extract the 6 CSFS-selected features from one telephone-quality .wav (8 or 16 kHz).

    Returns a 1-row DataFrame with exactly the columns in CSFS_FEATURES, on the same scale as
    features_voice.csv. Prints nothing.
    """
    try:
        y, sr = _load_and_prepare(wav_path)

        # F0 via pyin (same window/hop/resolution as training)
        fl = int(2 ** np.ceil(np.log2(0.064 * sr)))
        hop = fl // 8
        f0, vflag, _ = librosa.pyin(y, fmin=F_MIN, fmax=F_MAX, sr=sr, frame_length=fl,
                                    hop_length=hop, resolution=RESOLUTION)
        voiced = np.asarray(vflag, bool) & np.isfinite(f0)
        if int(voiced.sum()) < MIN_VOICED_FRAMES:
            raise AudioQualityError(
                f"too few voiced frames ({int(voiced.sum())} < {MIN_VOICED_FRAMES}): no steady vowel found")
        f0 = np.where(voiced, np.minimum(f0, TEL_FHI_CAP), np.nan)   # telephone cap
        f0v = f0[voiced]

        # frame alignment exactly as in training (rms only used for its length here)
        rms = librosa.feature.rms(y=y, frame_length=fl, hop_length=hop, center=True)[0]
        n = min(len(f0), len(rms))
        f0, voiced = f0[:n], voiced[:n]
        period = np.where(voiced, 1.0 / np.where(voiced, f0, 1.0), np.nan)

        jit_rap = perturbation_quotient(period, voiced, 3)
        feats: Dict[str, float] = {
            "jitter_local": perturbation_quotient(period, voiced, 2),
            "jitter_rap": jit_rap,
            "jitter_ddp": 3.0 * jit_rap,
            "Flo": float(np.min(f0v)),
            "D2": float("nan"),
            "pr_shimmer_apq11": float("nan"),
        }

        # D2: correlation dimension of the z-scored semitone F0 series (as in training)
        st = 12.0 * np.log2(f0v / F0_REF)
        try:
            if st.size >= 64 and np.std(st) > 1e-9:
                feats["D2"] = float(nolds.corr_dim((st - st.mean()) / st.std(), emb_dim=4,
                                                   fit="poly"))
        except Exception:
            pass

        # Praat cycle-level shimmer (APQ11), same call as training
        snd = parselmouth.Sound(y, sampling_frequency=float(sr))
        feats["pr_shimmer_apq11"] = float(praat_perturbation(snd)["pr_shimmer_apq11"])

        return pd.DataFrame([feats], columns=CSFS_FEATURES).astype(float)
    except AudioQualityError:
        if raise_on_error:
            raise
    except Exception as e:                                   # unexpected failure in a library
        if raise_on_error:
            raise AudioQualityError(f"feature extraction failed: {type(e).__name__}: {e}") from e
    return pd.DataFrame([{c: np.nan for c in CSFS_FEATURES}], columns=CSFS_FEATURES)


def parity_check(wav_path: str) -> pd.DataFrame:
    """Developer check (slow: runs the FULL training extractor). Returns a DataFrame with the
    live value, the training-extractor value and their absolute difference for the 6 features.
    All differences should be ~0 (D2 can differ only if nolds is non-deterministic: it is not
    with fit='poly')."""
    from feature_extraction_voice import extract_features
    live = extract_csfs_features(wav_path).iloc[0]
    train = extract_features(str(wav_path))
    out = pd.DataFrame({"live": live, "training": pd.Series({c: train[c] for c in CSFS_FEATURES})})
    out["abs_diff"] = (out["live"] - out["training"]).abs()
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python live_extractor.py file.wav [--parity]")
    if "--parity" in sys.argv:
        print(parity_check(sys.argv[1]).to_string())
    else:
        print(extract_csfs_features(sys.argv[1]).to_string(index=False))
