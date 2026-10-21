#!/usr/bin/env python3
"""
pd_common.py -- shared building blocks for PD-SynergyBoost (voice, tremor, fusion)
==================================================================================

WHY THIS FILE EXISTS: voice_model.pkl / tremor_model.pkl contain instances of
`StackedPD` and `ModalityHealthScorer`. A pickle stores the CLASS'S MODULE PATH, so if those
classes were defined inside a script run with `%run` they would be saved as `__main__.X` and
fusion_demo.py could never unpickle them. Keeping them here (a real module that lives in
BASE_DIR) fixes that, and gives voice and tremor the *same* CSFS / MHS / stacking code.

Put this file next to the other scripts in /content/drive/MyDrive/PD_SynergyBoost/.

Contents
    Checkpoint              config-aware, atomic pickle checkpoint (resume-safe)
    kl_entropy / CSFS       kNN (Kozachenko-Leonenko) entropy, MI, CMI, CSFS selection
    ModalityHealthScorer    Mahalanobis-based sensor/modality health in [0, 1]
    StackedPD               impute -> CSFS -> 3 base learners -> elastic-net meta-learner
    run_cv                  nested-safe CV runner (imputation + CSFS + stacking INSIDE folds)
    metrics / plots / markdown helpers

Library versions (pin after first successful run via `!pip freeze | grep -iE
"scikit-learn|lightgbm|xgboost|numpy|scipy|pandas|matplotlib"` and paste here):
    # scikit-learn==<fill>  lightgbm==<fill>  xgboost==<fill>  numpy==<fill>
    # scipy==<fill>  pandas==<fill>  matplotlib==<fill>
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # mount Drive (no-op outside Colab / when already mounted)
    from google.colab import drive  # type: ignore
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
except ImportError:
    pass

BASE_DIR = os.environ.get("PD_BASE_DIR", "/content/drive/MyDrive/PD_SynergyBoost/")

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import digamma
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

import lightgbm as lgb
import xgboost as xgb

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_LEARNER_NAMES = ["lgbm", "xgb", "rf"]
SNR_LEVELS_DB = (20, 10, 5, 0)
HEALTH_THRESHOLD = 0.3


# =========================================================================================
# Small utilities
# =========================================================================================
def atomic_write_text(path: Path, text: str) -> None:
    """Write text via temp file + rename (a disconnect cannot leave a half-written file)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_pickle(obj: Any, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def config_hash(cfg: Dict[str, Any]) -> str:
    """Stable short hash of a JSON-able config (used to detect stale checkpoints/models)."""
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


class Checkpoint:
    """Key->value store persisted as one pickle on Drive. If the stored config differs from the
    current one the checkpoint is discarded (stale results are never silently reused)."""

    def __init__(self, path: Path, config: Dict[str, Any], force: bool = False) -> None:
        self.path, self.config, self.data = Path(path), config, {}
        if force and self.path.exists():
            self.path.unlink()
            print(f"[ckpt] --force: removed {self.path.name}")
        if self.path.exists():
            try:
                with open(self.path, "rb") as f:
                    obj = pickle.load(f)
                if obj.get("config") == config:
                    self.data = obj["data"]
                    print(f"[ckpt] resumed {self.path.name}: {len(self.data)} saved items")
                else:
                    print(f"[ckpt] config changed -> discarding {self.path.name}")
            except Exception as e:  # corrupt checkpoint -> start clean
                print(f"[ckpt] could not read {self.path.name} ({e}); starting clean")

    def has(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str) -> Any:
        return self.data[key]

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        atomic_pickle({"config": self.config, "data": self.data}, self.path)


def df_to_md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Tiny DataFrame -> GitHub markdown table (avoids the `tabulate` dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                 else ("" if (isinstance(v, float) and np.isnan(v)) else str(v)) for v in r.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# =========================================================================================
# CSFS -- Complementary Synergy Feature Selection
#   score(f) = MI(f;y) + lambda * mean_j CMI(f;y | f_j)
#   CMI(f_i;y|f_j) = H(f_i|f_j) - H(f_i|y,f_j), entropies by Kozachenko-Leonenko kNN.
# =========================================================================================
def kl_entropy(X: np.ndarray, k: int = 3) -> float:
    """Kozachenko-Leonenko differential entropy (nats), max-norm:
    H = psi(N) - psi(k) + d * mean(log(2 * eps_i)), eps_i = distance to k-th neighbour."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n, d = X.shape
    if n <= k + 1:
        return float("nan")
    dist, _ = cKDTree(X).query(X, k=k + 1, p=np.inf)
    eps = np.maximum(dist[:, k], 1e-10)
    return float(digamma(n) - digamma(k) + d * np.mean(np.log(2.0 * eps)))


def csfs_scores(X: np.ndarray, y: np.ndarray, lam: float = 0.5, k_knn: int = 3,
                n_candidates: int = 30, max_samples: int = 1500, seed: int = 0
                ) -> Dict[str, np.ndarray]:
    """MI, mean CMI and CSFS score for every feature (X must have no NaN).

    * Features are z-scored and given 1e-6 jitter (kNN estimators hate exact ties).
    * Large n (tremor windows) is stratified-subsampled to `max_samples` for speed.
    * To keep the O(p^2) CMI step cheap, only the `n_candidates` features with the highest
      MI are scored (conditioning set = the same candidates). Others get score = NaN and can
      never be selected. For voice (~30 features) every feature is a candidate.
    * Raw CMI estimates can be slightly negative (estimator noise); they are NOT clipped.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    y = np.asarray(y).astype(int)
    n, p = X.shape
    if n > max_samples:
        parts = []
        for c in np.unique(y):
            pool = np.flatnonzero(y == c)
            size = min(len(pool), max(5, int(round(max_samples * len(pool) / n))))
            parts.append(rng.choice(pool, size=size, replace=False))
        idx = np.sort(np.concatenate(parts))
        X, y = X[idx], y[idx]
        n = len(y)
    sd = X.std(axis=0)
    Z = (X - X.mean(axis=0)) / np.where(sd < 1e-12, 1.0, sd)
    Z = Z + rng.normal(0.0, 1e-6, size=Z.shape)

    mi = mutual_info_classif(Z, y, n_neighbors=3, random_state=seed)
    cand = np.argsort(-mi)[: min(n_candidates, p)]
    classes = np.unique(y)
    pc = {c: float((y == c).mean()) for c in classes}
    ic = {c: np.flatnonzero(y == c) for c in classes}

    h_all = {j: kl_entropy(Z[:, j]) for j in cand}
    h_cls = {(c, j): kl_entropy(Z[ic[c], j]) for c in classes for j in cand}
    cmi_sum = {i: 0.0 for i in cand}
    cmi_cnt = {i: 0 for i in cand}
    for a in range(len(cand)):
        for b in range(a + 1, len(cand)):
            i, j = cand[a], cand[b]
            hij = kl_entropy(Z[:, [i, j]])
            hij_c = {c: kl_entropy(Z[ic[c]][:, [i, j]]) for c in classes}
            for fi, fj in ((i, j), (j, i)):
                h_i_given_j = hij - h_all[fj]
                h_i_given_yj = sum(pc[c] * (hij_c[c] - h_cls[(c, fj)]) for c in classes)
                cmi = h_i_given_j - h_i_given_yj
                if np.isfinite(cmi):
                    cmi_sum[fi] += cmi
                    cmi_cnt[fi] += 1
    mean_cmi = np.full(p, np.nan)
    score = np.full(p, np.nan)
    for i in cand:
        mean_cmi[i] = cmi_sum[i] / cmi_cnt[i] if cmi_cnt[i] else 0.0
        score[i] = mi[i] + lam * mean_cmi[i]
    return {"mi": mi, "mean_cmi": mean_cmi, "score": score}


def csfs_select(X: np.ndarray, y: np.ndarray, k: int = 6, lam: float = 0.5, **kw: Any
                ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Top-k feature indices by CSFS score (+ the full score table arrays)."""
    t = csfs_scores(X, y, lam=lam, **kw)
    order = np.argsort(-np.nan_to_num(t["score"], nan=-np.inf))
    sel = np.sort(order[:k])
    return sel, t


def csfs_table(names: Sequence[str], t: Dict[str, np.ndarray], sel: np.ndarray) -> pd.DataFrame:
    """DataFrame: feature | MI | mean CMI | final score | selected?  (sorted by score)."""
    df = pd.DataFrame({"feature": list(names), "MI": t["mi"], "mean_CMI": t["mean_cmi"],
                       "score": t["score"]})
    df["selected"] = ["yes" if i in set(sel.tolist()) else "" for i in range(len(df))]
    return df.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)


# =========================================================================================
# MHS -- ModalityHealthScorer
# =========================================================================================
class ModalityHealthScorer:
    """Health of a modality's feature vector relative to the training distribution.

    fit(X_train): mean and regularised covariance (cov + eps*I).
    score(x):     Mahalanobis distance d; health = exp(-0.5 * d^2)  in [0, 1].
    Rule:         health < threshold (0.3) -> modality "unhealthy" (drop in fusion).

    CALIBRATION NOTE (dim_normalize=True, default): for a d-dimensional Gaussian the expected
    d^2 of a *perfectly normal* sample is d (here 6), so the literal exp(-0.5 d^2) gives
    exp(-3) ~ 0.05 < 0.3 for clean data -- every sample would be flagged. With
    dim_normalize=True we use d^2 / dim (expected value 1 for clean data -> health ~ 0.6; in
    the ideal Gaussian case only ~2-3 % of clean samples fall below 0.3). On real, small,
    non-Gaussian data the false-flag rate on held-out clean samples is usually HIGHER (a few
    to ~10 %), because the covariance is estimated from few samples. The pipelines report the
    measured clean-sample false-flag rate -- quote that number, not the ideal one. Pass
    dim_normalize=False for the literal formula (flags nearly everything).
    """

    def __init__(self, eps: float = 1e-3, threshold: float = HEALTH_THRESHOLD,
                 dim_normalize: bool = True) -> None:
        self.eps, self.threshold, self.dim_normalize = eps, threshold, dim_normalize
        self.mean_: Optional[np.ndarray] = None
        self.prec_: Optional[np.ndarray] = None
        self.dim_: int = 0

    def fit(self, X: np.ndarray) -> "ModalityHealthScorer":
        X = np.asarray(X, float)
        self.dim_ = X.shape[1]
        self.mean_ = X.mean(axis=0)
        cov = np.atleast_2d(np.cov(X, rowvar=False)) + self.eps * np.eye(self.dim_)
        self.prec_ = np.linalg.pinv(cov)
        return self

    def mahalanobis_sq(self, X: np.ndarray) -> np.ndarray:
        D = np.atleast_2d(np.asarray(X, float)) - self.mean_
        return np.einsum("ij,jk,ik->i", D, self.prec_, D)

    def score(self, x: np.ndarray) -> np.ndarray:
        """Health in [0,1] for one vector or a matrix of rows."""
        d2 = self.mahalanobis_sq(x)
        if self.dim_normalize:
            d2 = d2 / self.dim_
        return np.exp(-0.5 * d2)

    def is_healthy(self, health: Any) -> Any:
        return np.asarray(health) >= self.threshold


def add_noise_snr(X: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise to STANDARDISED features so that signal power (unit variance
    per feature) / noise power = 10^(snr_db/10)."""
    sigma = np.sqrt(1.0 / (10.0 ** (snr_db / 10.0)))
    return X + rng.normal(0.0, sigma, size=X.shape)


# =========================================================================================
# StackedPD -- impute -> CSFS -> base learners -> elastic-net meta-learner
# =========================================================================================
def make_base_learners(balanced: bool, pos_weight: float, seed: int, small_data: bool,
                       n_jobs: int = 2) -> Dict[str, Any]:
    """LightGBM / XGBoost / RandomForest with conservative settings for small data."""
    return {
        "lgbm": lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=7,
            min_child_samples=5 if small_data else 20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, class_weight="balanced" if balanced else None,
            random_state=seed, n_jobs=n_jobs, verbose=-1),
        "xgb": xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss", tree_method="hist",
            scale_pos_weight=pos_weight if balanced else 1.0, random_state=seed,
            n_jobs=n_jobs, verbosity=0),
        "rf": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2 if small_data else 5, max_features="sqrt",
            class_weight="balanced" if balanced else None, random_state=seed, n_jobs=n_jobs),
    }


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1.0 - p))


def make_meta(balanced: bool, seed: int) -> LogisticRegression:
    """Elastic-Net logistic regression (l1_ratio=0.5, solver='saga').
    sklearn >= 1.8 deprecates `penalty=`; fall back transparently if it is ever removed."""
    kw = dict(l1_ratio=0.5, solver="saga", C=1.0, max_iter=5000,
              class_weight="balanced" if balanced else None, random_state=seed)
    try:
        return LogisticRegression(penalty="elasticnet", **kw)
    except TypeError:
        return LogisticRegression(**kw)


class StackedPD:
    """One modality's full model. Everything data-dependent (imputation, CSFS, scaling, base
    learners, meta-learner) is fitted ONLY inside fit(), so calling fit() on a CV training
    fold is leak-free.

    Proper stacking: base learners' OUT-OF-FOLD probabilities on the training set (inner CV,
    stratified, or grouped when `groups` are given) are the meta-learner's features
    (clipped logit of the probability). Base learners are then refit on all training data.
    """

    def __init__(self, k: int = 6, lam: float = 0.5, use_csfs: bool = True,
                 balanced: bool = False, n_inner: int = 5, seed: int = 42,
                 small_data: bool = True, csfs_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.k, self.lam, self.use_csfs, self.balanced = k, lam, use_csfs, balanced
        self.n_inner, self.seed, self.small_data = n_inner, seed, small_data
        self.csfs_kwargs = csfs_kwargs or {}

    # ---- helpers -----------------------------------------------------------------------
    @staticmethod
    def _clean(X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float, copy=True)
        X[~np.isfinite(X)] = np.nan
        return X

    def _prep(self, X: np.ndarray) -> np.ndarray:
        return self.imputer_.transform(self._clean(X))[:, self.sel_]

    def _inner_splits(self, y: np.ndarray, groups: Optional[np.ndarray]):
        if groups is not None:
            n = min(self.n_inner, len(np.unique(groups)))
            return list(GroupKFold(n_splits=max(2, n)).split(np.zeros(len(y)), y, groups))
        n = max(2, min(self.n_inner, int(np.bincount(y).min())))
        return list(StratifiedKFold(n_splits=n, shuffle=True, random_state=self.seed)
                    .split(np.zeros(len(y)), y))

    # ---- fit / predict -----------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None
            ) -> "StackedPD":
        y = np.asarray(y).astype(int)
        Xc = self._clean(X)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        Xi = self.imputer_.fit_transform(Xc)
        if self.use_csfs:
            self.sel_, self.csfs_t_ = csfs_select(Xi, y, k=self.k, lam=self.lam,
                                                  seed=self.seed, **self.csfs_kwargs)
        else:
            self.sel_, self.csfs_t_ = np.arange(Xi.shape[1]), None
        Xs = Xi[:, self.sel_]
        self.scaler_ = StandardScaler().fit(Xs)          # used for MHS only
        pos_w = float((y == 0).sum()) / max(1, int((y == 1).sum()))
        self.base_ = make_base_learners(self.balanced, pos_w, self.seed, self.small_data)

        oof = np.full((len(y), len(self.base_)), float(y.mean()))
        for tr, va in self._inner_splits(y, groups):
            if len(np.unique(y[tr])) < 2:
                continue                                 # degenerate inner fold: keep prior
            for j, (name, est) in enumerate(self.base_.items()):
                m = clone(est).fit(Xs[tr], y[tr])
                oof[va, j] = m.predict_proba(Xs[va])[:, 1]
        self.meta_ = make_meta(self.balanced, self.seed).fit(_logit(oof), y)
        self.fitted_ = {n: clone(est).fit(Xs, y) for n, est in self.base_.items()}
        return self

    def predict_components(self, X: np.ndarray) -> Dict[str, Any]:
        """Base-learner probs, simple average, and stacked probability (all P(class 1))."""
        Xs = self._prep(X)
        base = {n: m.predict_proba(Xs)[:, 1] for n, m in self.fitted_.items()}
        P = np.column_stack([base[n] for n in BASE_LEARNER_NAMES])
        return {"base": base, "avg": P.mean(axis=1),
                "stack": self.meta_.predict_proba(_logit(P))[:, 1]}

    def predict_pos(self, X: np.ndarray) -> np.ndarray:
        """Stacked P(class 1) -- the number fusion_demo.py consumes."""
        return self.predict_components(X)["stack"]

    def transform_selected(self, X: np.ndarray) -> np.ndarray:
        """Imputed, CSFS-selected, standardised features (the space MHS lives in)."""
        return self.scaler_.transform(self._prep(X))


# =========================================================================================
# Metrics
# =========================================================================================
def fold_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    """AUC, sensitivity, specificity, F1 + confusion counts. NaN where undefined (one class)."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "sens": tp / (tp + fn) if (tp + fn) else float("nan"),
        "spec": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)) if (tp + fn) else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def mean_std(vals: Sequence[float]) -> Tuple[float, float, int]:
    a = np.asarray(vals, float)
    ok = a[np.isfinite(a)]
    return (float(ok.mean()) if ok.size else float("nan"),
            float(ok.std(ddof=0)) if ok.size else float("nan"), int(ok.size))


# =========================================================================================
# CV runner (imputation + CSFS + stacking all INSIDE each training fold)
# =========================================================================================
def run_cv(X: np.ndarray, y: np.ndarray, splits: List[Tuple[np.ndarray, np.ndarray]],
           model_kwargs: Dict[str, Any], ck: Checkpoint, tag: str,
           groups: Optional[np.ndarray] = None, compare_nocsfs: bool = True,
           do_mhs: bool = True, mhs_frac: float = 0.2, mhs_repeats: int = 20,
           seed: int = 42) -> List[Dict[str, Any]]:
    """Run every outer fold; each finished fold is checkpointed under `{tag}:fold{i}`.

    Per fold we store: test indices, y_test, base-learner / average / stacked probabilities,
    stacked-without-CSFS probabilities (optional), the selected feature indices, and (optional)
    MHS health of clean test samples and of a noisy 20 % subset at each SNR.
    """
    X, y = np.asarray(X, float), np.asarray(y).astype(int)
    results: List[Dict[str, Any]] = []
    for i, (tr, te) in enumerate(tqdm(splits, desc=f"CV {tag}")):
        key = f"{tag}:fold{i}"
        if ck.has(key):
            results.append(ck.get(key))
            continue
        t0 = time.time()
        g_tr = groups[tr] if groups is not None else None
        m = StackedPD(use_csfs=True, **model_kwargs).fit(X[tr], y[tr], g_tr)
        comp = m.predict_components(X[te])
        res: Dict[str, Any] = {
            "fold": i, "test_idx": te, "y_test": y[te], "p_base": comp["base"],
            "p_avg": comp["avg"], "p_stack": comp["stack"], "sel": m.sel_,
            "n_train": len(tr), "n_test": len(te)}
        if compare_nocsfs:
            m0 = StackedPD(use_csfs=False, **model_kwargs).fit(X[tr], y[tr], g_tr)
            res["p_stack_nocsfs"] = m0.predict_components(X[te])["stack"]
        if do_mhs:
            rng = np.random.default_rng(seed + i)
            mhs = ModalityHealthScorer().fit(m.transform_selected(X[tr]))
            Zte = m.transform_selected(X[te])
            res["health_clean"] = mhs.score(Zte)
            n_sub = max(3, int(round(mhs_frac * len(te)))) if len(te) >= 3 else len(te)
            sub = rng.choice(len(te), size=min(n_sub, len(te)), replace=False)
            res["health_noisy"] = {
                snr: np.concatenate([mhs.score(add_noise_snr(Zte[sub], snr, rng))
                                     for _ in range(mhs_repeats)])
                for snr in SNR_LEVELS_DB}
        ck.set(key, res)
        results.append(res)
        print(f"  [{tag}] fold {i + 1}/{len(splits)} done in {time.time() - t0:.0f}s "
              f"(train={len(tr)}, test={len(te)}, selected={m.sel_.tolist()})")
    return results


def method_probs(res: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """All compared methods' probabilities for one fold result."""
    out = {f"single:{n}": p for n, p in res["p_base"].items()}
    out["simple_average"] = res["p_avg"]
    out["stack_CSFS"] = res["p_stack"]
    if "p_stack_nocsfs" in res:
        out["stack_noCSFS"] = res["p_stack_nocsfs"]
    return out


def summarize_cv(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """mean +/- std across folds of AUC / sens / spec / F1 for every method.
    Adds 'valid_folds' (folds where the metric was defined)."""
    rows = []
    names = list(method_probs(results[0]).keys())
    for name in names:
        per = [fold_metrics(r["y_test"], method_probs(r)[name]) for r in results]
        row: Dict[str, Any] = {"method": name}
        for m in ("auc", "sens", "spec", "f1"):
            mu, sd, n = mean_std([d[m] for d in per])
            row[m] = mu
            row[f"{m}_sd"] = sd
            if m == "auc":
                row["valid_folds"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_pm(df: pd.DataFrame) -> pd.DataFrame:
    """Pretty 'mean ± sd' table from summarize_cv output."""
    out = pd.DataFrame({"method": df["method"]})
    for m, lab in (("auc", "AUC"), ("sens", "Sensitivity"), ("spec", "Specificity"), ("f1", "F1")):
        out[lab] = [f"{a:.3f} ± {s:.3f}" if np.isfinite(a) else "n/a"
                    for a, s in zip(df[m], df[f"{m}_sd"])]
    out["valid_folds"] = df["valid_folds"].astype(int)
    return out


def best_single(summary: pd.DataFrame) -> str:
    s = summary[summary["method"].str.startswith("single:")]
    return str(s.loc[s["auc"].idxmax(), "method"])


def pooled(results: List[Dict[str, Any]], method: str) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate out-of-fold labels/probabilities of one method across folds."""
    y = np.concatenate([r["y_test"] for r in results])
    p = np.concatenate([method_probs(r)[method] for r in results])
    return y, p


def summed_confusion(results: List[Dict[str, Any]], method: str = "stack_CSFS") -> np.ndarray:
    cm = np.zeros((2, 2), int)
    for r in results:
        cm += confusion_matrix(r["y_test"], (method_probs(r)[method] >= 0.5).astype(int),
                               labels=[0, 1])
    return cm


def fold_selection_counts(results: List[Dict[str, Any]], names: Sequence[str]) -> Dict[str, int]:
    """How many outer folds selected each feature (stability of CSFS)."""
    cnt: Dict[str, int] = {}
    for r in results:
        for i in r["sel"]:
            cnt[names[i]] = cnt.get(names[i], 0) + 1
    return dict(sorted(cnt.items(), key=lambda kv: -kv[1]))


# =========================================================================================
# MHS summary + plots
# =========================================================================================
def mhs_summary(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Health vs SNR across folds: mean/median/std and fraction flagged unhealthy (<0.3)."""
    clean = np.concatenate([r["health_clean"] for r in results])
    rows = [{"condition": "clean test samples", "n": len(clean), "mean_health": clean.mean(),
             "median_health": float(np.median(clean)), "std": clean.std(),
             "frac_below_0.3": float((clean < HEALTH_THRESHOLD).mean())}]
    for snr in SNR_LEVELS_DB:
        h = np.concatenate([r["health_noisy"][snr] for r in results])
        rows.append({"condition": f"noise SNR {snr} dB", "n": len(h), "mean_health": h.mean(),
                     "median_health": float(np.median(h)), "std": h.std(),
                     "frac_below_0.3": float((h < HEALTH_THRESHOLD).mean())})
    return pd.DataFrame(rows)


def plot_mhs(summary: pd.DataFrame, path: Path, title: str) -> None:
    labels = ["clean"] + [f"{s} dB" for s in SNR_LEVELS_DB]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(range(len(labels)), summary["mean_health"], yerr=summary["std"], marker="o",
                capsize=4, label="mean ± sd")
    ax.plot(range(len(labels)), summary["median_health"], "s--", alpha=0.7, label="median")
    ax.axhline(HEALTH_THRESHOLD, color="red", ls=":", label="unhealthy threshold (0.3)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Added-noise SNR (feature space)")
    ax.set_ylabel("Modality health score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    for name, (y, p) in curves.items():
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.2f})")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, path: Path, title: str, classes: Sequence[str]) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
