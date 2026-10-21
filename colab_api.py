"""
PD-SynergyBoost - Colab inference API (FastAPI + pyngrok)         PART 1 / BACKEND
=================================================================================

Serves voice_model.pkl from Google Drive so the Streamlit app on your Windows machine never has
to unpickle it (no pd_common / NumPy / scikit-learn version problems on Windows).

HOW TO RUN (in Colab, from the SAME runtime kind you trained in):
  Cell 1:   !pip install -q fastapi "uvicorn[standard]" pyngrok
            (lightgbm / xgboost / scikit-learn are preinstalled in Colab; if the pickle was made with
             different versions and loading fails, see the error text below.)
  Cell 2:   %cd /content/drive/MyDrive/PD_SynergyBoost      (after mounting Drive)
            %run colab_api.py
  It prints API_URL and API_KEY: paste both at the top of app.py.

  * Get a free ngrok token at https://dashboard.ngrok.com/get-started/your-authtoken and either add it
    to Colab Secrets (key name NGROK_AUTH_TOKEN, "Notebook access" on) or paste it when prompted.
    Never commit the token to GitHub.
  * Keep the Colab session running while you use the app. The URL changes every time you re-run.
  * Use `%run` (not `!python`): the server lives in a background thread of the notebook kernel.

Needs in BASE_DIR: voice_model.pkl and pd_common.py (the module the pickle was created with).

ENDPOINTS
  GET  /health    -> {"status": "ok", ...}
  POST /predict   header  X-API-Key: <API_KEY>
                  body    {"jitter_local":..,"Flo":..,"D2":..,"jitter_ddp":..,"jitter_rap":..,
                           "pr_shimmer_apq11":..}   (null allowed for a missing value, e.g. D2)
                  returns {"p_pd": 0.63, "proba": [0.37, 0.63], "classes": ["HC","PwPD"],
                           "imputed_features": [...], "features_used": [...], ...}

NOTE ON THE MODEL INPUT: the deployed StackedPD was fitted on ALL training feature columns and picks
its CSFS features internally. The API therefore builds the full-width row (NaN for the columns it
does not need) from the 6 values it receives; only the 6 selected ones influence the prediction.
At start-up the script checks that the model's selected features are exactly among those 6 and
refuses to start otherwise (a silent mismatch would give meaningless probabilities).

SECURITY: the ngrok URL is public. API_KEY (shared secret, sent as X-API-Key) keeps casual visitors
out; it is not strong security. Do not send real patient data to a free public tunnel.
"""
import builtins
import getpass
import math
import os
import pickle
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from google.colab import drive  # type: ignore
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
except ImportError:
    pass

# ----------------------------------------------------------------------------- settings
BASE_DIR = os.environ.get("PD_BASE_DIR", "/content/drive/MyDrive/PD_SynergyBoost/")
MODEL_FILE = "voice_model.pkl"
PORT = 8000
API_KEY = "change-me-pd-demo"       # must be identical in app.py ("" disables the check)
NGROK_AUTH_TOKEN = ""               # leave empty: Colab Secret NGROK_AUTH_TOKEN, else you are prompted
FEATURES = ["jitter_local", "Flo", "D2", "jitter_ddp", "jitter_rap", "pr_shimmer_apq11"]
MIN_FEATURES = 4                    # at least this many of the 6 must be non-null

sys.path.insert(0, BASE_DIR)


# ----------------------------------------------------------------------------- model loading
def load_bundle(path: Path) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Unpickle voice_model.pkl and validate it against FEATURES.
    Returns (bundle, all_feature_names, selected_feature_names)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Is Drive mounted and BASE_DIR correct?")
    try:
        import pd_common  # noqa: F401  (the module the pickle's classes live in)
    except ModuleNotFoundError as e:
        raise RuntimeError(f"pd_common.py not found in {BASE_DIR}: copy it there. ({e})") from e
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
    except Exception as e:
        import sklearn
        raise RuntimeError(
            f"Could not unpickle {path.name}: {type(e).__name__}: {e}\n"
            f"Colab has scikit-learn {sklearn.__version__}, numpy {np.__version__}. If the model was trained "
            "with other versions, install the training versions (`!pip install scikit-learn==X ...`) and "
            "restart the session, or re-run voice_pipeline.py --force in this runtime.") from e
    for key in ("model", "feature_names"):
        if key not in bundle:
            raise RuntimeError(f"{path.name} has no '{key}' entry; re-run voice_pipeline.py.")
    names = list(bundle["feature_names"])
    selected = [names[i] for i in bundle["model"].sel_]
    outside = [s for s in selected if s not in FEATURES]
    if outside:
        raise RuntimeError(
            f"The model's CSFS-selected features are {selected}, but this API only accepts {FEATURES}. "
            f"Not accepted: {outside}. Edit FEATURES here, in app.py and in the request model.")
    print(f"[API] model loaded: {len(names)} training columns, selected = {selected}")
    return bundle, names, selected


def predict_from_features(bundle: Dict[str, Any], names: List[str], selected: List[str],
                          feats: Dict[str, Optional[float]]) -> Dict[str, Any]:
    """Pure function (no web code): 6 feature values -> probability. Raises ValueError on bad input."""
    clean: Dict[str, Optional[float]] = {}
    for k in FEATURES:
        v = feats.get(k)
        try:
            v = None if v is None else float(v)
        except (TypeError, ValueError):
            v = None
        clean[k] = v if v is not None and math.isfinite(v) else None
    n_ok = sum(v is not None for v in clean.values())
    if n_ok < MIN_FEATURES:
        raise ValueError(f"Only {n_ok} of {len(FEATURES)} features are usable; need at least {MIN_FEATURES}.")
    row = np.array([[clean[n] if clean.get(n) is not None else np.nan for n in names]], dtype=float)
    p = float(np.clip(bundle["model"].predict_pos(row)[0], 0.0, 1.0))
    return {
        "p_pd": p, "proba": [1.0 - p, p], "classes": ["HC", "PwPD"],
        "imputed_features": [k for k in selected if clean[k] is None],
        "features_used": selected, "n_model_features": len(names),
    }


# ----------------------------------------------------------------------------- web app
def build_app(bundle: Dict[str, Any], names: List[str], selected: List[str]):
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel

    class Features(BaseModel):
        jitter_local: Optional[float] = None
        Flo: Optional[float] = None
        D2: Optional[float] = None
        jitter_ddp: Optional[float] = None
        jitter_rap: Optional[float] = None
        pr_shimmer_apq11: Optional[float] = None

    app = FastAPI(title="PD-SynergyBoost voice API", version="1.0")
    lock = threading.Lock()

    def check_key(x_api_key: str) -> None:
        if API_KEY and not secrets.compare_digest(x_api_key or "", API_KEY):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "model": MODEL_FILE, "features": FEATURES}

    @app.post("/predict")
    def predict(f: Features, x_api_key: str = Header(default="")) -> Dict[str, Any]:
        check_key(x_api_key)
        data = f.model_dump() if hasattr(f, "model_dump") else f.dict()
        try:
            with lock:
                return predict_from_features(bundle, names, selected, data)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    return app


def start_server(app) -> Any:
    import uvicorn
    old = getattr(builtins, "_pd_api_server", None)      # re-running the cell: stop the old server
    if old is not None:
        old.should_exit = True
        time.sleep(1.5)
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    if not server.started:
        raise RuntimeError(f"Server did not start on port {PORT} (port busy? restart the Colab session).")
    builtins._pd_api_server = server
    return server


def open_tunnel() -> str:
    from pyngrok import ngrok
    token = NGROK_AUTH_TOKEN
    if not token:
        try:
            from google.colab import userdata  # type: ignore
            token = userdata.get("NGROK_AUTH_TOKEN")
        except Exception:
            token = ""
    if not token:
        token = getpass.getpass("Paste your ngrok authtoken: ").strip()
    ngrok.set_auth_token(token)
    for t in ngrok.get_tunnels():                         # clean up tunnels from earlier runs
        ngrok.disconnect(t.public_url)
    url = ngrok.connect(PORT, "http").public_url
    return url.replace("http://", "https://", 1)


def main() -> None:
    bundle, names, selected = load_bundle(Path(BASE_DIR) / MODEL_FILE)
    server = start_server(build_app(bundle, names, selected))
    try:                                                    # local smoke test with DUMMY numbers
        import requests
        r = requests.post(f"http://127.0.0.1:{PORT}/predict", timeout=10,
                          headers={"X-API-Key": API_KEY},
                          json={"jitter_local": 0.006, "Flo": 90.0, "D2": 2.5, "jitter_ddp": 0.009,
                                "jitter_rap": 0.003, "pr_shimmer_apq11": 0.05})
        print(f"[API] self-test (dummy values) -> HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[API] self-test failed: {type(e).__name__}: {e}")
    url = open_tunnel()
    print("\n" + "=" * 70)
    print("API is LIVE. Paste these two lines at the top of app.py:\n")
    print(f'API_URL = "{url}"')
    print(f'API_KEY = "{API_KEY}"')
    print("\nQuick check from anywhere:  " + url + "/health")
    print("Keep this Colab session running; the URL changes on every re-run.")
    print("=" * 70)


if __name__ == "__main__":
    main()
