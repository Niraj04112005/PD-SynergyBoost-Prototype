import os
import uvicorn
from pathlib import Path
import colab_api

# Point to the local folder
BASE_DIR = os.getcwd()
os.environ["PD_BASE_DIR"] = BASE_DIR

print("Loading model locally...")
bundle, names, selected = colab_api.load_bundle(
    Path(BASE_DIR) / colab_api.MODEL_FILE)
print(f"Model loaded! Selected features: {selected}")

# Build the FastAPI app
app = colab_api.build_app(bundle, names, selected)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("API is running locally at: http://127.0.0.1:8000")
    print("Keep this terminal open!")
    print("="*50 + "\n")

    # Start the server directly (no ngrok!)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
