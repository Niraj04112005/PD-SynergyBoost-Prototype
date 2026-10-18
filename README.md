# PD-SynergyBoost: Multi-Modal Parkinson's Detection

A final-year research prototype for early detection of Parkinson's Disease using **Voice and Wearable Tremor fusion**.

## Key Innovations
1. **CSFS (Complementary Synergy Feature Selection):** Reduces acoustic features down to a mathematically complementary subset to prevent overfitting and mitigate age confounds.
2. **MHS (Modality Health Score):** Uses Mahalanobis distance to monitor sensor health. If a sensor drops below 0.3 health, the system automatically discards it and relies 100% on the healthy sensor.
3. **Regularized Stacking:** Uses 3 base learners (LightGBM, XGBoost, RF) with an Elastic-Net Logistic Regression meta-learner.

## The Streamlit Prototype
The prototype processes real `.wav` files via `librosa` for the voice modality and simulates the wearable tremor sensor via sliders. 

### How to run locally:
1. `pip install -r requirements.txt`
2. `streamlit run app.py`

## Datasets & Limitations
- **Voice Data:** Figshare (Prior et al.). 
- **Tremor Data:** ALAMEDA (Zenodo).
- *Note: This is a cross-cohort late-fusion prototype (the datasets do not share patients). True paired fusion requires custom IoT hardware (ESP32 + MPU6050), currently in development.
- Due to GitHub file size limits, the trained .pkl models are hosted on Google Drive. [[Insert your Drive link here]](https://drive.google.com/drive/folders/16OX2ihLiXjf9lppAlN5VqyyIMS2FC-B0?usp=drive_link)
