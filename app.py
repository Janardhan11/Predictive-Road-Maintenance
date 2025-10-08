# app.py — Streamlit app: Image -> Condition -> Months_to_poor -> Risk Score
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import tensorflow as tf
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).parent.resolve()

st.set_page_config(page_title="Road Predictive Maintenance Demo", layout="wide")

# ----------------------------
# CONFIG (relative paths -> Docker/GitHub friendly)
# ----------------------------
CLASSIFIER_PATH = str(BASE_DIR / "models" / "road_condition_mobilenetv2_aug.h5")
MONTHS_MODEL_PATH = str(BASE_DIR / "models" / "months_to_poor_model.joblib")
DATASET_CSV_PATH = str(BASE_DIR / "data" / "simulated_road_deterioration.csv")

# ----------------------------
# Helper functions
# ----------------------------
@st.cache_resource
def load_classifier(path: str):
    """Load Keras model safely (no compile to avoid missing custom objects)."""
    if not os.path.exists(path):
        return None
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        st.error(f"Failed to load classifier: {e}")
        return None

@st.cache_resource
def load_months_model(path: str):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load months model: {e}")
        return None

def preprocess_image(img: Image.Image, target_size=(224, 224)):
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    return arr

def predict_condition(model, img_array, class_names):
    """Support multi-class logits or single-probability binary outputs."""
    x = np.expand_dims(img_array, axis=0)
    preds = model.predict(x, verbose=0)

    if preds.ndim == 2 and preds.shape[1] > 1:
        idx = int(np.argmax(preds, axis=1)[0])
        conf = float(np.max(preds))
    else:
        # binary: single probability for "positive" class
        p = float(np.clip(preds.flatten()[0], 0.0, 1.0))
        idx = int(p >= 0.5)
        conf = p

    idx = max(0, min(idx, len(class_names) - 1))  # guard against mismatch
    return class_names[idx], conf

def predict_months(pipe, row_dict):
    X = pd.DataFrame([row_dict])
    pred = pipe.predict(X)[0]
    return float(pred)

def compute_raw_risk(pred_months_float, traffic_day, pct_heavy_truck):
    """Higher means riskier. Uses continuous months for smoother scores."""
    m = max(0.0, float(pred_months_float))
    raw = (1.0 / (m + 1.0)) * float(traffic_day) * (1.0 + float(pct_heavy_truck))
    return float(raw)

def nearest_neighbor_fallback(df, row, thresh=0.5):
    """If the model outputs very small months, approximate from nearest row."""
    try:
        num_cols = [
            "traffic_day",
            "pct_heavy_truck",
            "rainfall_30d",
            "months_since_last_maintenance",
        ]
        if not all(c in df.columns for c in num_cols):
            return None
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        Xnum = scaler.fit_transform(df[num_cols].values)
        q = np.array([[row[c] for c in num_cols]], dtype=float)
        q_scaled = scaler.transform(q)
        dists = np.linalg.norm(Xnum - q_scaled, axis=1)
        best_idx = int(np.argmin(dists))
        return float(df.iloc[best_idx]["months_to_poor"])
    except Exception:
        return None

def normalize_category(s, mapping):
    if s is None:
        return s
    s = str(s).strip()
    if s in mapping:
        return mapping[s]
    s_title = s.title()
    if s_title in mapping:
        return mapping[s_title]
    s_lower = s.lower()
    if s_lower in mapping:
        return mapping[s_lower]
    return s_title

# ----------------------------
# UI
# ----------------------------
st.title("Predictive Road Maintenance — Demo")
st.write(
    "Upload a road image. The app classifies its condition, then predicts months to deterioration "
    "and a risk score using your metadata."
)

# quick cache reset for model hot-swaps
st.sidebar.button("♻ Reload models", on_click=lambda: (load_classifier.clear(), load_months_model.clear()))

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1) Upload road image")
    uploaded = st.file_uploader("Choose an image (jpg / png)", type=["jpg", "jpeg", "png"])
    st.markdown("**Classifier model path:** `" + CLASSIFIER_PATH + "`")
    classifier = load_classifier(CLASSIFIER_PATH)

    default_class_names = ["Good", "Satisfactory", "Poor", "Very Poor"]
    class_names_input = st.text_input(
        "Classifier class names (comma separated) — leave default if OK:",
        ",".join(default_class_names),
    )
    class_names = [c.strip() for c in class_names_input.split(",") if c.strip()]

    predicted_label = None
    pred_conf = None

    if uploaded is not None:
        try:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded image", use_column_width=True)
            if classifier is not None:
                arr = preprocess_image(img, target_size=(224, 224))
                pred_label, pred_conf = predict_condition(classifier, arr, class_names)
                predicted_label = pred_label
                st.success(f"Predicted condition: **{predicted_label}**  (confidence {pred_conf:.2f})")
            else:
                st.warning("Classifier model not loaded — select the condition manually on the right.")
        except Exception as e:
            st.error(f"Error reading/predicting image: {e}")

with col2:
    st.header("2) Metadata (condition can be auto-filled)")
    condition_mapping = {
        "good": "Good", "Good": "Good", "GOOD": "Good",
        "satisfactory": "Satisfactory", "Satisfactory": "Satisfactory",
        "poor": "Poor", "Poor": "Poor",
        "very poor": "Very Poor", "very_poor": "Very Poor", "Very Poor": "Very Poor",
    }
    pavement_mapping = {
        "asphalt": "Asphalt", "Asphalt": "Asphalt",
        "concrete": "Concrete", "Concrete": "Concrete",
        "gravel": "Gravel", "Gravel": "Gravel",
    }

    # prefill with model output if present; else allow manual
    if predicted_label is not None:
        condition_now_raw = predicted_label
    else:
        # show selectbox with the same order the user chose/typed
        condition_now_raw = st.selectbox("Condition (choose)", options=class_names or default_class_names)

    condition_now = normalize_category(condition_now_raw, condition_mapping)

    traffic_day = st.number_input("Traffic (vehicles per day)", min_value=0, value=10000, step=1)
    pct_heavy_truck = st.slider("Percent heavy trucks (0.00 – 1.00)", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    rainfall_30d = st.number_input("Rainfall in last 30 days (mm)", min_value=0, value=50, step=1)
    pavement_type_raw = st.selectbox("Pavement type", options=["Asphalt", "Concrete", "Gravel"])
    pavement_type = normalize_category(pavement_type_raw, pavement_mapping)
    months_since_last_maintenance = st.number_input("Months since last maintenance", min_value=0, value=12, step=1)

    st.markdown("---")
    st.write("Click **Predict** to get months to deterioration and risk score.")

    months_model = load_months_model(MONTHS_MODEL_PATH)

    if st.button("Predict"):
        if months_model is None:
            st.error("Months-to-poor model not loaded. Check MONTHS_MODEL_PATH.")
        else:
            row = {
                "traffic_day": int(traffic_day),
                "pct_heavy_truck": float(pct_heavy_truck),
                "rainfall_30d": int(rainfall_30d),
                "pavement_type": pavement_type,
                "months_since_last_maintenance": int(months_since_last_maintenance),
                "condition_now": condition_now,
            }

            st.subheader("Input sent to model")
            display_df = pd.DataFrame([{
                "Traffic (vehicles/day)": row["traffic_day"],
                "Heavy trucks (%)": f"{row['pct_heavy_truck']*100:.0f}%",
                "Rainfall (30d, mm)": row["rainfall_30d"],
                "Pavement type": row["pavement_type"],
                "Months since last maintenance": row["months_since_last_maintenance"],
                "Condition (from image)": row["condition_now"],
            }])
            st.table(display_df)

            try:
                raw_pred_months = predict_months(months_model, row)  # float
                st.write(f"Raw model output (months, before rounding): **{raw_pred_months:.3f}**")

                # clean integer months for display
                months_pred_int = int(np.round(raw_pred_months))
                months_pred_int = max(0, months_pred_int)

                # Fallback if prediction is extremely small
                fallback_used = False
                if raw_pred_months <= 0.5 and os.path.exists(DATASET_CSV_PATH):
                    try:
                        df_hist = pd.read_csv(DATASET_CSV_PATH)
                        nearest = nearest_neighbor_fallback(df_hist, row)
                        if nearest is not None and nearest > 0:
                            st.info(f"Model output is very small; using historical nearest-segment fallback: {nearest:.0f} months")
                            months_pred_int = int(nearest)
                            # keep raw_pred_months for risk smoothing, but show fallback for months
                            fallback_used = True
                    except Exception:
                        pass

                # Risk uses continuous months for smoother score
                raw_risk = compute_raw_risk(raw_pred_months, traffic_day, pct_heavy_truck)

                # Normalize risk 0–100
                if os.path.exists(DATASET_CSV_PATH):
                    try:
                        df_all = pd.read_csv(DATASET_CSV_PATH)
                        rr_all = (1.0 / (np.clip(df_all["months_to_poor"].values + 1, 1, None))) \
                                 * df_all["traffic_day"].values * (1 + df_all["pct_heavy_truck"].values)
                        min_r, max_r = float(rr_all.min()), float(rr_all.max())
                        if max_r > min_r:
                            risk_norm_pct = int(np.clip(np.round((raw_risk - min_r) / (max_r - min_r) * 100), 0, 100))
                        else:
                            heuristic_max = 50000 * 1.6
                            risk_norm_pct = int(min(100, raw_risk / heuristic_max * 100))
                    except Exception:
                        heuristic_max = 50000 * 1.6
                        risk_norm_pct = int(min(100, raw_risk / heuristic_max * 100))
                else:
                    heuristic_max = 50000 * 1.6
                    risk_norm_pct = int(min(100, raw_risk / heuristic_max * 100))

                # Human-friendly messages
                st.success(f"Predicted months until downgrade (to Poor or worse): **{months_pred_int} months**")
                if months_pred_int == 0 and not fallback_used:
                    st.warning("⚠️ This road segment is already at or near failure. Immediate maintenance recommended.")
                elif months_pred_int <= 3:
                    st.warning("⚠️ This road may deteriorate within the next 3 months. High priority for preventive action.")
                elif months_pred_int <= 6:
                    st.info("ℹ️ This road may deteriorate within 6 months. Plan preventive maintenance soon.")
                else:
                    st.info("✅ This road is stable for now. Monitor and re-evaluate periodically.")

                st.write(f"Risk score (0–100): **{risk_norm_pct}/100** — higher means more urgent")
                st.write(f"Raw risk (un-normalized): {int(raw_risk)}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.caption(
    "Note: Demo app. For production, validate with real data, secure model paths, add monitoring, "
    "and recalibrate normalization on deployment data."
)
