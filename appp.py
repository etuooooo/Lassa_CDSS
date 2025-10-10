# --- top of file ---
import os
USE_SHAP = os.getenv("USE_SHAP", "0") == "1"   # set to 1 later when SHAP works

from flask import Flask, request, jsonify, render_template
import numpy as np, joblib, json

# load artifacts
scaler = joblib.load("scaler.joblib")
cal_model = joblib.load("calibrated_model.joblib")     # probabilities
xgb_model = joblib.load("xgb_model.joblib")            # for feature_importance fallback
with open("feature_cols.json") as f:
    feature_cols = json.load(f)

THRESHOLD = 0.50

# --- SHAP optional ---
if USE_SHAP:
    import shap
    shap_bg = np.load("shap_bg.npy")
    explainer = shap.TreeExplainer(xgb_model, feature_names=feature_cols)
else:
    explainer = None  # fallback later

app = Flask(__name__)

def to_row(payload):
    return np.array([[float(payload.get(k, 0)) for k in feature_cols]], dtype=float)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", feature_cols=feature_cols, threshold=THRESHOLD)

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    x = to_row(payload)
    x_scaled = scaler.transform(x)

    prob = float(cal_model.predict_proba(x_scaled)[0, 1])
    label = int(prob >= THRESHOLD)

    # --- explanations: SHAP if available; else model-level importances (static) ---
    explanations = []
    if USE_SHAP and explainer is not None:
        shap_vals = explainer.shap_values(x_scaled)[0]
        k = int(payload.get("top_k", 5))
        order = np.argsort(np.abs(shap_vals))[::-1][:k]
        explanations = [
            {"feature": feature_cols[i], "value": float(x[0, i]), "contribution": float(shap_vals[i])}
            for i in order
        ]
    else:
        # fallback: top-k global feature importances from XGBoost (gain)
        importances = getattr(xgb_model, "feature_importances_", None)
        if importances is not None:
            k = int(payload.get("top_k", 5))
            order = np.argsort(importances)[::-1][:k]
            explanations = [
                {"feature": feature_cols[i], "value": float(x[0, i]), "contribution": float(importances[i])}
                for i in order
            ]

    return jsonify({
        "label": label,
        "probability": round(prob, 4),
        "threshold": THRESHOLD,
        "top_contributors": explanations  # may be empty in fallback
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)