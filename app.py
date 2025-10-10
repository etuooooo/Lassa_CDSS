from flask import Flask, request, jsonify, render_template
import numpy as np, joblib, json, shap

# load artifacts
scaler = joblib.load("scaler.joblib")
xgb_model = joblib.load("xgb_model.joblib")            # base model for SHAP
cal_model = joblib.load("calibrated_model.joblib")     # calibrated wrapper
with open("feature_cols.json") as f:
    feature_cols = json.load(f)
shap_bg = np.load("shap_bg.npy")

# global SHAP explainer (fast)
explainer = shap.TreeExplainer(xgb_model, feature_names=feature_cols)

THRESHOLD = 0.50

app = Flask(__name__)

def to_row(payload):
    # convert incoming JSON to the exact feature order
    x = np.array([[float(payload.get(k, 0)) for k in feature_cols]], dtype=float)
    return x

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", feature_cols=feature_cols, threshold=THRESHOLD)

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    x = to_row(payload)

    # scale like training
    x_scaled = scaler.transform(x)

    # calibrated probability + label
    prob = float(cal_model.predict_proba(x_scaled)[0, 1])
    label = int(prob >= THRESHOLD)

    # SHAP contributions (on scaled input to match training)
    shap_vals = explainer.shap_values(x_scaled)[0]  # 1D
    k = int(payload.get("top_k", 5))
    order = np.argsort(np.abs(shap_vals))[::-1][:k]
    top_contrib = [
        {"feature": feature_cols[i], "value": float(x[0, i]), "contribution": float(shap_vals[i])}
        for i in order
    ]

    return jsonify({
        "label": label,
        "probability": round(prob, 4),
        "threshold": THRESHOLD,
        "top_contributors": top_contrib
    })

if __name__ == "__main__":
    # dev server; in prod use waitress/gunicorn
    app.run(host="0.0.0.0", port=8000, debug=True)
