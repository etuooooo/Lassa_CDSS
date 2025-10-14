import os
import joblib
import numpy as np
import shap
from flask import Flask, render_template, request, jsonify
from supabase import create_client, Client

# -----------------------------
# Flask
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Artifacts (model + scaler)
# -----------------------------
MODEL_PATH = "xgb_model.joblib"
SCALER_PATH = "scaler.joblib"
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# The feature order MUST match your form and training pipeline
FEATURE_NAMES = [
    "fever", "sore_throat", "vomiting", "headache", "muscle_pain",
    "abdominal_pain", "diarrhea", "bleeding", "hearing_loss", "fatigue",
    "temperature", "heart_rate", "oxygen_level"
]

# Cache the SHAP explainer (much faster per request)
EXPLAINER = shap.Explainer(model, feature_names=FEATURE_NAMES)

# Ensure /static exists for saving the HTML force plot
os.makedirs("static", exist_ok=True)

# -----------------------------
# Supabase (optional)
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# Helpers
# -----------------------------
def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def to_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return int(default)

def build_feature_row(form):
    """Return numpy row in the exact FEATURE_NAMES order."""
    vals = {
        "fever":          to_int(form.get("fever", 0)),
        "sore_throat":    to_int(form.get("sore_throat", 0)),
        "vomiting":       to_int(form.get("vomiting", 0)),
        "headache":       to_int(form.get("headache", 0)),
        "muscle_pain":    to_int(form.get("muscle_pain", 0)),
        "abdominal_pain": to_int(form.get("abdominal_pain", 0)),
        "diarrhea":       to_int(form.get("diarrhea", 0)),
        "bleeding":       to_int(form.get("bleeding", 0)),
        "hearing_loss":   to_int(form.get("hearing_loss", 0)),
        "fatigue":        to_int(form.get("fatigue", 0)),
        "temperature":    to_float(form.get("temperature", 0)),
        "heart_rate":     to_float(form.get("heart_rate", 0)),
        "oxygen_level":   to_float(form.get("oxygen_level", 0)),
    }
    return np.array([[vals[k] for k in FEATURE_NAMES]], dtype=float), vals

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Patient info
            name   = (request.form.get("name") or "").strip()
            age    = to_int(request.form.get("age", 0))
            gender = (request.form.get("gender") or "").strip()

            # Build features row (in exact order)
            X_input, raw_vals = build_feature_row(request.form)
            X_scaled = scaler.transform(X_input)

            # Predict
            prob = float(model.predict_proba(X_scaled)[0, 1])
            prediction = "Likely Lassa Fever" if prob >= 0.5 else "Unlikely Lassa Fever"

            # SHAP explanation
            explanation = EXPLAINER(X_scaled)

            # Choose plot type: "bar" (simple) or "waterfall" (detailed)
            plot_type = (request.args.get("plot", "bar") or "bar").lower()
            shap_html_path = os.path.join("static", "shap_plot.html")

            if plot_type == "waterfall":
                # Waterfall shows contributions with labeled feature=value
                shap.save_html(shap_html_path, shap.plots.waterfall(explanation[0], show=False))
            else:
                # Bar is a simpler ranked list of contributions
                shap.save_html(shap_html_path, shap.plots.bar(explanation[0], show=False))

            # Save to Supabase (best-effort)
            if supabase is not None:
                record = {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    **raw_vals,                     # expands symptom/vital fields
                    "probability": round(prob, 4),
                    "prediction": prediction,
                }
                try:
                    supabase.table("predictions").insert(record).execute()
                except Exception as log_err:
                    print(f"[Supabase] insert failed: {log_err}")

            # Render result page
            return render_template(
                "result.html",
                name=name,
                prediction=prediction,
                probability=round(prob * 100, 2),
                shap_iframe_src="shap_plot.html",  # served from /static
            )

        except Exception as e:
            return render_template(
                "index.html",
                message=f"⚠️ Error: {str(e)}",
                message_class="danger",
            )

    # GET → show the form
    return render_template("index.html")

# Optional JSON API
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    feats = data.get("features")
    if not feats:
        return jsonify({"error": "no features provided"}), 400

    X = np.array([feats], dtype=float)
    X_scaled = scaler.transform(X)
    prob = float(model.predict_proba(X_scaled)[0, 1])
    prediction = "Likely Lassa Fever" if prob >= 0.5 else "Unlikely Lassa Fever"
    return jsonify({"prediction": prediction, "probability": round(prob, 4)})

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
