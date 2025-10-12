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
# Artifacts
# -----------------------------
MODEL_PATH = "xgb_model.joblib"
SCALER_PATH = "scaler.joblib"
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Cache the SHAP explainer (much faster than re-creating per request)
# Works with shap==0.44.x
EXPLAINER = shap.Explainer(model)

# Ensure /static exists for saving the HTML force plot
os.makedirs("static", exist_ok=True)

# -----------------------------
# Supabase (safe: uses env vars)
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# Helpers
# -----------------------------
def to_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return float(default)

def to_int(val, default=0):
    try:
        return int(val)
    except Exception:
        return int(default)

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # --- Personal info
            name   = request.form.get("name") or ""
            age    = to_int(request.form.get("age", 0))
            gender = request.form.get("gender") or ""

            # --- Symptoms (10 binary features)
            symptoms = [
                to_int(request.form.get("fever", 0)),
                to_int(request.form.get("sore_throat", 0)),
                to_int(request.form.get("vomiting", 0)),
                to_int(request.form.get("headache", 0)),
                to_int(request.form.get("muscle_pain", 0)),
                to_int(request.form.get("abdominal_pain", 0)),
                to_int(request.form.get("diarrhea", 0)),
                to_int(request.form.get("bleeding", 0)),
                to_int(request.form.get("hearing_loss", 0)),
                to_int(request.form.get("fatigue", 0)),
            ]

            # --- Vitals (3 numeric features)
            vitals = [
                to_float(request.form.get("temperature", 0)),
                to_float(request.form.get("heart_rate", 0)),
                to_float(request.form.get("oxygen_level", 0)),
            ]

            # --- Inference
            X_input  = np.array([symptoms + vitals], dtype=float)
            X_scaled = scaler.transform(X_input)
            prob     = float(model.predict_proba(X_scaled)[0, 1])
            prediction = "Likely Lassa Fever" if prob >= 0.5 else "Unlikely Lassa Fever"

            # --- SHAP: Force plot (Visualizer) → HTML
            explanation = EXPLAINER(X_scaled)               # Explanation object
            force_vis   = shap.plots.force(explanation[0])  # Visualizer (JS)
            shap_html_path = os.path.join("static", "shap_plot.html")
            shap.save_html(shap_html_path, force_vis)

            # --- Save to Supabase (if configured)
            if supabase is not None:
                record = {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "fever": symptoms[0],
                    "sore_throat": symptoms[1],
                    "vomiting": symptoms[2],
                    "headache": symptoms[3],
                    "muscle_pain": symptoms[4],
                    "abdominal_pain": symptoms[5],
                    "diarrhea": symptoms[6],
                    "bleeding": symptoms[7],
                    "hearing_loss": symptoms[8],
                    "fatigue": symptoms[9],
                    "temperature": vitals[0],
                    "heart_rate": vitals[1],
                    "oxygen_level": vitals[2],
                    "probability": round(prob, 4),
                    "prediction": prediction,
                }
                try:
                    supabase.table("predictions").insert(record).execute()
                except Exception as log_err:
                    # Don't break the UI if logging fails
                    print(f"[Supabase] insert failed: {log_err}")

            # --- Render result page
            return render_template(
                "result.html",
                name=name,
                prediction=prediction,
                probability=round(prob * 100, 2),  # %
                shap_iframe_src="shap_plot.html"   # served from /static
            )

        except Exception as e:
            return render_template(
                "index.html",
                message=f"⚠️ Error: {str(e)}",
                message_class="danger"
            )

    # GET → show form
    return render_template("index.html")

# Optional JSON API
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    features = np.array([data.get("features", [])], dtype=float)
    if features.size == 0:
        return jsonify({"error": "no features provided"}), 400
    X_scaled = scaler.transform(features)
    prob = float(model.predict_proba(X_scaled)[0, 1])
    prediction = "Likely Lassa Fever" if prob >= 0.5 else "Unlikely Lassa Fever"
    return jsonify({"prediction": prediction, "probability": round(prob, 4)})

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
