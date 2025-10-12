import os
import joblib
import numpy as np
import shap
from flask import Flask, render_template, request, jsonify
from supabase import create_client, Client

# Initialize Flask
app = Flask(__name__)

# -----------------------------
# Load model and scaler
# -----------------------------
MODEL_PATH = "xgb_model.joblib"
SCALER_PATH = "scaler.joblib"
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Supabase setup
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Collect user info
            name = request.form.get("name")
            age = int(request.form.get("age", 0))
            gender = request.form.get("gender")

            # Symptoms
            symptoms = [
                int(request.form.get("fever", 0)),
                int(request.form.get("sore_throat", 0)),
                int(request.form.get("vomiting", 0)),
                int(request.form.get("headache", 0)),
                int(request.form.get("muscle_pain", 0)),
                int(request.form.get("abdominal_pain", 0)),
                int(request.form.get("diarrhea", 0)),
                int(request.form.get("bleeding", 0)),
                int(request.form.get("hearing_loss", 0)),
                int(request.form.get("fatigue", 0)),
            ]

            # Vitals
            vitals = [
                float(request.form.get("temperature", 0)),
                float(request.form.get("heart_rate", 0)),
                float(request.form.get("oxygen_level", 0)),
            ]

            # Prepare input
            X_input = np.array([symptoms + vitals])
            X_scaled = scaler.transform(X_input)

            # Predict
            prob = model.predict_proba(X_scaled)[0, 1]
            prediction = "Likely Lassa Fever" if prob >= 0.5 else "Unlikely Lassa Fever"

            # SHAP explainability
            explainer = shap.Explainer(model)
            shap_values = explainer(X_scaled)
            shap.save_html("templates/shap_plot.html", shap.plots.waterfall(shap_values[0], show=False))

            # -----------------------------
            # Save to Supabase
            # -----------------------------
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
                "probability": round(float(prob), 4),
                "prediction": prediction,
            }

            supabase.table("predictions").insert(record).execute()

            # Return result
            return render_template(
                "result.html",
                prediction=prediction,
                probability=round(prob * 100, 2),
                name=name,
                shap_plot_path="shap_plot.html"
            )

        except Exception as e:
            return render_template(
                "index.html",
                message=f"⚠️ Error: {str(e)}",
                message_class="danger"
            )

    return render_template("index.html")


# -----------------------------
# API route for external clients (optional)
# -----------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data"}), 400

    features = np.array([data["features"]]).astype(float)
    X_scaled = scaler.transform(features)
    prob = model.predict_proba(X_scaled)[0, 1]
    prediction = "Likely Lassa Fever" if prob >= 0.5 else "Unlikely Lassa Fever"
    return jsonify({"prediction": prediction, "probability": round(prob, 4)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
