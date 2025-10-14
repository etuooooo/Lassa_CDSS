import os
import time
import io
import joblib
import numpy as np
import shap
from flask import (
    Flask, render_template, request, jsonify,
    send_file, session
)
from supabase import create_client, Client

# Headless plotting for PNG charts (bar/waterfall and PDF fallback)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")

# ----- Artifacts -----
MODEL_PATH = "xgb_model.joblib"
SCALER_PATH = "scaler.joblib"
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURE_NAMES = [
    "fever", "sore_throat", "vomiting", "headache", "muscle_pain",
    "abdominal_pain", "diarrhea", "bleeding", "hearing_loss", "fatigue",
    "temperature", "heart_rate", "oxygen_level"
]
EXPLAINER = shap.Explainer(model, feature_names=FEATURE_NAMES)
os.makedirs("static", exist_ok=True)

# ----- Supabase (optional) -----
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----- Helpers -----
def to_float(x, default=0.0):
    try: return float(x)
    except Exception: return float(default)

def to_int(x, default=0):
    try: return int(x)
    except Exception: return int(default)

def build_feature_row(form):
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
    X = np.array([[vals[k] for k in FEATURE_NAMES]], dtype=float)
    return X, vals

def top_k_contributors(explanation, raw_vals_dict, k=5):
    contribs = explanation.values[0]
    rows = []
    for i, name in enumerate(FEATURE_NAMES):
        c = float(contribs[i])
        val = raw_vals_dict[name]
        if name == "temperature":
            disp_val = f"{val:.1f} °C"
        elif name == "heart_rate":
            disp_val = f"{val:.0f} bpm"
        elif name == "oxygen_level":
            disp_val = f"{val:.0f} %"
        else:
            disp_val = str(val)
        rows.append({
            "feature": name.replace("_", " ").title(),
            "value": disp_val,
            "contribution": round(c, 5),
            "direction": "↑ increases risk" if c > 0 else ("↓ decreases risk" if c < 0 else "– no effect"),
            "impact": abs(c)
        })
    rows.sort(key=lambda r: r["impact"], reverse=True)
    return rows[:k]

# ----- Routes -----
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            name   = (request.form.get("name") or "").strip()
            age    = to_int(request.form.get("age", 0))
            gender = (request.form.get("gender") or "").strip()

            X_raw, raw_vals = build_feature_row(request.form)
            X_scaled = scaler.transform(X_raw)

            prob = float(model.predict_proba(X_scaled)[0, 1])
            prediction = "Likely Lassa Fever" if prob >= 0.5 else "Unlikely Lassa Fever"

            # SHAP (label with RAW for readability)
            explanation = EXPLAINER(X_scaled)
            explanation.feature_names = FEATURE_NAMES
            explanation.data = X_raw

            plot_type = (request.args.get("plot", "bar") or "bar").lower()
            shap_iframe_src = None
            shap_img_src = None

            if plot_type == "force":
                # HTML (Visualizer)
                vis = shap.plots.force(explanation[0])
                html_path = os.path.join("static", "shap_plot.html")
                shap.save_html(html_path, vis)
                shap_iframe_src = "shap_plot.html"

                # ALSO create a PNG bar for PDF
                png_path = os.path.join("static", "shap_plot.png")
                plt.figure(figsize=(9, 5))
                shap.plots.bar(explanation[0], show=False)
                plt.tight_layout()
                plt.savefig(png_path, dpi=130, bbox_inches="tight")
                plt.close()
            else:
                # bar/waterfall → PNG (with sensible size)
                png_path = os.path.join("static", "shap_plot.png")
                figsize = (9, 5) if plot_type == "bar" else (9, 6)
                plt.figure(figsize=figsize)
                if plot_type == "waterfall":
                    shap.plots.waterfall(explanation[0], show=False)
                else:
                    shap.plots.bar(explanation[0], show=False)
                plt.tight_layout()
                plt.savefig(png_path, dpi=130, bbox_inches="tight")
                plt.close()
                shap_img_src = "shap_plot.png"

            top_feats = top_k_contributors(explanation, raw_vals, k=5)

            # Log to Supabase (best-effort)
            if supabase is not None:
                try:
                    supabase.table("predictions").insert({
                        "name": name, "age": age, "gender": gender,
                        **raw_vals, "probability": round(prob, 4),
                        "prediction": prediction
                    }).execute()
                except Exception as log_err:
                    print(f"[Supabase] insert failed: {log_err}")

            # Store minimal context for PDF export
            session["last_result"] = {
                "name": name,
                "prediction": prediction,
                "probability": round(prob * 100, 2),
                "top_features": top_feats,
                "timestamp": int(time.time()),
            }

            return render_template(
                "result.html",
                name=name,
                prediction=prediction,
                probability=round(prob * 100, 2),
                shap_iframe_src=shap_iframe_src,
                shap_img_src=shap_img_src,
                top_features=top_feats,
                plot_type=plot_type,
                cache_bust=str(int(time.time()))
            )

        except Exception as e:
            return render_template("index.html", message=f"⚠️ Error: {str(e)}", message_class="danger")

    return render_template("index.html")

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

# ---- PDF download ----
@app.route("/download/result.pdf")
def download_result_pdf():
    last = session.get("last_result")
    if not last:
        return jsonify({"error": "No result found. Please run a prediction first."}), 400

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.lib import colors

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    x0 = 40
    y = height - 50

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x0, y, "Lassa Fever AI Prediction — Report")
    y -= 24

    # Summary
    c.setFont("Helvetica", 11)
    c.drawString(x0, y, f"Patient: {last.get('name') or '-'}"); y -= 16
    c.drawString(x0, y, f"Outcome: {last.get('prediction')}"); y -= 16
    c.drawString(x0, y, f"Predicted Probability: {last.get('probability')}%"); y -= 24

    # SHAP image
    png_path = os.path.join("static", "shap_plot.png")
    if os.path.exists(png_path):
        try:
            img = ImageReader(png_path)
            img_w = width - 2 * x0
            img_h = img_w * 0.56
            c.drawImage(img, x0, y - img_h, img_w, img_h, preserveAspectRatio=True, mask='auto')
            y -= (img_h + 16)
        except Exception:
            c.setFillColor(colors.red)
            c.drawString(x0, y, "SHAP figure could not be embedded.")
            c.setFillColor(colors.black)
            y -= 16

    # Table
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x0, y, "Top 5 contributing factors"); y -= 18
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x0, y, "Feature")
    c.drawString(x0 + 200, y, "Entered Value")
    c.drawString(x0 + 330, y, "Direction")
    c.drawString(x0 + 460, y, "Contribution (|SHAP|)")
    y -= 14
    c.setLineWidth(0.5)
    c.line(x0, y, width - x0, y)
    y -= 12
    c.setFont("Helvetica", 10)

    for r in last.get("top_features", []):
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica-Bold", 12)
            c.drawString(x0, y, "Top 5 contributing factors (cont.)"); y -= 18
            c.setFont("Helvetica-Bold", 10)
            c.drawString(x0, y, "Feature")
            c.drawString(x0 + 200, y, "Entered Value")
            c.drawString(x0 + 330, y, "Direction")
            c.drawString(x0 + 460, y, "Contribution (|SHAP|)")
            y -= 14
            c.setLineWidth(0.5)
            c.line(x0, y, width - x0, y)
            y -= 12
            c.setFont("Helvetica", 10)

        c.drawString(x0, y, r["feature"])
        c.drawString(x0 + 200, y, str(r["value"]))
        c.drawString(x0 + 330, y, r["direction"])
        c.drawRightString(width - x0, y, f'{r["impact"]:.5f}')
        y -= 14

    c.showPage()
    c.save()
    buf.seek(0)
    filename = f"AI_Lassa_Report_{(last.get('name') or 'patient').replace(' ','_')}_{last.get('timestamp')}.pdf"
    return send_file(buf, as_attachment=True, download_name=filename, mimetype="application/pdf")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
