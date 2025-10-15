# AI-Driven Lassa Fever Clinical Decision Support System (Lassa_CDSS)

This repository contains the full implementation of an AI-driven Clinical Decision Support System (CDSS) for early Lassa fever risk prediction.  
The system combines synthetic data generation, supervised machine learning, model calibration, and a Flask-based web interface connected to a Supabase database.

---

## 🧠 Overview

The CDSS supports healthcare professionals in identifying potential Lassa fever cases through:
- Symptom- and vitals-based risk assessment
- Real-time inference via a Flask API
- Cloud deployment on Render
- Statistical validation (Paired t-test and McNemar’s test)

---

## ⚙️ System Architecture

**Workflow:**  
`User Input → Flask API → Model Inference → Supabase Database → Web UI`

- `Generate_Synthetic_Dataset.py` — creates the synthetic dataset  
- `Model_Training_Testing.py` — trains and evaluates models (LR, SVM, RF, XGBoost)  
- `app.py` — serves predictions through Flask  
- `templates/` — contains the frontend (`index.html`, `result.html`)

---

## 🧪 Key Features
- Multi-model comparison with statistical significance testing  
- Calibration via Platt scaling  
- Explainability using SHAP  
- Cloud-based deployment on Render  
- Open-source and reproducible (Open Science compliant)

---

## 🧰 Installation

```bash
git clone https://github.com/etuuoooo/Lassa_CDSS.git
cd Lassa_CDSS
pip install -r requirements.txt
python app.py
