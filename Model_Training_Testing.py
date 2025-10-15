#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json, joblib, shap, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
OUT = Path(".")
OUT.mkdir(parents=True, exist_ok=True)

print("✅ Environment ready")
print("shap", shap.__version__)


# In[2]:


# If you haven't generated the data yet, run your generate_synthetic.py before this.
df = pd.read_csv("synthetic_lassa.csv")
feature_cols = [
    'fever','sore_throat','vomiting','headache','muscle_pain',
    'abdominal_pain','diarrhea','bleeding','hearing_loss','fatigue',
    'temp_c','heart_rate','spo2'
]
X = df[feature_cols].astype(float).values
y = df["lassa_fever"].astype(int).values

print("Shape:", X.shape, " Positives:", y.sum(), " Negatives:", (y==0).sum())
df.head()


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
X_tr, X_cal, y_tr, y_cal = train_test_split(
    X_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_STATE
)

scaler = MinMaxScaler()
X_tr_s  = scaler.fit_transform(X_tr)
X_cal_s = scaler.transform(X_cal)
X_test_s= scaler.transform(X_test)

print("Train:", X_tr_s.shape, "Cal:", X_cal_s.shape, "Test:", X_test_s.shape)


# In[4]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, brier_score_loss, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

models_and_grids = [
    ("LogisticRegression",
     LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
     {"C":[0.1,1,10], "penalty":["l2"], "solver":["lbfgs"]}),

    ("SVM_RBF",
     SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
     {"C":[0.5,1,10], "gamma":["scale", 0.1, 0.01]}),

    ("RandomForest",
     RandomForestClassifier(random_state=RANDOM_STATE),
     {"n_estimators":[200,400], "max_depth":[None,8,12], "min_samples_split":[2,5]}),

    ("XGBoost",
     XGBClassifier(
         n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
         colsample_bytree=1.0, reg_lambda=1.0, n_jobs=-1,
         random_state=RANDOM_STATE, eval_metric="logloss"
     ),
     {"n_estimators":[200,300,500], "max_depth":[4,6,8], "learning_rate":[0.05,0.1], "subsample":[0.8,1.0]}
    ),
]

best_models = {}
for name, base_model, grid in models_and_grids:
    gs = GridSearchCV(base_model, grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X_tr_s, y_tr)
    best_models[name] = gs.best_estimator_
    print(f"[{name}] best params:", gs.best_params_)


# In[5]:


def evaluate_model(name, clf, Xs, ys, threshold=0.5):
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(Xs)[:, 1]
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(Xs)
        proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        proba = None

    if proba is not None:
        yhat = (proba >= threshold).astype(int)
        auc  = roc_auc_score(ys, proba)
    else:
        yhat = clf.predict(Xs)
        auc  = float("nan")

    acc = accuracy_score(ys, yhat)
    prec= precision_score(ys, yhat, zero_division=0)
    rec = recall_score(ys, yhat, zero_division=0)
    f1  = f1_score(ys, yhat, zero_division=0)
    cm  = confusion_matrix(ys, yhat)
    return {"model":name,"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"auc":auc}, yhat, proba, cm

results = []
for name in ["LogisticRegression","SVM_RBF","RandomForest","XGBoost"]:
    # fit best model on all train (tr + cal) before testing
    best_models[name].fit(np.vstack([X_tr_s, X_cal_s]), np.hstack([y_tr, y_cal]))
    metrics, yhat, proba, cm = evaluate_model(name, best_models[name], X_test_s, y_test)
    results.append(metrics)
    print(metrics)
pd.DataFrame(results)


# In[6]:


from sklearn.calibration import CalibratedClassifierCV

xgb_best = best_models["XGBoost"]
cal = CalibratedClassifierCV(base_estimator=xgb_best, cv=cv, method="isotonic")
cal.fit(X_cal_s, y_cal)

# Test evaluation
from sklearn.calibration import calibration_curve
proba_cal = cal.predict_proba(X_test_s)[:, 1]
yhat_cal  = (proba_cal >= 0.5).astype(int)

auc_cal   = roc_auc_score(y_test, proba_cal)
rec_cal   = recall_score(y_test, yhat_cal)
prec_cal  = precision_score(y_test, yhat_cal)
f1_cal    = f1_score(y_test, yhat_cal)
brier     = brier_score_loss(y_test, proba_cal)
cm_cal    = confusion_matrix(y_test, yhat_cal)

metrics_cal = {"model":"XGBoost_calibrated","accuracy":accuracy_score(y_test,yhat_cal),
               "precision":prec_cal,"recall":rec_cal,"f1":f1_cal,"auc":auc_cal,"brier":brier}
metrics_cal


# In[7]:


# =============================================
# 🔍 Model Evaluation with Statistical Significance Tests
# =============================================

from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from scipy.stats import ttest_rel
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np

# ---------------------------------------------
# STEP 1: Collect 10-fold F1 scores for each model (Paired t-test)
# ---------------------------------------------
cv = 10  # same number of folds you used for GridSearchCV
f1_scores_lr = cross_val_score(best_models["LogisticRegression"], X_tr_s, y_tr, cv=cv, scoring="f1")
f1_scores_svm = cross_val_score(best_models["SVM_RBF"], X_tr_s, y_tr, cv=cv, scoring="f1")
f1_scores_rf = cross_val_score(best_models["RandomForest"], X_tr_s, y_tr, cv=cv, scoring="f1")
f1_scores_xgb = cross_val_score(best_models["XGBoost"], X_tr_s, y_tr, cv=cv, scoring="f1")

print("\nMean F1-scores from 10-fold CV:")
print(f"Logistic Regression: {f1_scores_lr.mean():.4f}")
print(f"SVM (RBF):           {f1_scores_svm.mean():.4f}")
print(f"Random Forest:       {f1_scores_rf.mean():.4f}")
print(f"XGBoost:             {f1_scores_xgb.mean():.4f}")

# ---------------------------------------------
# STEP 2: Paired t-tests comparing models with XGBoost
# ---------------------------------------------
print("\nPaired t-test (comparing F1 across 10 folds with XGBoost):")
for name, scores in {
    "Logistic Regression": f1_scores_lr,
    "SVM (RBF)": f1_scores_svm,
    "Random Forest": f1_scores_rf
}.items():
    t_stat, p_val = ttest_rel(f1_scores_xgb, scores)
    print(f"{name} vs XGBoost -> t-stat={t_stat:.3f}, p={p_val:.4f} {'(significant)' if p_val < 0.05 else '(ns)'}")

# Interpretation guidance:
# p < 0.05 → statistically significant performance difference
# p ≥ 0.05 → no statistically significant difference (ns = not significant)

# ---------------------------------------------
# STEP 3: McNemar’s test (top two models on test set)
# ---------------------------------------------
# Predict test labels
y_pred_rf = best_models["RandomForest"].predict(X_test_s)
y_pred_xgb = best_models["XGBoost"].predict(X_test_s)

# Build 2x2 contingency table
both_correct = np.sum((y_pred_rf == y_test) & (y_pred_xgb == y_test))
rf_correct_xgb_wrong = np.sum((y_pred_rf == y_test) & (y_pred_xgb != y_test))
rf_wrong_xgb_correct = np.sum((y_pred_rf != y_test) & (y_pred_xgb == y_test))
both_wrong = np.sum((y_pred_rf != y_test) & (y_pred_xgb != y_test))

table = [[both_correct, rf_correct_xgb_wrong],
         [rf_wrong_xgb_correct, both_wrong]]

result = mcnemar(table, exact=True)
print("\nMcNemar’s test comparing Random Forest vs XGBoost on test set:")
print(f"p-value = {result.pvalue:.4f} {'(significant)' if result.pvalue < 0.05 else '(ns)'}")

# Interpretation:
# p < 0.05 → significant difference in test predictions
# p ≥ 0.05 → no significant difference (ns = not significant)


# In[9]:


# Save results
df_res = pd.DataFrame(results + [metrics_cal])
df_res.to_csv(OUT / "results_table.csv", index=False)
display(df_res)

# ROC curves
plt.figure()
for name in ["LogisticRegression","SVM_RBF","RandomForest"]:
    clf = best_models[name]
    proba = clf.predict_proba(X_test_s)[:,1] if hasattr(clf,"predict_proba") else None
    if proba is None: 
        continue
    auc = roc_auc_score(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

# calibrated XGB
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, proba_cal)
plt.plot(fpr, tpr, label=f"XGBoost_calibrated (AUC={auc_cal:.3f})")

plt.plot([0,1],[0,1],'--', lw=1, color='gray')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves (Test Set)")
plt.legend(); plt.tight_layout()
plt.savefig(OUT / "fig_roc_all.png", dpi=300); plt.show()

# Confusion matrices
def plot_cm(cm, title):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues",
                xticklabels=["Pred 0","Pred 1"], yticklabels=["True 0","True 1"])
    plt.title(title); plt.tight_layout(); plt.show()

for name in ["LogisticRegression","SVM_RBF","RandomForest"]:
    clf = best_models[name]
    proba = clf.predict_proba(X_test_s)[:,1]
    cm = confusion_matrix(y_test, (proba>=0.5).astype(int))
    plot_cm(cm, f"Confusion Matrix — {name}")
    plt.savefig(OUT / f"fig_confusion_{name}.png", dpi=300)

plot_cm(cm_cal, "Confusion Matrix — XGBoost (Calibrated)")
plt.savefig(OUT / "fig_confusion_XGBoost_calibrated.png", dpi=300)

# Calibration curve
prob_true, prob_pred = calibration_curve(y_test, proba_cal, n_bins=10, strategy='quantile')
plt.figure()
plt.plot(prob_pred, prob_true, marker='o', label="Calibrated")
plt.plot([0,1],[0,1],'--', lw=1, color='gray', label="Ideal")
plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
plt.title(f"Calibration Curve — XGBoost (Brier={brier:.3f})")
plt.legend(); plt.tight_layout()
plt.savefig(OUT / "fig_calibration_xgb.png", dpi=300); plt.show()

print("✅ Saved: results_table.csv, fig_roc_all.png, fig_confusion_*.png, fig_calibration_xgb.png")


# In[10]:


# Use base (uncalibrated) XGB for SHAP TreeExplainer
explainer = shap.TreeExplainer(xgb_best, feature_names=feature_cols)

# Example case
x_example = X_test_s[0:1]
shap_vals = explainer.shap_values(x_example)[0]
order = np.argsort(np.abs(shap_vals))[::-1][:8]

plt.figure(figsize=(6,4))
plt.bar([feature_cols[i] for i in order], shap_vals[order])
plt.xticks(rotation=45, ha="right")
plt.title("Example SHAP Contributions (1 Test Case)")
plt.tight_layout(); 
plt.savefig(OUT / "fig_shap_example.png", dpi=300); 
plt.show()

print("✅ Saved: fig_shap_example.png")


# In[11]:


# SHAP background for fast per-request explanations
rs = np.random.RandomState(RANDOM_STATE)
bg_idx = rs.choice(np.arange(X_tr_s.shape[0]), size=min(200, X_tr_s.shape[0]), replace=False)
bg = X_tr_s[bg_idx]
np.save(OUT / "shap_bg.npy", bg)

joblib.dump(scaler, OUT / "scaler.joblib")
joblib.dump(xgb_best, OUT / "xgb_model.joblib")            # tree model for SHAP
joblib.dump(cal, OUT / "calibrated_model.joblib")          # calibrated proba
with open(OUT / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print("✅ Artifacts saved: scaler.joblib, xgb_model.joblib, calibrated_model.joblib, shap_bg.npy, feature_cols.json")


# In[12]:


pd.options.display.float_format = "{:.4f}".format
display(pd.read_csv("results_table.csv"))


# In[ ]:




