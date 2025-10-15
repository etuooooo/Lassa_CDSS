#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json


# In[15]:


# reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# In[16]:


# samples
n_samples = 1500


# In[17]:


# binary symptoms (0/1)
symptoms = {
    "fever":          np.random.randint(0, 2, n_samples),
    "sore_throat":    np.random.randint(0, 2, n_samples),
    "vomiting":       np.random.randint(0, 2, n_samples),
    "headache":       np.random.randint(0, 2, n_samples),
    "muscle_pain":    np.random.randint(0, 2, n_samples),
    "abdominal_pain": np.random.randint(0, 2, n_samples),
    "diarrhea":       np.random.randint(0, 2, n_samples),
    "bleeding":       np.random.randint(0, 2, n_samples),
    "hearing_loss":   np.random.randint(0, 2, n_samples),
    "fatigue":        np.random.randint(0, 2, n_samples),
}


# In[18]:


# continuous vitals
vitals = {
    "temp_c":    np.round(np.random.uniform(36.0, 41.0, n_samples), 1),
    "heart_rate": np.random.randint(60, 140, n_samples),
    "spo2":      np.round(np.random.uniform(88, 100, n_samples), 1),
}


# In[19]:


# target (class imbalance ~ 30% positive)
target = {
    "lassa_fever": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
}


# In[20]:


# dataframe
df = pd.DataFrame({**symptoms, **vitals, **target})


# In[21]:


# add mild clinical correlation for positives
mask = df["lassa_fever"] == 1
df.loc[mask, ["fever", "vomiting", "bleeding", "fatigue"]] = 1
df.loc[mask, "temp_c"] = np.round(np.random.uniform(38.5, 41.0, mask.sum()), 1)
df.loc[mask, "spo2"]   = np.round(np.random.uniform(88.0, 95.0, mask.sum()), 1)


# In[22]:


# save
df.to_csv("synthetic_lassa.csv", index=False)
print("✅ Saved dataset -> synthetic_lassa.csv")


# In[24]:


# save metadata (now properly closed dictionary!)
meta = {
    "seed": RANDOM_STATE,
    "n_samples": int(n_samples),
    "symptom_features": list(symptoms.keys()),
    "vital_features": list(vitals.keys()),
    "target": "lassa_fever"
}
with open("dataset_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print("✅ Saved metadata -> dataset_metadata.json")


# In[25]:


# quick figures
sns.countplot(x="lassa_fever", data=df)
plt.title("Class Distribution of Lassa Fever")
plt.tight_layout()
plt.savefig("fig_class_distribution.png", dpi=300)
plt.close()


# In[26]:


corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="RdBu_r", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("fig_correlation_heatmap.png", dpi=300)
plt.close()

print("✅ Saved figures -> fig_class_distribution.png, fig_correlation_heatmap.png")


# In[ ]:




