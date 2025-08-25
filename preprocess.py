import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load dataset
df = pd.read_csv("heart.csv")

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ensure artifacts folder exists
os.makedirs("artifacts", exist_ok=True)

# Save preprocessed data
np.savez("artifacts/datasets.npz",
         X_train=X_train_scaled, X_test=X_test_scaled,
         y_train=y_train, y_test=y_test)

# Save scaler
joblib.dump(scaler, "artifacts/scaler.joblib")

print("✅ Preprocessing complete. Saved datasets and scaler in artifacts/")
