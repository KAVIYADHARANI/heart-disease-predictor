import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import os

# Load dataset
data = np.load("artifacts/datasets.npz")
X_train, y_train = data["X_train"], data["y_train"]

os.makedirs("artifacts", exist_ok=True)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
joblib.dump(log_reg, "artifacts/logistic_regression.joblib")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, "artifacts/random_forest.joblib")

# SVM
svm = SVC(probability=True, kernel="rbf")
svm.fit(X_train, y_train)
joblib.dump(svm, "artifacts/svm.joblib")

# XGBoost (⚡ fixed: removed use_label_encoder)
xgb = XGBClassifier(eval_metric="logloss", random_state=42)
xgb.fit(X_train, y_train)
joblib.dump(xgb, "artifacts/xgboost.joblib")

print("✅ Base models trained and saved in artifacts/")
