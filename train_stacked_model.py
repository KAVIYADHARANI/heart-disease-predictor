import numpy as np
import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Load dataset
data = np.load("artifacts/datasets.npz")
X_train, y_train = data["X_train"], data["y_train"]

# Load base models
log_reg = joblib.load("artifacts/logistic_regression.joblib")
rf = joblib.load("artifacts/random_forest.joblib")
svm = joblib.load("artifacts/svm.joblib")
xgb = joblib.load("artifacts/xgboost.joblib")

# Define stacking ensemble
estimators = [
    ("log_reg", log_reg),
    ("rf", rf),
    ("svm", svm),
    ("xgb", xgb)
]

stacked_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000)
)

# Train stacking model
stacked_model.fit(X_train, y_train)

# Save stacked model
joblib.dump(stacked_model, "artifacts/stacked_model.joblib")

print("✅ Stacked model trained and saved at artifacts/stacked_model.joblib")
