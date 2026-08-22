"""
train_models.py
-----------------
Trains THREE models on the synthetic recovery dataset and compares them:
    1. Logistic Regression  -> interpretable baseline
    2. Random Forest        -> handles non-linear feature interactions
    3. Gradient Boosting    -> usually best raw performance

We deliberately do NOT just pick one blindly - we compare on Accuracy,
Precision, Recall, F1, and ROC-AUC, and explain why RECALL matters most
for this task (missing a recoverable customer = money left on the table,
which is more costly than a slightly-too-generous plan offer).

The best model is saved to ml/recovery_model.pkl along with the fitted
encoders, so the backend can load it and score customer+plan combinations
at inference time.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_HERE = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(_HERE, "..", "data", "customers.csv"))

NUMERIC_FEATURES = [
    "amount_owed",
    "days_overdue",
    "past_default_count",
    "past_ontime_pct",
    "reminder_open_rate",
]
CATEGORICAL_FEATURES = ["reason_code", "plan_offered"]
TARGET = "recovered"

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Preprocessing ---
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[NUMERIC_FEATURES])
X_test_num = scaler.transform(X_test[NUMERIC_FEATURES])

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_cat = encoder.fit_transform(X_train[CATEGORICAL_FEATURES])
X_test_cat = encoder.transform(X_test[CATEGORICAL_FEATURES])

X_train_final = np.hstack([X_train_num, X_train_cat])
X_test_final = np.hstack([X_test_num, X_test_cat])

feature_names = NUMERIC_FEATURES + list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))

# --- Train & compare models ---
models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.08, random_state=42
    ),
}

results = {}
fitted_models = {}

for name, model in models.items():
    model.fit(X_train_final, y_train)
    preds = model.predict(X_test_final)
    probs = model.predict_proba(X_test_final)[:, 1]

    results[name] = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
    }
    fitted_models[name] = model

print("\n=== MODEL COMPARISON ===")
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.sort_values("recall", ascending=False)
print(comparison_df)

# We select on RECALL primarily (catching recoverable customers matters most
# for a revenue-recovery use case - a false negative here is money lost),
# with ROC-AUC as a tiebreaker for overall ranking quality.
best_model_name = comparison_df.index[0]
best_model = fitted_models[best_model_name]
print(f"\nSelected model: {best_model_name}")

# Feature importance (for RF / GB) - useful for the README / demo
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=feature_names)
    print("\nTop feature importances:")
    print(importances.sort_values(ascending=False).head(8))

# --- Save everything the backend needs ---
joblib.dump(
    {
        "model": best_model,
        "model_name": best_model_name,
        "scaler": scaler,
        "encoder": encoder,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "plans": sorted(df["plan_offered"].unique().tolist()),
        "reasons": sorted(df["reason_code"].unique().tolist()),
    },
    os.path.join(_HERE, "recovery_model.pkl"),
)

with open(os.path.join(_HERE, "model_comparison.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved model -> ml/recovery_model.pkl")
print("Saved comparison metrics -> ml/model_comparison.json")
