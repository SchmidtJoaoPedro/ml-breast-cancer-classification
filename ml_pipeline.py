import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from joblib import dump
import sys

data = load_breast_cancer(as_frame=True)
X = data.frame.drop(columns=["target"])
y = data.frame["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))
])

param_grid = {
    "clf__n_estimators": [100, 250],
    "clf__max_depth": [None, 6, 12],
    "clf__min_samples_split": [2, 5],
    "clf__min_samples_leaf": [1, 2]
}

cv = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=0)
cv.fit(X_train, y_train)

best = cv.best_estimator_
y_pred = best.predict(X_test)
y_prob = best.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("Best params:", cv.best_params_)
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC AUC: {roc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

dump(best, "model_breast_cancer_rf.joblib")
print('Modelo salvo em "model_breast_cancer_rf.joblib"')
