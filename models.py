# train_models.py

import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_and_evaluate_models(X_train, X_test, y_train, y_test):

    # Create model folder if not exists
    os.makedirs("model", exist_ok=True)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    results = {}

    for name, model in models.items():

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Probabilities (for AUC)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = "Not Available"

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="binary")
        recall = recall_score(y_test, y_pred, average="binary")
        f1 = f1_score(y_test, y_pred, average="binary")
        mcc = matthews_corrcoef(y_test, y_pred)

        results[name] = {
            "Accuracy": accuracy,
            "AUC Score": auc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "MCC Score": mcc
        }

        # Save model
        os.makedirs("model", exist_ok=True)
        filename = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, f"model/{filename}")

    results_df = pd.DataFrame(results).T

    return results_df
