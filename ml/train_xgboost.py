import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

from .preprocessing import load_and_prepare_data


# ==================================================
# PATHS
# ==================================================

DATA_PATH = (
    r"D:\AI-Connected-Vehicle-Analytics"
    r"\data\historical\connected_vehicle_telemetry.csv"
)

MODEL_PATH = (
    r"D:\AI-Connected-Vehicle-Analytics"
    r"\models\xgboost_vehicle_failure.pkl"
)


# ==================================================
# LOAD DATA
# ==================================================

print("=" * 70)
print("XGBOOST VEHICLE FAILURE PREDICTION")
print("=" * 70)

print("\nLoading dataset...")

X, y = load_and_prepare_data(DATA_PATH)

print("\nFeature shape:", X.shape)


# ==================================================
# FEATURES USED BY MODEL
# ==================================================

print("\n====================================")
print("       FEATURES USED BY MODEL")
print("====================================")

for i, feature in enumerate(X.columns, start=1):
    print(f"{i:2d}. {feature}")

print("\nTotal features:", len(X.columns))


# ==================================================
# DATASET SPLIT
# ==================================================

# 80% training + validation
# 20% testing

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 80% of training data → training
# 20% of training data → validation
#
# Final:
# 64% training
# 16% validation
# 20% testing

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)


print("\n====================================")
print("          DATASET SPLIT")
print("====================================")

print("Training set   :", X_train.shape)
print("Validation set :", X_val.shape)
print("Test set       :", X_test.shape)


# ==================================================
# CLASS INFORMATION
# ==================================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = (
    negative_count / positive_count
)

print("\n====================================")
print("          CLASS INFORMATION")
print("====================================")

print("Negative samples :", negative_count)
print("Positive samples :", positive_count)
print(
    "Scale Pos Weight :",
    round(scale_pos_weight, 4)
)


# ==================================================
# FINAL MODEL PARAMETERS
# ==================================================

n_estimators = 1500
max_depth = 4
learning_rate = 0.03
subsample = 0.8
colsample_bytree = 0.8
min_child_weight = 5
gamma = 0.5
reg_alpha = 0.1
reg_lambda = 3.0


# ==================================================
# MODEL PARAMETERS
# ==================================================

print("\n====================================")
print("       FINAL MODEL PARAMETERS")
print("====================================")

print("n_estimators     :", n_estimators)
print("max_depth        :", max_depth)
print("learning_rate    :", learning_rate)
print("subsample        :", subsample)
print("colsample_bytree :", colsample_bytree)
print("min_child_weight :", min_child_weight)
print("gamma            :", gamma)
print("reg_alpha        :", reg_alpha)
print("reg_lambda       :", reg_lambda)
print("scale_pos_weight :", round(scale_pos_weight, 4))


# ==================================================
# CREATE XGBOOST MODEL
# ==================================================

model = XGBClassifier(

    n_estimators=n_estimators,

    learning_rate=learning_rate,

    max_depth=max_depth,

    min_child_weight=min_child_weight,

    subsample=subsample,

    colsample_bytree=colsample_bytree,

    gamma=gamma,

    reg_alpha=reg_alpha,

    reg_lambda=reg_lambda,

    scale_pos_weight=scale_pos_weight,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1,

    early_stopping_rounds=50
)


# ==================================================
# TRAIN MODEL
# ==================================================

print("\n====================================")
print("          TRAINING MODEL")
print("====================================")

model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_val, y_val)
    ],
    verbose=False
)

print("\nModel training completed.")


# ==================================================
# PREDICTION
# ==================================================

# Probability of failure
y_probability = model.predict_proba(
    X_test
)[:, 1]


# Default classification threshold
threshold = 0.50

y_pred = (
    y_probability >= threshold
).astype(int)


# ==================================================
# MODEL PERFORMANCE
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)


print("\n====================================")
print("          MODEL PERFORMANCE")
print("====================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")


# ==================================================
# CLASSIFICATION REPORT
# ==================================================

print("\n====================================")
print("       CLASSIFICATION REPORT")
print("====================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Failure",
            "Failure"
        ],
        zero_division=0
    )
)


# ==================================================
# CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n====================================")
print("          CONFUSION MATRIX")
print("====================================")

print(cm)

print("\nTrue Negative :", cm[0][0])
print("False Positive:", cm[0][1])
print("False Negative:", cm[1][0])
print("True Positive :", cm[1][1])


# ==================================================
# FEATURE IMPORTANCE
# ==================================================

importance = model.feature_importances_

feature_importance = sorted(
    zip(
        X.columns,
        importance
    ),
    key=lambda x: x[1],
    reverse=True
)


print("\n====================================")
print("        FEATURE IMPORTANCE")
print("====================================")

for feature, score in feature_importance[:15]:

    print(
        f"{feature:35s} {score:.4f}"
    )


# ==================================================
# SAVE MODEL
# ==================================================

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)


model_data = {

    "model": model,

    "features": X.columns.tolist(),

    "metrics": {

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1),

        "roc_auc": float(roc_auc),

        "pr_auc": float(pr_auc)
    },

    "threshold": threshold,

    "hyperparameters": {

        "n_estimators": n_estimators,

        "max_depth": max_depth,

        "learning_rate": learning_rate,

        "subsample": subsample,

        "colsample_bytree": colsample_bytree,

        "min_child_weight": min_child_weight,

        "gamma": gamma,

        "reg_alpha": reg_alpha,

        "reg_lambda": reg_lambda,

        "scale_pos_weight": scale_pos_weight
    }
}


joblib.dump(
    model_data,
    MODEL_PATH
)


# ==================================================
# FINAL OUTPUT
# ==================================================

print("\n====================================")
print("          MODEL SAVED")
print("====================================")

print("Saved to:")
print(MODEL_PATH)

print("\nXGBoost training completed successfully.")