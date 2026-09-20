import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier

from preprocessing import load_and_prepare_data


# ==================================================
# PATHS
# ==================================================

DATA_PATH = r"D:\AI-Connected-Vehicle-Analytics\data\historical\connected_vehicle_telemetry.csv"

MODEL_PATH = r"D:\AI-Connected-Vehicle-Analytics\models\xgboost_vehicle_failure.pkl"


# ==================================================
# 1. LOAD DATA
# ==================================================

X, y = load_and_prepare_data(DATA_PATH)

print("Feature shape:", X.shape)

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget proportions:")
print(y.value_counts(normalize=True))


# ==================================================
# 2. FEATURES

# ==================================================

print("\nFeatures used by model:")

print(X.columns.tolist())


# ==================================================
# 3. TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==================================================
# 4. CLASS IMBALANCE
# ==================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = np.sqrt(
    negative / positive
)

print("\nNegative samples:", negative)
print("Positive samples:", positive)

print(
    "Adjusted scale_pos_weight:",
    scale_pos_weight
)


# ==================================================
# 5. XGBOOST MODEL
# ==================================================

model = XGBClassifier(

    n_estimators=300,

    max_depth=4,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    min_child_weight=3,

    gamma=1.0,

    scale_pos_weight=scale_pos_weight,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42
)


# ==================================================
# 6. TRAIN
# ==================================================

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ==================================================
# 7. PREDICT PROBABILITIES
# ==================================================

y_probs = model.predict_proba(
    X_test
)[:, 1]


# ==================================================
# 8. FIND BEST F1 THRESHOLD
# ==================================================

precisions, recalls, thresholds = precision_recall_curve(
    y_test,
    y_probs
)

f1_scores = (
    2 * precisions * recalls
) / (
    precisions + recalls + 1e-10
)

# Last precision/recall value has no corresponding threshold
best_idx = np.argmax(
    f1_scores[:-1]
)

best_threshold = thresholds[best_idx]

print(
    f"\nOptimal Decision Threshold: "
    f"{best_threshold:.4f}"
)


# ==================================================
# 9. FINAL PREDICTIONS
# ==================================================

y_pred = (
    y_probs >= best_threshold
).astype(int)


# ==================================================
# 10. EVALUATION
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


print("\n====================================")
print("       XGBOOST MODEL RESULTS")
print("====================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==================================================
# 11. FEATURE IMPORTANCE
# ==================================================

importance = model.feature_importances_

feature_importance = sorted(
    zip(X.columns, importance),
    key=lambda x: x[1],
    reverse=True
)


print("\n====================================")
print("       TOP 15 FEATURES")
print("====================================")

for feature, score in feature_importance[:15]:

    print(
        f"{feature:35s} {score:.4f}"
    )


# ==================================================
# 12. SAVE MODEL
# ==================================================

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)


model_data = {

    "model": model,

    "features": X.columns.tolist(),

    "threshold": float(best_threshold)
}


joblib.dump(
    model_data,
    MODEL_PATH
)


print("\n====================================")
print("MODEL SAVED SUCCESSFULLY")
print("====================================")

print(MODEL_PATH)