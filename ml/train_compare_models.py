import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = r"D:\AI-Connected-Vehicle-Analytics"

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "historical",
    "connected_vehicle_telemetry.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# CREATE TARGET
# ============================================================

df["failure"] = (
    (df["engine_failure_imminent"] == 1)
    | (df["brake_issue_imminent"] == 1)
    | (df["battery_issue_imminent"] == 1)
).astype(int)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nCreating engineered features...")

# Vehicle age
vehicle_age = 2026 - df["vehicle_year"]
vehicle_age = vehicle_age.clip(lower=1)

# 1. km_per_year
df["km_per_year"] = (
    df["odometer_km"] / vehicle_age
)

# 2. hours_per_km
df["hours_per_km"] = (
    df["engine_hours"] /
    (df["odometer_km"] + 1)
)

# 3. temp_ratio
df["temp_ratio"] = (
    df["engine_temp_c"] /
    (df["coolant_temp_c"] + 1)
)

# 4. power_draw_w
df["power_draw_w"] = (
    df["battery_voltage_v"] *
    df["battery_current_a"]
)

# 5. wheel_speed_std
wheel_speed_columns = [
    "wheel_speed_fl_kph",
    "wheel_speed_fr_kph",
    "wheel_speed_rl_kph",
    "wheel_speed_rr_kph"
]

df["wheel_speed_std"] = (
    df[wheel_speed_columns].std(axis=1)
)


# ============================================================
# REMOVE UNWANTED COLUMNS
# ============================================================

DROP_COLUMNS = [
    "vehicle_id",
    "timestamp",
    "vehicle_year",

    "engine_failure_imminent",
    "brake_issue_imminent",
    "battery_issue_imminent",

    "failure_date",
    "failure_type",

    "gps_latitude",
    "gps_longitude",

    "failure"
]


X = df.drop(
    columns=DROP_COLUMNS,
    errors="ignore"
)

y = df["failure"]


# ============================================================
# KEEP NUMERIC FEATURES
# ============================================================

X = X.select_dtypes(
    include=[
        "int64",
        "float64",
        "int32",
        "float32"
    ]
)


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

constant_columns = [
    column
    for column in X.columns
    if X[column].nunique() <= 1
]

if constant_columns:

    print("\nRemoving constant features:")

    for column in constant_columns:
        print("-", column)

    X = X.drop(
        columns=constant_columns
    )


# ============================================================
# HANDLE INFINITE / MISSING VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median(numeric_only=True)
)


# ============================================================
# PRINT FEATURES
# ============================================================

print("\nFinal feature shape:", X.shape)

print("\nFinal features:")

for i, feature in enumerate(
    X.columns,
    start=1
):
    print(f"{i:2d}. {feature}")


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")

print(y.value_counts())

print("\nTarget percentage:")

print(
    y.value_counts(normalize=True) * 100
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\nCreating train/validation/test split...")

# 80% train+validation, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

# From remaining 80%:
# 20% becomes validation
# final = 64% train, 16% validation, 20% test

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.20,
    stratify=y_train_val,
    random_state=RANDOM_STATE
)


print("\nDataset split:")

print("Training   :", X_train.shape)
print("Validation :", X_val.shape)
print("Testing    :", X_test.shape)


# ============================================================
# CLASS IMBALANCE
# ============================================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = (
    negative_count /
    positive_count
)

print("\nClass imbalance:")
print("Negative:", negative_count)
print("Positive:", positive_count)

print(
    "Scale Pos Weight:",
    round(scale_pos_weight, 4)
)


# ============================================================
# DEFINE MODELS
# ============================================================

models = {

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),


    # --------------------------------------------------------
    # LIGHTGBM
    # --------------------------------------------------------

    "LightGBM": LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=3.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    ),


    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    "XGBoost": XGBClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.5,
        reg_alpha=0.1,
        reg_lambda=3.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=50
    )
}


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

results = []


for name, model in models.items():

    print("\n")
    print("=" * 70)
    print(f"TRAINING: {name}")
    print("=" * 70)

    # --------------------------------------------------------
    # XGBoost uses validation set for early stopping
    # --------------------------------------------------------

    if name == "XGBoost":

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_val, y_val)
            ],
            verbose=False
        )

    else:

        model.fit(
            X_train,
            y_train
        )

    # --------------------------------------------------------
    # Prediction probability
    # --------------------------------------------------------

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Classification threshold
    # --------------------------------------------------------

    threshold = 0.50

    y_prediction = (
        y_probability >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_prediction
    )

    precision = precision_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_prediction,
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

    cm = confusion_matrix(
        y_test,
        y_prediction
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nResults:")

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": roc_auc,

        "PR-AUC": pr_auc
    })


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    file_name = (
        name.lower()
        .replace(" ", "_")
        + "_vehicle_failure.pkl"
    )

    model_path = os.path.join(
        MODEL_DIR,
        file_name
    )

    model_data = {

        "model": model,

        "features": list(
            X_train.columns
        ),

        "metrics": {

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "roc_auc": roc_auc,

            "pr_auc": pr_auc
        },

        "threshold": threshold,

        "scale_pos_weight": scale_pos_weight
    }

    joblib.dump(
        model_data,
        model_path
    )

    print(
        f"\nModel saved: {model_path}"
    )


# ============================================================
# COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE COMPARISON RESULTS
# ============================================================

comparison_path = os.path.join(
    MODEL_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

print(
    f"\nComparison saved to: {comparison_path}"
)

print("\nTraining completed successfully.")