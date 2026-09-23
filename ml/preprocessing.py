import numpy as np
import pandas as pd

# ==================================================
# CONFIGURATION
# ==================================================

TARGET_COLUMN = "failure"

DROP_COLUMNS = [
    "vehicle_id",
    "timestamp",

    # Original vehicle year is removed
    "vehicle_year",

    # Columns used to create the target
    "engine_failure_imminent",
    "brake_issue_imminent",
    "battery_issue_imminent",

    # Target-related information
    "failure_date",
    "failure_type",

    # Location information
    "gps_latitude",
    "gps_longitude",
]

# ==================================================
# LOAD AND PREPARE DATA
# ==================================================

def load_and_prepare_data(file_path):

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    df = pd.read_csv(file_path)

    print("\nDataset loaded successfully.")
    print("Original dataset shape:", df.shape)

    # --------------------------------------------------
    # Create target variable
    # --------------------------------------------------

    df[TARGET_COLUMN] = (
        (df["engine_failure_imminent"] == 1)
        | (df["brake_issue_imminent"] == 1)
        | (df["battery_issue_imminent"] == 1)
    ).astype(int)

    # ==================================================
    # FEATURE ENGINEERING
    # ==================================================

    print("\nCreating engineered features...")

    # --------------------------------------------------
    # 1. KM per year
    # --------------------------------------------------

    vehicle_age = 2026 - df["vehicle_year"]

    # Avoid division by zero
    vehicle_age = vehicle_age.clip(lower=1)

    df["km_per_year"] = (
        df["odometer_km"] / vehicle_age
    )

    # --------------------------------------------------
    # 2. Engine hours per kilometer
    # --------------------------------------------------

    df["hours_per_km"] = (
        df["engine_hours"] /
        (df["odometer_km"] + 1)
    )

    # --------------------------------------------------
    # 3. Engine temperature / coolant temperature
    # --------------------------------------------------

    df["temp_ratio"] = (
        df["engine_temp_c"] /
        (df["coolant_temp_c"] + 1)
    )

    # --------------------------------------------------
    # 4. Battery power draw
    # --------------------------------------------------

    df["power_draw_w"] = (
        df["battery_voltage_v"] *
        df["battery_current_a"]
    )

    # --------------------------------------------------
    # 5. Wheel speed variation
    # --------------------------------------------------

    wheel_speed_columns = [
        "wheel_speed_fl_kph",
        "wheel_speed_fr_kph",
        "wheel_speed_rl_kph",
        "wheel_speed_rr_kph"
    ]

    df["wheel_speed_std"] = (
        df[wheel_speed_columns].std(axis=1)
    )

    # --------------------------------------------------
    # Select features
    # --------------------------------------------------

    X = df.drop(
        columns=DROP_COLUMNS + [TARGET_COLUMN],
        errors="ignore"
    )

    y = df[TARGET_COLUMN]

    # --------------------------------------------------
    # Keep numeric columns only
    # --------------------------------------------------

    X = X.select_dtypes(
        include=[
            "int64",
            "float64",
            "int32",
            "float32"
        ]
    )

    # --------------------------------------------------
    # Remove constant columns
    # --------------------------------------------------

    constant_columns = [
        column
        for column in X.columns
        if X[column].nunique() <= 1
    ]

    if constant_columns:

        print("\nConstant features removed:")

        for column in constant_columns:
            print("-", column)

        X = X.drop(
            columns=constant_columns
        )

    # --------------------------------------------------
    # Handle infinite values
    # --------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------
    # Handle missing values
    # --------------------------------------------------

    X = X.fillna(
        X.median(numeric_only=True)
    )

    # ==================================================
    # DISPLAY INFORMATION
    # ==================================================

    print("\nFinal feature shape:", X.shape)

    print("\nFinal features:")

    for i, feature in enumerate(
        X.columns,
        start=1
    ):
        print(
            f"{i:2d}. {feature}"
        )

    print("\nTarget distribution:")

    print(
        y.value_counts()
    )

    print("\nTarget proportions:")

    print(
        y.value_counts(
            normalize=True
        )
    )

    # --------------------------------------------------
    # Return data
    # --------------------------------------------------

    return X, y