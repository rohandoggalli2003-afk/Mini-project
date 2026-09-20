import numpy as np
import pandas as pd

TARGET_COLUMN = "failure"

DROP_COLUMNS = [
    "vehicle_id",
    "timestamp",
    "engine_failure_imminent",
    "brake_issue_imminent",
    "battery_issue_imminent",
    "failure_date",
    "failure_type",
    "gps_latitude",
    "gps_longitude",
]


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)

    # 1. Create binary target
    df[TARGET_COLUMN] = (
        (df["engine_failure_imminent"] == 1)
        | (df["brake_issue_imminent"] == 1)
        | (df["battery_issue_imminent"] == 1)
    ).astype(int)

    # 2. Remove non-feature columns
    X = df.drop(columns=DROP_COLUMNS + [TARGET_COLUMN], errors="ignore")
    y = df[TARGET_COLUMN]

    # 3. Keep numeric columns only
    X = X.select_dtypes(include=["int64", "float64", "int32", "float32"])

    # 4. Feature Engineering
    # Telemetry Stress Ratios
    if "odometer_km" in X.columns and "vehicle_year" in X.columns:
        # Assuming current operational reference year
        X["km_per_year"] = X["odometer_km"] / (2026 - X["vehicle_year"] + 1)

    if "engine_hours" in X.columns and "odometer_km" in X.columns:
        X["hours_per_km"] = X["engine_hours"] / (X["odometer_km"] + 1)

    # Thermal & Power Metrics
    if "engine_temp_c" in X.columns and "coolant_temp_c" in X.columns:
        X["temp_ratio"] = X["engine_temp_c"] / (X["coolant_temp_c"] + 1)

    if "battery_voltage_v" in X.columns and "battery_current_a" in X.columns:
        X["power_draw_w"] = X["battery_voltage_v"] * X["battery_current_a"]

    # Mechanical Anomaly Metrics (Wheel Speed Dispersion)
    wheel_cols = [
        "wheel_speed_fl_kph",
        "wheel_speed_fr_kph",
        "wheel_speed_rl_kph",
        "wheel_speed_rr_kph",
    ]
    if all(col in X.columns for col in wheel_cols):
        X["wheel_speed_std"] = X[wheel_cols].std(axis=1)

    # 5. Handle missing values (including any inf values created by division)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    return X, y