from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, when


def create_features(df: DataFrame) -> DataFrame:

    # ==================================================
    # 1. KM PER YEAR
    # ==================================================

    df = df.withColumn(
        "km_per_year",
        col("odometer_km") /
        when(
            (lit(2026) - col("vehicle_year")) < 1,
            lit(1)
        ).otherwise(
            lit(2026) - col("vehicle_year")
        )
    )

    # ==================================================
    # 2. ENGINE HOURS PER KM
    # ==================================================

    df = df.withColumn(
        "hours_per_km",
        col("engine_hours") /
        (col("odometer_km") + lit(1))
    )

    # ==================================================
    # 3. TEMPERATURE RATIO
    # ==================================================

    df = df.withColumn(
        "temp_ratio",
        col("engine_temp_c") /
        (col("coolant_temp_c") + lit(1))
    )

    # ==================================================
    # 4. BATTERY POWER DRAW
    # ==================================================

    df = df.withColumn(
        "power_draw_w",
        col("battery_voltage_v") *
        col("battery_current_a")
    )

    # ==================================================
    # 5. WHEEL SPEED VARIATION
    # ==================================================

    mean_wheel_speed = (
        col("wheel_speed_fl_kph") +
        col("wheel_speed_fr_kph") +
        col("wheel_speed_rl_kph") +
        col("wheel_speed_rr_kph")
    ) / lit(4)

    df = df.withColumn(
        "wheel_speed_std",
        (
            (
                col("wheel_speed_fl_kph") -
                mean_wheel_speed
            ) ** 2 +
            (
                col("wheel_speed_fr_kph") -
                mean_wheel_speed
            ) ** 2 +
            (
                col("wheel_speed_rl_kph") -
                mean_wheel_speed
            ) ** 2 +
            (
                col("wheel_speed_rr_kph") -
                mean_wheel_speed
            ) ** 2
        ) / lit(4)
    )

    # ==================================================
    # SELECT 35 MODEL FEATURES
    # ==================================================

    feature_columns = [
        "odometer_km",
        "engine_hours",
        "vehicle_speed_kph",
        "engine_temp_c",
        "engine_rpm",
        "oil_pressure_psi",
        "coolant_temp_c",
        "fuel_level_percent",
        "fuel_consumption_lph",
        "engine_load_percent",
        "throttle_position_percent",
        "air_flow_rate_gps",
        "exhaust_gas_temp_c",
        "engine_vibration",
        "brake_pad_wear_mm",
        "brake_fluid_level",
        "brake_temp_c",
        "brake_pedal_position",
        "wheel_speed_fl_kph",
        "wheel_speed_fr_kph",
        "wheel_speed_rl_kph",
        "wheel_speed_rr_kph",
        "battery_voltage_v",
        "battery_current_a",
        "battery_temp_c",
        "alternator_output_v",
        "battery_charge_percent",
        "battery_health_percent",
        "ambient_temp_c",
        "humidity_percent",
        "km_per_year",
        "hours_per_km",
        "temp_ratio",
        "power_draw_w",
        "wheel_speed_std"
    ]

    return df.select(*feature_columns)