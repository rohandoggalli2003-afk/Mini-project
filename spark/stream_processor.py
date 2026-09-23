from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json,
    col,
    lit,
    when,
    sqrt,
    pow,
    avg,
    stddev
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
    TimestampType
)


# ============================================================
# CONFIGURATION
# ============================================================

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "vehicle_telemetry"


# ============================================================
# CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("ConnectedVehicleRealTimeAnalytics")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("=" * 70)
print("CONNECTED VEHICLE - PYSPARK STREAM PROCESSOR")
print("=" * 70)


# ============================================================
# KAFKA MESSAGE SCHEMA
# ============================================================

schema = StructType([
    StructField("vehicle_id", StringType(), True),
    StructField("timestamp", StringType(), True),

    StructField("odometer_km", DoubleType(), True),
    StructField("engine_hours", DoubleType(), True),
    StructField("vehicle_speed_kph", DoubleType(), True),

    StructField("engine_temp_c", DoubleType(), True),
    StructField("engine_rpm", DoubleType(), True),
    StructField("oil_pressure_psi", DoubleType(), True),
    StructField("coolant_temp_c", DoubleType(), True),

    StructField("fuel_level_percent", DoubleType(), True),
    StructField("fuel_consumption_lph", DoubleType(), True),
    StructField("engine_load_percent", DoubleType(), True),
    StructField("throttle_position_percent", DoubleType(), True),
    StructField("air_flow_rate_gps", DoubleType(), True),
    StructField("exhaust_gas_temp_c", DoubleType(), True),

    StructField("engine_vibration", DoubleType(), True),

    StructField("brake_pad_wear_mm", DoubleType(), True),
    StructField("brake_fluid_level", DoubleType(), True),
    StructField("brake_temp_c", DoubleType(), True),
    StructField("brake_pedal_position", DoubleType(), True),

    StructField("wheel_speed_fl_kph", DoubleType(), True),
    StructField("wheel_speed_fr_kph", DoubleType(), True),
    StructField("wheel_speed_rl_kph", DoubleType(), True),
    StructField("wheel_speed_rr_kph", DoubleType(), True),

    StructField("battery_voltage_v", DoubleType(), True),
    StructField("battery_current_a", DoubleType(), True),
    StructField("battery_temp_c", DoubleType(), True),
    StructField("alternator_output_v", DoubleType(), True),
    StructField("battery_charge_percent", DoubleType(), True),
    StructField("battery_health_percent", DoubleType(), True),

    StructField("ambient_temp_c", DoubleType(), True),
    StructField("humidity_percent", DoubleType(), True),

    # Ground-truth simulator information.
    # These are NOT used as XGBoost features.
    StructField("failure_type", StringType(), True),
    StructField("state", StringType(), True)
])


# ============================================================
# READ DATA FROM KAFKA
# ============================================================

raw_stream = (
    spark.readStream
    .format("kafka")
    .option(
        "kafka.bootstrap.servers",
        KAFKA_BOOTSTRAP_SERVERS
    )
    .option(
        "subscribe",
        KAFKA_TOPIC
    )
    .option(
        "startingOffsets",
        "latest"
    )
    .load()
)


# ============================================================
# CONVERT KAFKA VALUE FROM BINARY TO JSON
# ============================================================

telemetry = (
    raw_stream
    .selectExpr("CAST(value AS STRING) AS json_value")
    .select(
        from_json(
            col("json_value"),
            schema
        ).alias("data")
    )
    .select("data.*")
)


# ============================================================
# REAL-TIME FEATURE ENGINEERING
# ============================================================

features = (
    telemetry

    # --------------------------------------------------------
    # km_per_year
    # --------------------------------------------------------
    .withColumn(
        "km_per_year",
        col("odometer_km") /
        lit(1.0)
    )

    # --------------------------------------------------------
    # hours_per_km
    # --------------------------------------------------------
    .withColumn(
        "hours_per_km",
        col("engine_hours") /
        (col("odometer_km") + lit(1.0))
    )

    # --------------------------------------------------------
    # temp_ratio
    # --------------------------------------------------------
    .withColumn(
        "temp_ratio",
        col("engine_temp_c") /
        (col("coolant_temp_c") + lit(1.0))
    )

    # --------------------------------------------------------
    # power_draw_w
    # --------------------------------------------------------
    .withColumn(
        "power_draw_w",
        col("battery_voltage_v") *
        col("battery_current_a")
    )

    # --------------------------------------------------------
    # wheel_speed_std
    # --------------------------------------------------------
    .withColumn(
        "wheel_speed_std",
        sqrt(
            (
                pow(col("wheel_speed_fl_kph") -
                    (
                        col("wheel_speed_fl_kph") +
                        col("wheel_speed_fr_kph") +
                        col("wheel_speed_rl_kph") +
                        col("wheel_speed_rr_kph")
                    ) / 4, 2)

                +

                pow(col("wheel_speed_fr_kph") -
                    (
                        col("wheel_speed_fl_kph") +
                        col("wheel_speed_fr_kph") +
                        col("wheel_speed_rl_kph") +
                        col("wheel_speed_rr_kph")
                    ) / 4, 2)

                +

                pow(col("wheel_speed_rl_kph") -
                    (
                        col("wheel_speed_fl_kph") +
                        col("wheel_speed_fr_kph") +
                        col("wheel_speed_rl_kph") +
                        col("wheel_speed_rr_kph")
                    ) / 4, 2)

                +

                pow(col("wheel_speed_rr_kph") -
                    (
                        col("wheel_speed_fl_kph") +
                        col("wheel_speed_fr_kph") +
                        col("wheel_speed_rl_kph") +
                        col("wheel_speed_rr_kph")
                    ) / 4, 2)
            ) / 3
        )
    )
)


# ============================================================
# SELECT 35 XGBOOST FEATURES
# ============================================================

model_features = [
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


# ============================================================
# FINAL STREAM
# ============================================================

final_stream = features.select(
    "vehicle_id",
    "timestamp",
    "failure_type",
    "state",
    *model_features
)


# ============================================================
# DISPLAY REAL-TIME DATA
# ============================================================

query = (
    final_stream
    .writeStream
    .outputMode("append")
    .format("console")
    .option("truncate", "false")
    .option("numRows", 20)
    .option("checkpointLocation", "data/live/spark_checkpoint")
    .start()
)


print("\nPySpark streaming started...")
print("Kafka Topic:", KAFKA_TOPIC)
print("Kafka Broker:", KAFKA_BOOTSTRAP_SERVERS)
print("Model Features:", len(model_features))
print("\nWaiting for telemetry...\n")


# ============================================================
# KEEP STREAM RUNNING
# ============================================================

query.awaitTermination()