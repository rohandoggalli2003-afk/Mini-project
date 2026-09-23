from .predict_failure import VehicleFailurePredictor


predictor = VehicleFailurePredictor()

sample_record = {
    "odometer_km": 50000,
    "engine_hours": 3000,
    "vehicle_speed_kph": 40,
    "engine_temp_c": 90,
    "engine_rpm": 1800,
    "oil_pressure_psi": 40,
    "coolant_temp_c": 90,
    "fuel_level_percent": 60,
    "fuel_consumption_lph": 3,
    "engine_load_percent": 25,
    "throttle_position_percent": 20,
    "air_flow_rate_gps": 6,
    "exhaust_gas_temp_c": 350,
    "engine_vibration": 0.5,
    "brake_pad_wear_mm": 5,
    "brake_fluid_level": 85,
    "brake_temp_c": 45,
    "brake_pedal_position": 2,
    "wheel_speed_fl_kph": 40,
    "wheel_speed_fr_kph": 40,
    "wheel_speed_rl_kph": 40,
    "wheel_speed_rr_kph": 40,
    "battery_voltage_v": 12.5,
    "battery_current_a": 10,
    "battery_temp_c": 25,
    "alternator_output_v": 14,
    "battery_charge_percent": 80,
    "battery_health_percent": 90,
    "ambient_temp_c": 30,
    "humidity_percent": 50,
    "km_per_year": 10000,
    "hours_per_km": 0.06,
    "temp_ratio": 1.0,
    "power_draw_w": 125,
    "wheel_speed_std": 0.1
}

result = predictor.predict(sample_record)

print("\nPrediction Result")
print("-----------------")
print(f"Failure Probability : {result['failure_probability']:.4f}")
print(f"Prediction          : {result['prediction']}")
print(f"Status              : {result['status']}")