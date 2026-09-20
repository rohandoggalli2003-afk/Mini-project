import random
from datetime import datetime


class VehicleState:

    def __init__(self, vehicle_id):

        self.vehicle_id = vehicle_id

        # Vehicle information
        self.vehicle_year = random.randint(2018, 2024)

        # Usage
        self.odometer_km = random.uniform(20000, 150000)
        self.engine_hours = random.uniform(1000, 8000)

        # Engine
        self.vehicle_speed_kph = 0.0
        self.engine_temp_c = random.uniform(85, 95)
        self.engine_rpm = 0.0
        self.oil_pressure_psi = random.uniform(35, 50)
        self.coolant_temp_c = random.uniform(85, 95)

        # Fuel
        self.fuel_level_percent = random.uniform(30, 100)
        self.fuel_consumption_lph = 0.0

        # Engine load
        self.engine_load_percent = 0.0
        self.throttle_position_percent = 0.0
        self.air_flow_rate_gps = 0.0
        self.exhaust_gas_temp_c = random.uniform(300, 500)

        # Vibration
        self.engine_vibration = random.uniform(0.2, 1.0)

        # Brakes
        self.brake_pad_wear_mm = random.uniform(2, 8)
        self.brake_fluid_level = random.uniform(70, 100)
        self.brake_temp_c = random.uniform(30, 60)
        self.abs_fault_indicator = 0
        self.brake_pedal_position = 0.0

        # Wheels
        self.wheel_speed_fl_kph = 0.0
        self.wheel_speed_fr_kph = 0.0
        self.wheel_speed_rl_kph = 0.0
        self.wheel_speed_rr_kph = 0.0

        # Battery
        self.battery_voltage_v = random.uniform(12.4, 12.8)
        self.battery_current_a = random.uniform(5, 20)
        self.battery_temp_c = random.uniform(20, 35)
        self.alternator_output_v = random.uniform(13.5, 14.5)
        self.battery_charge_percent = random.uniform(70, 100)
        self.battery_health_percent = random.uniform(75, 100)

        # Environment
        self.ambient_temp_c = random.uniform(20, 35)
        self.humidity_percent = random.uniform(40, 80)

        # Internal simulator state
        self.failure_type = None
        self.failure_progress = 0.0
        self.vehicle_state = "NORMAL"

    def assign_failure_scenario(self, failure_type=None):

        if failure_type is None:
            failure_type = random.choice([
                None,
                None,
                None,
                "engine",
                "battery",
                "brake"
            ])

        self.failure_type = failure_type

    def update_usage(self):

        # Vehicle speed
        self.vehicle_speed_kph = max(
            0,
            self.vehicle_speed_kph + random.uniform(-10, 10)
        )

        self.vehicle_speed_kph = min(
            self.vehicle_speed_kph,
            120
        )

        # RPM
        self.engine_rpm = max(
            700,
            800 + self.vehicle_speed_kph * 30
            + random.uniform(-100, 100)
        )

        # Engine load
        self.engine_load_percent = min(
            100,
            max(
                10,
                self.vehicle_speed_kph * 0.6
                + random.uniform(-10, 10)
            )
        )

        # Throttle
        self.throttle_position_percent = min(
            100,
            max(
                0,
                self.engine_load_percent
                + random.uniform(-5, 5)
            )
        )

        # Fuel consumption
        self.fuel_consumption_lph = (
            1.5
            + self.engine_load_percent * 0.08
            + random.uniform(-0.5, 0.5)
        )

        # Air flow
        self.air_flow_rate_gps = (
            2
            + self.engine_load_percent * 0.25
            + random.uniform(-2, 2)
        )

        # Exhaust temperature
        self.exhaust_gas_temp_c = (
            300
            + self.engine_load_percent * 3
            + random.uniform(-20, 20)
        )

        # Wheel speeds
        self.wheel_speed_fl_kph = max(
            0,
            self.vehicle_speed_kph + random.uniform(-1, 1)
        )

        self.wheel_speed_fr_kph = max(
            0,
            self.vehicle_speed_kph + random.uniform(-1, 1)
        )

        self.wheel_speed_rl_kph = max(
            0,
            self.vehicle_speed_kph + random.uniform(-1, 1)
        )

        self.wheel_speed_rr_kph = max(
            0,
            self.vehicle_speed_kph + random.uniform(-1, 1)
        )

        # Brake pedal
        self.brake_pedal_position = random.uniform(0, 5)

        # Usage
        self.odometer_km += (
            self.vehicle_speed_kph / 3600
        )

        self.engine_hours += 1 / 3600

    def apply_failure_scenario(self):

        if self.failure_type is None:

            self.vehicle_state = "NORMAL"

            return

        # Gradual degradation
        self.failure_progress += 0.01

        if self.failure_progress < 0.30:

            self.vehicle_state = "WARNING"

        elif self.failure_progress < 0.70:

            self.vehicle_state = "CRITICAL"

        else:

            self.vehicle_state = "FAILURE"

        # Engine failure
        if self.failure_type == "engine":

            self.engine_temp_c += 0.15
            self.coolant_temp_c += 0.12

            self.oil_pressure_psi -= 0.05
            self.engine_vibration += 0.01

            self.engine_temp_c = min(
                self.engine_temp_c,
                130
            )

            self.oil_pressure_psi = max(
                self.oil_pressure_psi,
                15
            )

        # Battery failure
        elif self.failure_type == "battery":

            self.battery_health_percent -= 0.05
            self.battery_voltage_v -= 0.003
            self.battery_charge_percent -= 0.05
            self.battery_temp_c += 0.03

            self.battery_health_percent = max(
                self.battery_health_percent,
                20
            )

            self.battery_voltage_v = max(
                self.battery_voltage_v,
                10.5
            )

        # Brake failure
        elif self.failure_type == "brake":

            self.brake_pad_wear_mm += 0.01
            self.brake_temp_c += 0.15
            self.brake_fluid_level -= 0.03

            if self.failure_progress > 0.5:

                self.abs_fault_indicator = 1

            self.brake_pad_wear_mm = min(
                self.brake_pad_wear_mm,
                12
            )

            self.brake_fluid_level = max(
                self.brake_fluid_level,
                20
            )

    def update(self):

        self.update_usage()

        self.apply_failure_scenario()

        # Natural sensor noise
        self.engine_temp_c += random.uniform(-0.3, 0.3)

        self.engine_vibration += random.uniform(
            -0.02,
            0.02
        )

        self.battery_voltage_v += random.uniform(
            -0.02,
            0.02
        )

        return self.to_dict()

    def to_dict(self):

        return {

            "timestamp": datetime.now().isoformat(),

            "vehicle_id": self.vehicle_id,

            "vehicle_year": self.vehicle_year,
            "odometer_km": round(self.odometer_km, 2),
            "engine_hours": round(self.engine_hours, 2),

            "vehicle_speed_kph": round(
                self.vehicle_speed_kph, 2
            ),

            "engine_temp_c": round(
                self.engine_temp_c, 2
            ),

            "engine_rpm": round(
                self.engine_rpm, 2
            ),

            "oil_pressure_psi": round(
                self.oil_pressure_psi, 2
            ),

            "coolant_temp_c": round(
                self.coolant_temp_c, 2
            ),

            "fuel_level_percent": round(
                self.fuel_level_percent, 2
            ),

            "fuel_consumption_lph": round(
                self.fuel_consumption_lph, 2
            ),

            "engine_load_percent": round(
                self.engine_load_percent, 2
            ),

            "throttle_position_percent": round(
                self.throttle_position_percent, 2
            ),

            "air_flow_rate_gps": round(
                self.air_flow_rate_gps, 2
            ),

            "exhaust_gas_temp_c": round(
                self.exhaust_gas_temp_c, 2
            ),

            "engine_vibration": round(
                self.engine_vibration,
                3
            ),

            "brake_pad_wear_mm": round(
                self.brake_pad_wear_mm,
                2
            ),

            "brake_fluid_level": round(
                self.brake_fluid_level,
                2
            ),

            "brake_temp_c": round(
                self.brake_temp_c,
                2
            ),

            "abs_fault_indicator":
                self.abs_fault_indicator,

            "brake_pedal_position": round(
                self.brake_pedal_position,
                2
            ),

            "wheel_speed_fl_kph": round(
                self.wheel_speed_fl_kph,
                2
            ),

            "wheel_speed_fr_kph": round(
                self.wheel_speed_fr_kph,
                2
            ),

            "wheel_speed_rl_kph": round(
                self.wheel_speed_rl_kph,
                2
            ),

            "wheel_speed_rr_kph": round(
                self.wheel_speed_rr_kph,
                2
            ),

            "battery_voltage_v": round(
                self.battery_voltage_v,
                3
            ),

            "battery_current_a": round(
                self.battery_current_a,
                2
            ),

            "battery_temp_c": round(
                self.battery_temp_c,
                2
            ),

            "alternator_output_v": round(
                self.alternator_output_v,
                2
            ),

            "battery_charge_percent": round(
                self.battery_charge_percent,
                2
            ),

            "battery_health_percent": round(
                self.battery_health_percent,
                2
            ),

            "ambient_temp_c": round(
                self.ambient_temp_c,
                2
            ),

            "humidity_percent": round(
                self.humidity_percent,
                2
            ),

            # Ground truth — NOT an XGBoost input
            "vehicle_state": self.vehicle_state,
            "failure_type": self.failure_type
        }