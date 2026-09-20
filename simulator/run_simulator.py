import time
import json

from telemetry_generator import TelemetryGenerator


def main():

    generator = TelemetryGenerator(
        number_of_vehicles=20
    )

    print("=" * 60)
    print("CONNECTED VEHICLE TELEMETRY SIMULATOR")
    print("=" * 60)

    print("\nVehicles:")

    for vehicle_id, vehicle in generator.vehicles.items():

        print(
            f"{vehicle_id} -> "
            f"{vehicle.failure_type or 'NORMAL'}"
        )

    print("\nStarting telemetry generation...\n")

    while True:

        telemetry = generator.generate()

        for record in telemetry:

            print(json.dumps(record))

        print("-" * 60)

        time.sleep(5)


if __name__ == "__main__":
    main()