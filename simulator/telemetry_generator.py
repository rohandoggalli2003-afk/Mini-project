from .vehicle_state import VehicleState
from .failure_scenarios import assign_failure_scenarios


class TelemetryGenerator:

    def __init__(self, number_of_vehicles=20):

        self.vehicles = {}

        for i in range(1, number_of_vehicles + 1):

            vehicle_id = f"V{i:03d}"

            self.vehicles[vehicle_id] = VehicleState(
                vehicle_id
            )

        self.vehicles = assign_failure_scenarios(
            self.vehicles
        )

    def generate(self):

        telemetry = []

        for vehicle in self.vehicles.values():

            record = vehicle.update()

            telemetry.append(record)

        return telemetry