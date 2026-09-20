import random


def assign_failure_scenarios(vehicles):

    vehicle_ids = list(vehicles.keys())

    # 6 vehicles will have developing failures
    failure_vehicles = random.sample(
        vehicle_ids,
        6
    )

    failure_types = [
        "engine",
        "engine",
        "battery",
        "battery",
        "brake",
        "brake"
    ]

    random.shuffle(failure_types)

    for vehicle_id, failure_type in zip(
        failure_vehicles,
        failure_types
    ):

        vehicles[vehicle_id].assign_failure_scenario(
            failure_type
        )

    return vehicles