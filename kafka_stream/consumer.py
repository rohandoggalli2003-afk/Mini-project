import json
import csv
import os

from kafka import KafkaConsumer


KAFKA_SERVER = "localhost:9092"
TOPIC = "vehicle_telemetry"

CSV_FILE = r"D:\AI-Connected-Vehicle-Analytics\data\live\live_vehicle_telemetry.csv"


def create_consumer():

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_SERVER,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="vehicle_analytics_group",

        value_deserializer=lambda value:
            json.loads(value.decode("utf-8"))
    )

    return consumer


def save_to_csv(record):

    file_exists = os.path.exists(CSV_FILE)

    with open(
        CSV_FILE,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=record.keys()
        )

        # Write column names only for a new file
        if not file_exists:
            writer.writeheader()

        writer.writerow(record)


def main():

    consumer = create_consumer()

    print("=" * 60)
    print("KAFKA VEHICLE TELEMETRY CONSUMER")
    print("=" * 60)

    print(f"\nBroker : {KAFKA_SERVER}")
    print(f"Topic  : {TOPIC}")
    print(f"CSV    : {CSV_FILE}")

    print("\nReceiving telemetry...\n")

    try:

        for message in consumer:

            record = message.value

            # Save Kafka data into CSV
            save_to_csv(record)

            print(
                f"Received: "
                f"{record['vehicle_id']} | "
                f"State: {record['vehicle_state']} | "
                f"Failure: {record['failure_type']}"
            )

    except KeyboardInterrupt:

        print("\nConsumer stopped.")

    finally:

        consumer.close()


if __name__ == "__main__":
    main()