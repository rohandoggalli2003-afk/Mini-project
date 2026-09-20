import json
import time

from kafka import KafkaProducer

from simulator.telemetry_generator import TelemetryGenerator


KAFKA_SERVER = "localhost:9092"
TOPIC = "vehicle_telemetry"


def create_producer():

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        value_serializer=lambda value:
            json.dumps(value).encode("utf-8")
    )

    return producer


def main():

    producer = create_producer()

    generator = TelemetryGenerator(
        number_of_vehicles=20
    )

    print("=" * 60)
    print("KAFKA VEHICLE TELEMETRY PRODUCER")
    print("=" * 60)

    print("\nSending telemetry to Kafka...")
    print(f"Broker : {KAFKA_SERVER}")
    print(f"Topic  : {TOPIC}")

    try:

        while True:

            telemetry = generator.generate()

            for record in telemetry:

                producer.send(
                    TOPIC,
                    value=record
                )

                print(
                    f"Sent: "
                    f"{record['vehicle_id']} | "
                    f"State: {record['vehicle_state']} | "
                    f"Failure: {record['failure_type']}"
                )

            producer.flush()

            print("-" * 60)

            time.sleep(5)

    except KeyboardInterrupt:

        print("\nProducer stopped.")

    finally:

        producer.close()


if __name__ == "__main__":
    main()