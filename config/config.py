from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "historical" / "connected_vehicle_telemetry.csv"

MODEL_PATH = BASE_DIR / "models" / "xgboost_vehicle_failure.pkl"