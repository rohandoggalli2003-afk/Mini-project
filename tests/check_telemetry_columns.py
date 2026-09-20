import pandas as pd

HISTORICAL = r"D:\AI-Connected-Vehicle-Analytics\data\historical\connected_vehicle_telemetry.csv"
LIVE = r"D:\AI-Connected-Vehicle-Analytics\data\live\live_vehicle_telemetry.csv"


historical_df = pd.read_csv(HISTORICAL)
live_df = pd.read_csv(LIVE)


historical_columns = set(historical_df.columns)
live_columns = set(live_df.columns)


print("=" * 60)
print("TELEMETRY COLUMN CHECK")
print("=" * 60)

print(f"\nHistorical columns : {len(historical_columns)}")
print(f"Live columns       : {len(live_columns)}")


print("\nColumns in Historical but NOT in Live:")
missing_live = historical_columns - live_columns

for column in sorted(missing_live):
    print(" -", column)


print("\nColumns in Live but NOT in Historical:")
extra_live = live_columns - historical_columns

for column in sorted(extra_live):
    print(" +", column)


if not missing_live:
    print("\n✓ All historical columns are present in live data.")

if not extra_live:
    print("✓ Live data has no unexpected columns.")