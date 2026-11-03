import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

df = pd.read_csv("data/fleet_data.csv")

# Select numerical telemetry features
features = ["temperature_c", "battery_pct", "signal_rssi", "last_comm_minutes_ago"]

# Isolation Forest flags outliers (-1 = anomaly, 1 = normal)
model = IsolationForest(contamination=0.08, random_state=42)
df["anomaly_score"] = model.fit_predict(df[features])

# Save trained model logic
joblib.dump(model, "anomaly_model.pkl")
print("Model trained and saved to anomaly_model.pkl!")