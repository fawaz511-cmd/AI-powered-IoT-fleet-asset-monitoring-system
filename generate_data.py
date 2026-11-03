import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
n_assets = 100

data = {
    "asset_id": [f"TRK-{1000+i}" if i < 70 else f"SC1000-{2000+i}" for i in range(n_assets)],
    "asset_type": ["Cellular Truck" if i < 70 else "Satellite Asset (SC1000)" for i in range(n_assets)],
    "latitude": np.random.uniform(23.5, 25.5, n_assets),  # Broader UAE / Desert region
    "longitude": np.random.uniform(53.0, 56.0, n_assets),
    "temperature_c": np.random.normal(loc=4.0, scale=1.5, size=n_assets),
    "battery_pct": np.random.uniform(70, 100, n_assets),
    "signal_rssi": np.random.uniform(-105, -60, n_assets),
    "last_comm_minutes_ago": np.random.exponential(scale=15, size=n_assets),
    # New Advanced Telematics Metrics
    "vibration_mm_s": np.random.normal(loc=2.0, scale=0.5, size=n_assets), # For predictive maintenance
    "geofence_status": np.random.choice(["Inside", "Breached"], size=n_assets, p=[0.95, 0.05]),
    "network_mode": np.random.choice(["Cellular", "Satellite (OGx)"], size=n_assets, p=[0.7, 0.3])
}

df = pd.DataFrame(data)

# Inject intentional anomalies for testing
df.loc[5, "temperature_c"] = 28.5        # Severe temperature spike (Cold-chain issue)
df.loc[12, "battery_pct"] = 8.0          # Sudden battery drop
df.loc[25, "vibration_mm_s"] = 8.5       # High vibration -> Predictive maintenance warning
df.loc[40, "geofence_status"] = "Breached" # Security / Unauthorized movement
df.loc[85, "last_comm_minutes_ago"] = 300 # Remote satellite delay/gap test

file_path = DATA_DIR / "fleet_data.csv"
df.to_csv(file_path, index=False)
print(f"Success! Saved advanced telemetry to {file_path}")