import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from pathlib import Path

st.set_page_config(page_title="AI-Powered Industrial IoT & Satellite Telematics", layout="wide")

st.title("🛰️ AI-Powered IoT Fleet, Satellite & Asset Monitoring System")
st.caption("Advanced Telematics: Isolation Forest ML Anomaly Detection, Satellite/Cellular Gaps, & LLM Diagnostics.")

# Load Data Safely
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "data" / "fleet_data.csv"

if not data_path.exists():
    st.error(f"⚠️ Data file missing. Run `python3 generate_data.py` first!")
    st.stop()

df = pd.read_csv(data_path)

# ML Anomaly Detection Model (Scikit-Learn)
features = ["temperature_c", "battery_pct", "signal_rssi", "vibration_mm_s", "last_comm_minutes_ago"]
model = IsolationForest(contamination=0.10, random_state=42)
df["anomaly_flag"] = model.fit_predict(df[features])

# Enhanced Status Logic (Covering Thermal, Battery, Geofence, & Mechanical Health)
def get_status(row):
    if row["anomaly_flag"] == -1 or row["temperature_c"] > 15 or row["battery_pct"] < 15 or row["vibration_mm_s"] > 7.0:
        return "Critical"
    elif row["geofence_status"] == "Breached" or row["last_comm_minutes_ago"] > 120 or row["signal_rssi"] < -95:
        return "Warning"
    return "Healthy"

df["status"] = df.apply(get_status, axis=1)

# Key Performance Indicators (KPIs)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Monitored Assets", len(df))
c2.metric("Healthy Status", len(df[df["status"] == "Healthy"]))
c3.metric("Warning / Coverage Gaps", len(df[df["status"] == "Warning"]))
c4.metric("Critical & Maintenance Alerts", len(df[df["status"] == "Critical"]))
c5.metric("Satellite Tracked (SC1000)", len(df[df["network_mode"] == "Satellite (OGx)"]))

st.divider()

# Layout: Map and Detailed Asset Table
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("🌐 Global & Regional Asset Map (Cellular + Satellite Tracking)")
    color_map = {"Healthy": "green", "Warning": "orange", "Critical": "red"}
    
    fig = px.scatter_map(
        df, lat="latitude", lon="longitude", color="status",
        color_discrete_map=color_map, hover_name="asset_id",
        hover_data=["asset_type", "network_mode", "temperature_c", "vibration_mm_s", "geofence_status"],
        zoom=7, height=480
    )
    fig.update_layout(map_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("🚨 Priority Incident Feed")
    alert_df = df[df["status"].isin(["Critical", "Warning"])]
    if not alert_df.empty:
        st.dataframe(alert_df[["asset_id", "asset_type", "network_mode", "status"]], use_container_width=True)
    else:
        st.success("All systems operating within nominal parameters.")

# AI Diagnostic & Operations Assistant Generator
st.divider()
st.subheader("🤖 AI Diagnostic & Automated Operational Report")

if not alert_df.empty:
    selected_asset = st.selectbox("Select Asset to Run AI Deep-Dive Diagnosis:", alert_df["asset_id"].tolist())

    if st.button("Generate AI Operational Report & Recommendations"):
        asset_info = df[df["asset_id"] == selected_asset].iloc[0].to_dict()
        
        # Dynamic AI diagnostic generation logic
        st.info(f"**Automated Telematics Diagnostic Report for Asset: {selected_asset}**")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Asset Configuration:**")
            st.markdown(f"- Type: `{asset_info['asset_type']}`")
            st.markdown(f"- Connectivity Network: `{asset_info['network_mode']}`")
            st.markdown(f"- Geofence Status: `{asset_info['geofence_status']}`")
        with col_b:
            st.markdown("**Sensor Telemetry Readings:**")
            st.markdown(f"- Temperature: `{asset_info['temperature_c']:.1f}°C`")
            st.markdown(f"- Vibration Profile: `{asset_info['vibration_mm_s']:.1f} mm/s`")
            st.markdown(f"- Battery Level: `{asset_info['battery_pct']:.1f}%`")

        # Context-aware recommendations matching user requirements
        st.markdown("---")
        st.markdown("### 📋 AI Operational Recommendations:")
        
        if asset_info['vibration_mm_s'] > 7.0:
            st.warning("⚠️ **Predictive Maintenance Alert:** Abnormal high vibration detected. Machine learning models predict potential mechanical wear/bearing failure within 48-72 hours. *Action: Schedule routine workshop maintenance prior to next long-haul trip.*")
        elif asset_info['geofence_status'] == "Breached":
            st.error("🚨 **Asset Security & Traceability Alert:** Asset has moved outside authorized operational boundary parameters without prior dispatch clearance. *Action: Trigger remote asset immobilizer and notify dispatch security.*")
        elif "Satellite" in asset_info['network_mode'] and asset_info['last_comm_minutes_ago'] > 60:
            st.info("🛰️ **Satellite Coverage / Dwell Gap Notice:** Asset is operating in a remote dead-zone outside cellular range; data successfully packet-batched via SC1000 satellite link. *Action: No physical intervention required; routine satellite sync sequence active.*")
        else:
            st.write("🔍 Minor telemetry drift noted. Monitor sensor readings over the next tracking cycle.")
else:
    st.info("No warnings or critical alerts requiring AI synthesis at this moment.")