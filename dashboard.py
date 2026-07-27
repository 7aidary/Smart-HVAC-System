import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from spc_monitor import IMRMonitor
from spc_panel import SPCPanel

REFRESH_MS = 3000
DATA_FILE = Path("sensor_data.json")
SPC_STATE_FILE = Path("spc_state.json")

FIXED_1TON_POWER_W = 1200.0

FIXED_SPC = {
    "target": 23.5,
    "ucl_i": 23.9677,
    "lcl_i": 23.0323,
    "ucl_mr": 0.5745,
    "sigma_hat": 0.1559,
    "mr_bar": 0.1759,
}

st.set_page_config(layout="wide", page_title="SMART HVAC DASHBOARD")
st_autorefresh(interval=REFRESH_MS, key="hvac_refresh")


def load_sensor_data():
    if not DATA_FILE.exists():
        return {
            "timestamp": "--:--:--",
            "temperature": 0.0,
            "humidity": 0.0,
            "energy_power_w": FIXED_1TON_POWER_W,
        }

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)

        return {
            "timestamp": raw.get("timestamp", "--:--:--"),
            "temperature": float(raw.get("temperature", 0.0)),
            "humidity": float(raw.get("humidity", 0.0)),
            "energy_power_w": FIXED_1TON_POWER_W,
        }
    except Exception:
        return {
            "timestamp": "--:--:--",
            "temperature": 0.0,
            "humidity": 0.0,
            "energy_power_w": FIXED_1TON_POWER_W,
        }


def load_spc_series():
    if not SPC_STATE_FILE.exists():
        return []

    try:
        with open(SPC_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        values = data.get("values", [])
        return [float(v) for v in values]
    except Exception:
        return []


def save_spc_series(values):
    payload = {
        "values": values,
        "saved_at": str(datetime.now()),
    }

    with open(SPC_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_monitor(values, auto_calibrate=False):
    monitor = IMRMonitor(
        target=FIXED_SPC["target"],
        ucl_i=None if auto_calibrate else FIXED_SPC["ucl_i"],
        lcl_i=None if auto_calibrate else FIXED_SPC["lcl_i"],
        ucl_mr=None if auto_calibrate else FIXED_SPC["ucl_mr"],
        comfort_low=22,
        comfort_high=25,
        window=300,
        auto_calibrate=auto_calibrate,
        phase1_size=25,
    )

    for v in values:
        monitor.add(v)

    return monitor


data = load_sensor_data()

comfort = (
    22 <= data["temperature"] <= 25
    and 40 <= data["humidity"] <= 60
)

bg = "#d9f7e5" if comfort else "#ffe1e1"
fg = "#0b6b3a" if comfort else "#9b1c1c"

st.markdown(
    """
    <style>
    .main-title {
        font-size: 46px;
        font-weight: 900;
        text-align: center;
        color: #1f2937;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .comfort-box {
        border-radius: 22px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    .metric-card {
        padding: 24px;
        border-radius: 24px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: 0.3s;
        min-height: 170px;
    }

    .metric-card:hover {
        transform: scale(1.03);
    }

    .temp-card {
        background: linear-gradient(135deg, #ff7a18, #ff3d00);
    }

    .humidity-card {
        background: linear-gradient(135deg, #00c6ff, #0072ff);
    }

    .energy-card {
        background: linear-gradient(135deg, #8e2de2, #4a00e0);
    }

    .metric-icon {
        font-size: 46px;
        margin-bottom: 8px;
    }

    .metric-label {
        font-size: 19px;
        font-weight: 700;
    }

    .metric-value {
        font-size: 40px;
        font-weight: 900;
        margin-top: 10px;
    }

    .live-box {
        background: linear-gradient(135deg, #ffffff, #f3f4f6);
        border-radius: 30px;
        padding: 32px;
        text-align: center;
        border: 2px solid #e5e7eb;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin-top: 22px;
    }

    .big-temp {
        font-size: 88px;
        font-weight: 900;
        color: #ff3d00;
        margin: 10px 0;
    }

    .status-card {
        background: linear-gradient(135deg, #111827, #374151);
        color: white;
        padding: 22px;
        border-radius: 24px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        margin-bottom: 15px;
    }

    .status-card h3 {
        margin-top: 0;
        color: white;
    }

    .status-item {
        font-size: 17px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="comfort-box" style="background:{bg}; border:2px solid {fg};">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:34px; font-weight:900; color:{fg};">
                Comfort Zone: {"YES ✅" if comfort else "NO ❌"}
            </div>
            <div style="font-size:18px; color:{fg};">
                🕒 {data["timestamp"]}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🌡️ SMART HVAC DASHBOARD</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Real-time monitoring for temperature, humidity, comfort, and energy consumption</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([3, 1])

with left:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card humidity-card">
                <div class="metric-icon">💧</div>
                <div class="metric-label">Humidity</div>
                <div class="metric-value">{data['humidity']:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card energy-card">
                <div class="metric-icon">⚡</div>
                <div class="metric-label">Energy Consumption</div>
                <div class="metric-value">{data['energy_power_w']:.0f} W</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card temp-card">
                <div class="metric-icon">🌡️</div>
                <div class="metric-label">Temperature</div>
                <div class="metric-value">{data['temperature']:.2f}°C</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="live-box">
            <div style="font-size:23px; font-weight:800;">🔥 Live DHT22 Temperature</div>
            <div class="big-temp">{data['temperature']:.2f}°C</div>
            <div style="font-size:22px;">💧 Humidity: <b>{data['humidity']:.2f}%</b></div>
            <div style="font-size:22px;">⚡ Estimated Energy: <b>{data['energy_power_w']:.0f} W</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        f"""
        <div class="status-card">
            <h3>📡 System Status</h3>
            <div class="status-item">🕒 <b>Last Update</b><br>{data["timestamp"]}</div>
            <div class="status-item">🌡️ <b>Sensor</b><br>DHT22</div>
            <div class="status-item">🔌 <b>Source</b><br>COM11</div>
            <div class="status-item">🔄 <b>Refresh</b><br>{REFRESH_MS / 1000:.0f} s</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "Real-time dashboard for temperature, humidity, and energy consumption. "
    "The dashboard refreshes every 3 seconds, which satisfies the ≤30 s update requirement."
)

st.divider()
st.subheader("📊 Statistical Process Control (I-MR)")

spc_mode = st.radio(
    "SPC Mode",
    ["Fixed FDR limits", "Auto-calibrate (Phase-I first 25 samples)"],
    horizontal=True,
)

auto_calibrate = spc_mode.startswith("Auto")

series = load_spc_series()
current_temp = float(data["temperature"])
last_saved = series[-1] if series else None

if last_saved is None or abs(last_saved - current_temp) > 1e-9:
    series.append(current_temp)
    series = series[-300:]
    save_spc_series(series)

if st.button("Reset SPC History"):
    series = []
    save_spc_series(series)
    st.success("SPC history reset")

monitor = build_monitor(series, auto_calibrate=auto_calibrate)
snapshot = monitor.get_snapshot()

stats1, stats2, stats3, stats4 = st.columns(4)

stats1.metric(
    "CL / Target",
    f"{snapshot['target']:.4f}" if snapshot["target"] is not None else "--",
)

stats2.metric(
    "σ̂",
    f"{snapshot['sigma_hat']:.4f}" if snapshot["sigma_hat"] is not None else "--",
)

stats3.metric(
    "MR̄",
    f"{snapshot['mr_bar']:.4f}" if snapshot["mr_bar"] is not None else "--",
)

stats4.metric("Alarm Count", str(len(snapshot["alarms"])))

panel = SPCPanel()
panel.update(snapshot)
st.pyplot(panel.fig, use_container_width=True)

if snapshot["alarms"]:
    st.warning("⚠️ Out-of-control conditions detected")

    rows = [
        {"Index": idx, "Value": value, "Reason": reason}
        for idx, value, reason in snapshot["alarms"][-15:]
    ]

    st.dataframe(rows, use_container_width=True)
else:
    st.success("✅ No SPC alarms in current window")

with st.expander("SPC Limits"):
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "UCL_I",
        f"{snapshot['ucl_i']:.4f}" if snapshot["ucl_i"] is not None else "--",
    )

    c2.metric(
        "LCL_I",
        f"{snapshot['lcl_i']:.4f}" if snapshot["lcl_i"] is not None else "--",
    )

    c3.metric(
        "UCL_MR",
        f"{snapshot['ucl_mr']:.4f}" if snapshot["ucl_mr"] is not None else "--",
    )

    c4.metric(
        "Mode",
        "Calibrated"
        if snapshot["calibrated"]
        else f"Phase-I {snapshot['phase1_count']}/{snapshot['phase1_size']}",
    )