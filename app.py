import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="NetPulse AI", page_icon="📡", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #07111f 0%, #101d32 100%); color: #eef6ff;}
[data-testid="stMetric"] {background: rgba(25,45,70,.65); padding: 14px; border-radius: 12px; border: 1px solid #29415f;}
div[data-testid="stAlert"] {border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("📡 NetPulse AI")
st.subheader("AI-Based Network Anomaly Detection & Early-Warning System")
st.caption("Educational prototype • Simulated network data • Not connected to a live network")

with st.sidebar:
    st.header("⚙️ Demo controls")
    scenario = st.selectbox(
        "Choose a network scenario",
        ["Normal traffic", "Traffic spike", "High packet loss", "High latency", "Mixed network issue"]
    )
    n_rows = st.slider("Number of traffic samples", 100, 600, 300, 50)
    contamination = st.slider("Expected anomaly proportion", 0.02, 0.20, 0.08, 0.01)
    st.info("Use the scenario selector to demonstrate how unusual traffic changes the AI result.")

rng = np.random.default_rng(42)
traffic = rng.normal(55, 12, n_rows).clip(5, 100)
latency = rng.normal(35, 9, n_rows).clip(5, 100)
packet_loss = rng.normal(0.7, 0.35, n_rows).clip(0, 2.5)
connections = rng.normal(35, 8, n_rows).clip(5, 80)

# Add a scenario-specific disturbance to the most recent samples.
k = max(10, n_rows // 8)
if scenario == "Traffic spike":
    traffic[-k:] = rng.normal(95, 3, k).clip(80, 100)
    connections[-k:] = rng.normal(75, 4, k).clip(60, 100)
elif scenario == "High packet loss":
    packet_loss[-k:] = rng.normal(8, 1.5, k).clip(5, 12)
elif scenario == "High latency":
    latency[-k:] = rng.normal(180, 25, k).clip(120, 250)
elif scenario == "Mixed network issue":
    traffic[-k:] = rng.normal(92, 4, k).clip(80, 100)
    latency[-k:] = rng.normal(150, 20, k).clip(110, 220)
    packet_loss[-k:] = rng.normal(6, 1, k).clip(4, 10)

df = pd.DataFrame({
    "Sample": np.arange(1, n_rows + 1),
    "Traffic Utilization (%)": traffic,
    "Latency (ms)": latency,
    "Packet Loss (%)": packet_loss,
    "Active Connections": connections
})

features = df[["Traffic Utilization (%)", "Latency (ms)", "Packet Loss (%)", "Active Connections"]]
model = IsolationForest(contamination=contamination, random_state=42, n_estimators=150)
df["AI Result"] = model.fit_predict(features)
df["Anomaly"] = df["AI Result"].eq(-1)
anomaly_count = int(df["Anomaly"].sum())
anomaly_pct = 100 * anomaly_count / len(df)
mean_latency = float(df["Latency (ms)"].mean())
mean_loss = float(df["Packet Loss (%)"].mean())
mean_traffic = float(df["Traffic Utilization (%)"].mean())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Samples analysed", f"{len(df):,}")
m2.metric("Anomalies flagged", f"{anomaly_count:,}")
m3.metric("Average latency", f"{mean_latency:.1f} ms")
m4.metric("Average traffic", f"{mean_traffic:.1f}%")

st.markdown("### 🧠 AI analysis")
if scenario == "Normal traffic":
    st.success(f"The model flagged {anomaly_count} unusual samples out of {len(df)}. Some flags are expected because anomaly detection can produce false positives; review the charts and data before drawing conclusions.")
elif scenario == "Traffic spike":
    st.warning("Traffic-spike scenario selected. The latest samples have unusually high traffic and connection counts. Check whether the flagged samples overlap this period.")
elif scenario == "High packet loss":
    st.warning("Packet-loss scenario selected. The latest samples have elevated packet loss, which can be associated with poor connection quality.")
elif scenario == "High latency":
    st.warning("High-latency scenario selected. The latest samples have elevated delay, which may affect real-time applications.")
else:
    st.error("Mixed network issue selected. Multiple measurements have changed at once. Investigate traffic, delay, and packet loss together.")

if scenario != "Normal traffic":
    st.markdown("**Suggested investigation:** check network load, connection quality, device configuration, and the affected time window. This is a simulated scenario, not proof of a real fault or attack.")

left, right = st.columns(2)
with left:
    fig1 = px.line(df, x="Sample", y="Traffic Utilization (%)", title="Network traffic utilization")
    fig1.update_layout(template="plotly_dark", height=330, margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig1, use_container_width=True)
with right:
    fig2 = px.line(df, x="Sample", y="Latency (ms)", title="Latency over samples", color_discrete_sequence=["#00c2ff"])
    fig2.update_layout(template="plotly_dark", height=330, margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig2, use_container_width=True)

fig3 = px.scatter(
    df, x="Traffic Utilization (%)", y="Packet Loss (%)",
    color=df["Anomaly"].map({False: "Normal-like", True: "AI-flagged anomaly"}),
    hover_data=["Sample", "Latency (ms)", "Active Connections"],
    title="Traffic vs packet loss — AI-flagged samples",
    color_discrete_map={"Normal-like": "#00c2ff", "AI-flagged anomaly": "#ff5c7a"}
)
fig3.update_layout(template="plotly_dark", height=380, margin=dict(l=10,r=10,t=50,b=10))
st.plotly_chart(fig3, use_container_width=True)

st.markdown("### 📋 Recent samples")
show = df.tail(20).copy()
show["AI Status"] = np.where(show["Anomaly"], "⚠️ Anomaly flagged", "✅ Normal-like")
st.dataframe(show[["Sample", "Traffic Utilization (%)", "Latency (ms)", "Packet Loss (%)", "Active Connections", "AI Status"]], use_container_width=True, hide_index=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download analysed sample data (CSV)", csv, "netpulse_ai_results.csv", "text/csv")

with st.expander("ℹ️ How the AI works"):
    st.write("""
    This prototype uses Isolation Forest, an unsupervised machine-learning algorithm.
    It learns the distribution of the selected traffic measurements and flags samples
    that appear unusual relative to the rest of the dataset. It does not automatically
    prove that an attack occurred, and its output depends on the data and settings.
    The scenarios in this first version are simulated for safe demonstration.
    """)

st.caption("NetPulse AI • Student project prototype • Validate alerts before acting on them.")
