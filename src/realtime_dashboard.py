import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
import tensorflow as tf
import numpy as np
import io
from battery_sim import BatterySimulator
from rnn_prognostic import PINN_LSTM, physics_informed_loss
from fuzzy_governor import run_governor_test
from ga_optimizer import ga_instance

# 1. Load AI Models
model = tf.keras.models.load_model('models/fatigue_model.h5', 
                                  custom_objects={'physics_informed_loss': physics_informed_loss})

st.set_page_config(page_title="AI Battery Digital Twin", layout="wide")
st.title("🤖 Autonomous Energy Management System")

# 2. Session State for persistent logging
if 'sim' not in st.session_state:
    st.session_state.sim = BatterySimulator()
    st.session_state.data_history = []
    st.session_state.action_log = [] 
    st.session_state.last_limit = 100.0

# 3. Sidebar UI
st.sidebar.header("🕹️ Simulation Settings")
base_load = st.sidebar.slider("Base Load (Amps)", 0, 40, 15)
noise = st.sidebar.slider("Fluctuation Level", 0.0, 5.0, 1.5)

# --- REPORT GENERATION FEATURE ---
if st.sidebar.button("📄 Generate Research Report"):
    if st.session_state.action_log:
        log_df = pd.DataFrame(st.session_state.action_log)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            log_df.to_excel(writer, index=False, sheet_name='AI_Decisions')
        st.sidebar.download_button(
            label="Download Excel Report",
            data=buffer,
            file_name="Battery_AI_Research_Report.xlsx",
            mime="application/vnd.ms-excel"
        )
    else:
        st.sidebar.warning("No actions logged yet!")

# UI Layout
m_col = st.empty()
p_col = st.empty()
l_col = st.empty()

# 4. Continuous Autonomous Loop
while True:
    # A. Generate Fluctuating Data (t) - mimics real-world jitter
    load = max(0, base_load + np.random.normal(0, noise))
    state = st.session_state.sim.simulate_step(load)
    
    # B. Update History
    st.session_state.data_history.append({
        "Time": len(st.session_state.data_history), 
        "Current": load, 
        "Temp": state['temp'], 
        "SOH": state['soh']
    })
    df = pd.DataFrame(st.session_state.data_history[-100:])
    
    # C. AI Perception (RNN)
    if len(df) >= 60:
        window = df[['Current', 'Temp']].tail(60).values.reshape(1, 60, 2)
        pred_soh = model.predict(window, verbose=0)[0][0]
        # D. AI Governance (Fuzzy)
        limit = run_governor_test(pred_soh, state['temp'])
    else:
        pred_soh, limit = 1.0, 100.0

    # E. AUTOMATED ACTION (Rolling Horizon)
    # If the Fuzzy limit drops significantly, re-run GA automatically
    if abs(limit - st.session_state.last_limit) > 5.0:
        ga_instance.run()
        solution, _, _ = ga_instance.best_solution()
        st.session_state.last_limit = limit
        
        st.session_state.action_log.insert(0, {
            "Timestamp": time.strftime('%H:%M:%S'),
            "Trigger": f"Temp {state['temp']:.1f}°C",
            "AI_Response": f"Limit set to {limit:.1f}%",
            "GA_Schedule": f"Times: {np.round(solution, 1)}"
        })

    # 5. Refresh UI Elements
    with m_col.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Live Load", f"{load:.1f} A")
        c2.metric("RNN Health", f"{pred_soh:.4f}")
        c3.metric("Fuzzy Limit", f"{limit:.1f}%")
        c4.metric("Status", "PROTECTED" if limit < 80 else "OPTIMAL")

    with p_col.container():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Time'], y=df['Temp'], name="Temp (°C)", line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df['Time'], y=df['SOH']*100, name="Health (%)", line=dict(color='blue'), yaxis="y2"))
        fig.update_layout(height=350, yaxis2=dict(overlaying='y', side='right'), margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with l_col.container():
        st.subheader("📋 Automated AI Action Log")
        st.table(st.session_state.action_log[:5])

    time.sleep(1)