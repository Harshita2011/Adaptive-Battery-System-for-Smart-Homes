# 🤖 Adaptive Battery System for Smart Homes

An autonomous AI-powered battery management system that combines **Physics-Informed Neural Networks**, **Fuzzy Logic**, and **Genetic Algorithms** for intelligent energy optimization in smart homes.

## 🌟 Features

- **Physics-Informed LSTM (RNN)**: Predicts battery State of Health (SOH) using current and temperature data
- **Fuzzy Logic Governor**: Dynamically sets safe discharge power limits based on battery health and temperature
- **Genetic Algorithm Optimizer**: Schedules appliances (EV charger, AC, dishwasher) to minimize cost while respecting battery constraints
- **Real-Time Dashboard**: Interactive Streamlit UI with live metrics, trends, and automated action logs
- **Rolling Horizon Control**: Re-optimizes only when battery conditions change significantly (>5% threshold)
- **Research Report Generation**: Export AI decisions to Excel for analysis

## 📋 Project Structure

```
Battery/
├── data/
│   └── battery_history.csv          # Simulated battery data
├── models/
│   └── fatigue_model.h5             # Trained RNN model
├── notebooks/                        # Jupyter notebooks & analysis
├── src/
│   ├── battery_sim.py               # Battery simulator with thermal & fatigue models
│   ├── rnn_prognostic.py            # Physics-informed LSTM for SOH prediction
│   ├── fuzzy_governor.py            # Fuzzy logic controller for safety limits
│   ├── ga_optimizer.py              # Genetic algorithm for load scheduling
│   ├── realtime_dashboard.py        # Streamlit UI (main app)
│   ├── main.py                      # Autonomous rolling horizon loop
│   └── evaluate_performance.py      # Performance visualization
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Training Data
```bash
python src/battery_sim.py
```
This creates `data/battery_history.csv` with 2000 simulation steps.

### 3. Train the RNN Model
```bash
python src/rnn_prognostic.py
```
Trains a physics-informed LSTM and saves to `models/fatigue_model.h5`.

### 4. Run the Dashboard
```bash
streamlit run src/realtime_dashboard.py
```
Opens interactive dashboard at `http://localhost:8501`

## 🔧 Core Components

### Battery Simulator (`src/battery_sim.py`)
- **Thermal Model**: Joule heating with internal resistance
- **Discharge Model**: Capacity-based state of charge (SOC) calculation
- **Fatigue Model**: Accelerated aging when temp > 45°C

### RNN Prognostic (`src/rnn_prognostic.py`)
- **Architecture**: LSTM(64) → Dense(32) → Output(sigmoid)
- **Custom Loss**: Data loss + physics residual penalty
- **Input**: 60-minute window of [current, temp]
- **Output**: SOH prediction (0-1)

### Fuzzy Governor (`src/fuzzy_governor.py`)
- **Inputs**: Predicted SOH, Current Temperature
- **Output**: Discharge power limit (0-100%)
- **Rules**:
  - Poor SOH OR Hot temp → Low limit
  - Fair SOH AND Warm temp → Medium limit
  - Good SOH AND Cool temp → High limit

### GA Optimizer (`src/ga_optimizer.py`)
- **Genes**: Start times (0-23 hours) for 3 appliances
- **Fitness**: Minimize cost + fatigue penalty
- **Population**: 20 individuals, 100 generations
- **Mutation**: 10% of genes

## 📊 Dashboard Features

| Metric | Description |
|--------|-------------|
| Live Load | Current discharge in Amps |
| RNN Health | Predicted SOH (0-1) |
| Fuzzy Limit | Safe power limit (%) |
| Status | PROTECTED (limit < 80%) or OPTIMAL |

**Visualizations**:
- Real-time temperature trend (red line)
- Battery health evolution (blue line)
- Automated action log with timestamps

**Export**:
- Download AI decisions as Excel report
- Track system responses to changing conditions

## 🧪 How It Works

```
1. Simulate battery step (load + thermal effects)
   ↓
2. Update history (current, temp, SOH)
   ↓
3. RNN predicts SOH from 60-minute window
   ↓
4. Fuzzy logic sets safe discharge limit
   ↓
5. If limit changed >5%:
   - Run GA to reschedule appliances
   - Log decision with timestamp
   ↓
6. Display updated metrics & charts
   ↓
7. Sleep 1 second, repeat
```

## 📈 Performance Evaluation

Run analysis and visualization:
```bash
python src/evaluate_performance.py
```
Generates `notebooks/final_performance_plot.png`

## 🔬 Research Applications

- Battery longevity optimization
- Smart grid load balancing
- Predictive maintenance for EVs
- Cost-aware energy scheduling
- Adaptive thermal management

## 📦 Dependencies

- **TensorFlow 2.x**: Deep learning framework
- **scikit-fuzzy**: Fuzzy logic implementation
- **PyGAD**: Genetic algorithm library
- **Streamlit**: Web dashboard
- **Plotly**: Interactive visualizations
- **Pandas/NumPy**: Data processing

See `requirements.txt` for exact versions.

## 💡 Future Enhancements

- Multi-battery system support
- Real grid price API integration
- Battery degradation curve fitting
- Hardware deployment (Raspberry Pi)
- Cloud monitoring dashboard
- Predictive weather integration

**Last Updated**: December 2025
