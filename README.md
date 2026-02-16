# Adaptive Battery System for Smart Homes

AI-powered battery management prototype using a Python backend and a React frontend.

## Stack
- Backend: FastAPI (`backend/main.py`)
- Frontend: React + Vite + Tailwind (`frontend/`)
- AI/Control modules: simulator, RNN, fuzzy governor, GA optimizer (`src/`)

## Project Structure
- `backend/main.py`: FastAPI API for system status and control
- `frontend/`: React dashboard application
- `src/battery_sim.py`: battery simulation
- `src/rnn_prognostic.py`: model training script
- `src/fuzzy_governor.py`: fuzzy safety governor
- `src/ga_optimizer.py`: schedule optimizer
- `data/`: generated battery data
- `models/`: trained model artifacts

## Run (React + FastAPI)

### 1. Start backend
```bash
pip install fastapi uvicorn tensorflow numpy
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend runs on Vite dev server (usually `http://localhost:5173`) and calls the backend at `http://localhost:8000`.

## Training flow (optional)
```bash
python src/battery_sim.py
python src/rnn_prognostic.py
```

This generates `data/battery_history.csv` and updates `models/fatigue_model.h5`.

## Evaluation Workflow
Run structured scenarios and compute KPI metrics:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_test_scenarios.ps1 -Repeats 5
python .\scripts\evaluate_scenarios.py
```

Artifacts are written under `data/eval/<run_id>/`:
- `scenario_log.csv`
- `metrics_per_experiment.csv`
- `metrics_summary.csv`
- `significance_vs_baseline.csv`
- `evaluation_report.md`

## Persistent Experiment Datastore
Runtime telemetry/control data is also persisted to SQLite:
- default DB path: `data/ops/battery_ops.db`
- tables: `experiments`, `telemetry`, `control_decisions`, `dispatch_commands`

Experiment lifecycle endpoints:
- `POST /api/experiments/start`
- `POST /api/experiments/end`
- `GET /api/experiments/current`

Control reliability endpoints:
- `POST /api/control/ack` (device ACK/NACK ingestion)
- `GET /api/control/metrics` (success/failure/latency/stale ratios)

## SOH Validation and Calibration
Evaluate SOH model on time-aware hold-out split:

```powershell
python .\scripts\validate_soh_model.py --data .\data\battery_history.csv --model .\models\fatigue_model.h5
```

Outputs are written under `data/eval/soh_validation_<timestamp>/`:
- `soh_metrics.csv` (MAE/RMSE/MAPE/sMAPE for raw, calibrated, baseline, ID/OOD)
- `soh_error_drift.csv` (error drift across test horizon bins)
- `soh_test_predictions.csv`
- `soh_validation_summary.json` (domain limits + calibration params)
- `soh_validation_report.md`

<<<<<<< HEAD
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
=======
Metric definitions and architecture diagram:
- `docs/EVALUATION_AND_ARCHITECTURE.md`
>>>>>>> f4de271 (Add multi-device control, MQTT dispatch, evaluation pipeline, and SOH validation)
