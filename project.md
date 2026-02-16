# Battery Project Overview

## What This Project Does
Adaptive battery management prototype for smart homes. It combines simulation/telemetry ingestion, SOH prediction, fuzzy safety governance, and optional schedule optimization to keep battery operation safe while reducing cost and user impact.

## Tech Stack
- Backend API: FastAPI (`backend/main.py`)
- Persistence: SQLite (`backend/persistence.py`, default `data/ops/battery_ops.db`)
- AI/Control: TensorFlow LSTM SOH model, scikit-fuzzy governor, GA-based scheduling (`src/`)
- Frontend: React + Vite dashboard (`frontend/`)
- Evaluation: PowerShell scenario runner + Python KPI/statistics scripts (`scripts/`)

## System Architecture
```mermaid
flowchart LR
    UI[React Dashboard\nfrontend/src/App.jsx] --> API[FastAPI Backend\nbackend/main.py]
    Devices[Device Telemetry\n/api/telemetry + /api/devices/*] --> API

    API --> Control[Control Snapshot Engine\n_predict_soh + fuzzy limit + mode fallback]
    API --> Dispatch[Dispatch Engine\nSafety override + optimization]
    Dispatch --> Adapter[Command Adapter\nmock | mqtt]
    Adapter --> Broker[MQTT Broker / Device Endpoints]

    API --> DB[(SQLite\ndata/ops/battery_ops.db)]
    Control --> DB
    Dispatch --> DB

    API --> Model[(SOH Model\nmodels/fatigue_model.h5)]
    API --> Gov[Fuzzy Governor\nsrc/fuzzy_governor.py]
    API --> GA[GA Optimizer\nsrc/ga_optimizer.py]
```

## Runtime Control Sequence
```mermaid
sequenceDiagram
    participant D as Device/Simulator
    participant A as FastAPI
    participant M as SOH Model
    participant F as Fuzzy Governor
    participant O as Optimizer
    participant C as Command Adapter
    participant S as SQLite

    D->>A: POST /api/telemetry (LIVE) or SIM load
    A->>A: Build control snapshot
    A->>M: Predict SOH (if model + history available)
    M-->>A: soh_pred
    A->>F: Compute fuzzy_limit(soh,temp)
    F-->>A: limit %, status
    A->>S: log_control_decision

    alt status == PROTECTED
        A->>A: Select high-power controllable devices
        A->>C: send turn_off commands
    else optimization_enabled
        A->>O: optimize flexible start times
        O-->>A: delay_start schedule
        A->>C: send delay_start commands
    end

    A->>S: log_dispatch_command (+ ack updates when received)
    A-->>D: /api/system-status and /api/control/* responses
```

## Evaluation and Reporting Pipeline
```mermaid
flowchart TD
    R[run_test_scenarios.ps1] --> E1[Create run_id folder\ndata/eval/<run_id>/]
    E1 --> E2[Start experiments\nPOST /api/experiments/start]
    E2 --> E3[Execute scenarios\nnormal, heatwave, sudden_spike, stale_telemetry]
    E3 --> E4[Collect per-step rows\nscenario_log.csv]
    E4 --> E5[evaluate_scenarios.py]
    E5 --> A1[metrics_per_experiment.csv]
    E5 --> A2[metrics_summary.csv]
    E5 --> A3[significance_vs_baseline.csv]
    E5 --> A4[evaluation_report.md]
```

## Persistence Model
```mermaid
erDiagram
    EXPERIMENTS ||--o{ TELEMETRY : has
    EXPERIMENTS ||--o{ CONTROL_DECISIONS : has
    EXPERIMENTS ||--o{ DISPATCH_COMMANDS : has
    EXPERIMENTS ||--o{ COMMAND_ACKS : has

    EXPERIMENTS {
      text experiment_id PK
      text started_at_utc
      text ended_at_utc
      text status
      text controller_mode
      text runtime_mode
      text scenario
    }

    TELEMETRY {
      int id PK
      text experiment_id FK
      text ts_utc
      real current_a
      real temperature_c
      real soh
      text source
    }

    CONTROL_DECISIONS {
      int id PK
      text experiment_id FK
      text ts_utc
      text controller_mode
      real load_a
      real temp_c
      real soh_pred
      real fuzzy_limit_pct
      text status
    }

    DISPATCH_COMMANDS {
      int id PK
      text experiment_id FK
      text ts_utc
      text command_id
      text device_id
      text command
      text reason
      text adapter
      text status
      int dry_run
    }

    COMMAND_ACKS {
      int id PK
      text experiment_id FK
      text ts_utc
      text command_id
      text device_id
      text ack_status
      real latency_ms
    }
```

## Key API Surface
- Runtime and control:
  - `GET /api/system-status`
  - `POST /api/control/dispatch`
  - `POST /api/control/mode`
  - `POST /api/mode`
  - `POST /api/set-scenario`
- Telemetry and devices:
  - `POST /api/telemetry`
  - `POST /api/devices/telemetry`
  - `POST /api/devices/ingest-raw`
  - `GET /api/devices`
- Experiments and reliability:
  - `POST /api/experiments/start`
  - `POST /api/experiments/end`
  - `GET /api/experiments/current`
  - `POST /api/control/ack`
  - `GET /api/control/metrics`

## Typical Workflows
1. Start backend:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
2. Start frontend:
```bash
cd frontend
npm install
npm run dev
```
3. Optional model generation/training:
```bash
python src/battery_sim.py
python src/rnn_prognostic.py
```
4. Scenario evaluation:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_test_scenarios.ps1 -Repeats 5
python .\scripts\evaluate_scenarios.py
```
5. SOH validation:
```bash
python .\scripts\validate_soh_model.py --data .\data\battery_history.csv --model .\models\fatigue_model.h5
```
