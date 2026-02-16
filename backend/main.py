from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import uuid

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.persistence import (
    init_db,
    start_experiment,
    end_experiment,
    log_telemetry,
    log_control_decision,
    log_dispatch_command,
    log_command_ack,
    update_dispatch_command_status,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

# Soft-computing modules are optional at runtime; API still serves with fallbacks.
try:
    import sys

    sys.path.append(str(SRC_DIR))
    from fuzzy_governor import run_governor_test
except Exception:
    run_governor_test = None

try:
    from ga_optimizer import ga_instance
except Exception:
    ga_instance = None

try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

app = FastAPI()
init_db()

# Allow React (Vite) to communicate with Python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI Model
MODEL_PATH = ROOT_DIR / "models" / "fatigue_model.h5"
try:
    # Use compile=False if you have custom loss functions like physics_informed_loss
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
except Exception:
    model = None


class TelemetryPayload(BaseModel):
    current: float
    temperature: float
    voltage: Optional[float] = None
    soc: Optional[float] = None
    soh: Optional[float] = None
    timestamp: Optional[str] = None


class NormalizedDevicePayload(BaseModel):
    device_id: str
    device_type: Optional[str] = None
    power_w: float
    state: Literal["on", "off", "idle", "fault"]
    temp_c: Optional[float] = None
    capabilities: List[str] = Field(default_factory=list)
    priority: int = 3
    comfort_weight: float = 1.0
    duration_h: int = 1
    preferred_start_h: Optional[int] = None
    flex_start_h: Optional[int] = None
    flex_end_h: Optional[int] = None
    timestamp: Optional[str] = None


class RawDevicePayload(BaseModel):
    vendor: str
    payload: Dict[str, Any]
    device_id: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)


class DispatchRequest(BaseModel):
    dry_run: bool = True
    optimization_enabled: bool = True


class AdapterConfigPayload(BaseModel):
    adapter_type: Literal["mock", "mqtt"]
    mqtt_host: Optional[str] = None
    mqtt_port: Optional[int] = None
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    mqtt_topic_prefix: Optional[str] = None


class CommandAckPayload(BaseModel):
    command_id: str
    device_id: str
    ack_status: Literal["ack", "nack"]
    message: Optional[str] = None
    ack_ts_utc: Optional[str] = None


class ExperimentStartPayload(BaseModel):
    experiment_id: Optional[str] = None
    run_seed: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Global state for demo runtime
battery_state = {
    "load": 22.0,
    "health": 0.9992,
    "temperature": 38.5,
    "scenario": "NORMAL",
    "mode": "SIM",  # SIM or LIVE
    "active_experiment_id": None,
    "controller_mode": "rnn_fuzzy_ga",  # rule_only | fuzzy_only | rnn_fuzzy | rnn_fuzzy_ga
    "last_telemetry_at": None,
    "telemetry": {},
    "history": [],  # rolling window of dicts: {current, temp}
    "last_safe_limit": 100.0,
    "last_schedule": [0.0, 0.0, 0.0],
    "last_status": "OPTIMAL",
    "last_logged_protection_limit": None,
    "devices": {},  # normalized device telemetry by device_id
    "last_dispatch": [],
    "command_outbox": [],
    "command_ack_log": [],
    "recent_command_keys": {},
    "command_retry_max": 2,
    "command_dedupe_window_sec": 30,
    "command_adapter": {
        "type": os.getenv("COMMAND_ADAPTER_TYPE", "mock").lower(),
        "mqtt_host": os.getenv("MQTT_HOST", "localhost"),
        "mqtt_port": int(os.getenv("MQTT_PORT", "1883")),
        "mqtt_username": os.getenv("MQTT_USERNAME"),
        "mqtt_password": os.getenv("MQTT_PASSWORD"),
        "mqtt_topic_prefix": os.getenv("MQTT_TOPIC_PREFIX", "battery/commands"),
    },
    "mqtt_connected": False,
    "mqtt_error": None,
    "mqtt_client": None,
    "actions": [],
}


@app.get("/")
def home():
    return {"status": "Autonomous AI Digital Twin API is Active"}


@app.post("/api/set-scenario")
def set_scenario(name: str):
    battery_state["scenario"] = name
    return {"status": "success", "active": name}


@app.post("/api/mode")
def set_mode(name: str):
    mode = (name or "").strip().upper()
    if mode not in {"SIM", "LIVE"}:
        raise HTTPException(status_code=400, detail="Mode must be SIM or LIVE")

    battery_state["mode"] = mode
    return {"status": "success", "mode": mode}


@app.post("/api/control/mode")
def set_control_mode(name: str):
    mode = (name or "").strip().lower()
    valid = {"rule_only", "fuzzy_only", "rnn_fuzzy", "rnn_fuzzy_ga"}
    if mode not in valid:
        raise HTTPException(status_code=400, detail=f"Control mode must be one of: {sorted(valid)}")
    battery_state["controller_mode"] = mode
    return {"status": "success", "controller_mode": mode}


@app.get("/api/experiments/current")
def get_current_experiment():
    return {
        "active_experiment_id": battery_state["active_experiment_id"],
        "controller_mode": battery_state["controller_mode"],
        "runtime_mode": battery_state["mode"],
        "scenario": battery_state["scenario"],
    }


@app.post("/api/experiments/start")
def start_experiment_run(payload: ExperimentStartPayload):
    if battery_state["active_experiment_id"]:
        raise HTTPException(
            status_code=409,
            detail=f"Experiment already active: {battery_state['active_experiment_id']}",
        )

    experiment_id = (payload.experiment_id or str(uuid.uuid4())).strip()
    if not experiment_id:
        raise HTTPException(status_code=400, detail="experiment_id cannot be empty")

    now_utc = datetime.utcnow().isoformat() + "Z"
    config_snapshot = {
        "controller_mode": battery_state["controller_mode"],
        "runtime_mode": battery_state["mode"],
        "scenario": battery_state["scenario"],
        "command_adapter": battery_state["command_adapter"],
    }

    try:
        start_experiment(
            experiment_id=experiment_id,
            started_at_utc=now_utc,
            run_seed=payload.run_seed,
            controller_mode=battery_state["controller_mode"],
            runtime_mode=battery_state["mode"],
            scenario=battery_state["scenario"],
            config_snapshot=config_snapshot,
            metadata=payload.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    battery_state["active_experiment_id"] = experiment_id
    return {"status": "ok", "experiment_id": experiment_id, "started_at_utc": now_utc}


@app.post("/api/experiments/end")
def end_experiment_run(experiment_id: Optional[str] = None, status: str = "completed"):
    active_id = battery_state["active_experiment_id"]
    target = experiment_id or active_id
    if not target:
        raise HTTPException(status_code=400, detail="No active experiment")

    end_experiment(
        experiment_id=target,
        ended_at_utc=datetime.utcnow().isoformat() + "Z",
        status=status,
    )

    if active_id == target:
        battery_state["active_experiment_id"] = None

    return {"status": "ok", "experiment_id": target, "ended": True}


@app.post("/api/telemetry")
def ingest_telemetry(payload: TelemetryPayload):
    battery_state["telemetry"] = payload.model_dump()
    battery_state["last_telemetry_at"] = datetime.utcnow()

    if payload.soh is not None:
        battery_state["health"] = max(0.0, min(1.0, float(payload.soh)))

    battery_state["temperature"] = float(payload.temperature)
    battery_state["load"] = max(0.0, float(payload.current))

    log_telemetry(
        experiment_id=battery_state.get("active_experiment_id"),
        ts_utc=datetime.utcnow().isoformat() + "Z",
        current_a=float(payload.current),
        temperature_c=float(payload.temperature),
        voltage_v=float(payload.voltage) if payload.voltage is not None else None,
        soc=float(payload.soc) if payload.soc is not None else None,
        soh=float(payload.soh) if payload.soh is not None else None,
        mode=battery_state["mode"],
        source="telemetry_ingest",
        raw_payload=payload.model_dump(),
    )

    return {
        "status": "accepted",
        "mode": battery_state["mode"],
        "received_at_utc": battery_state["last_telemetry_at"].isoformat() + "Z",
    }


def _safe_timestamp(raw_ts: Optional[str]) -> str:
    if raw_ts:
        return raw_ts
    return datetime.utcnow().isoformat() + "Z"


def _to_state(value: Any) -> Literal["on", "off", "idle", "fault"]:
    v = str(value).strip().lower()
    if v in {"on", "running", "active", "charging"}:
        return "on"
    if v in {"off", "stopped", "disabled"}:
        return "off"
    if v in {"idle", "standby", "sleep"}:
        return "idle"
    if v in {"fault", "error", "alarm"}:
        return "fault"
    return "idle"


def _normalize_vendor_payload(raw: RawDevicePayload) -> NormalizedDevicePayload:
    payload = raw.payload

    device_id = raw.device_id or str(payload.get("device_id") or payload.get("id") or f"{raw.vendor}_device")

    power_w = payload.get("power_w")
    if power_w is None and payload.get("power_kw") is not None:
        power_w = float(payload.get("power_kw")) * 1000.0
    if power_w is None and payload.get("usage_kw") is not None:
        power_w = float(payload.get("usage_kw")) * 1000.0
    if power_w is None:
        power_w = 0.0

    temp_c = payload.get("temp_c")
    if temp_c is None and payload.get("temperature") is not None:
        temp_c = float(payload.get("temperature"))

    state = _to_state(payload.get("state") or payload.get("status") or payload.get("mode") or "idle")

    capabilities = raw.capabilities or payload.get("capabilities") or []
    if not isinstance(capabilities, list):
        capabilities = []

    def _int_or_none(x: Any) -> Optional[int]:
        if x is None:
            return None
        try:
            return int(x)
        except Exception:
            return None

    priority = _int_or_none(payload.get("priority")) or 3
    duration_h = _int_or_none(payload.get("duration_h")) or 1
    comfort_weight = float(payload.get("comfort_weight")) if payload.get("comfort_weight") is not None else 1.0

    return NormalizedDevicePayload(
        device_id=device_id,
        device_type=str(payload.get("device_type")) if payload.get("device_type") is not None else None,
        power_w=max(0.0, float(power_w)),
        state=state,
        temp_c=float(temp_c) if temp_c is not None else None,
        capabilities=[str(x) for x in capabilities],
        priority=max(1, min(5, int(priority))),
        comfort_weight=max(0.0, float(comfort_weight)),
        duration_h=max(1, min(24, int(duration_h))),
        preferred_start_h=_int_or_none(payload.get("preferred_start_h")),
        flex_start_h=_int_or_none(payload.get("flex_start_h")),
        flex_end_h=_int_or_none(payload.get("flex_end_h")),
        timestamp=_safe_timestamp(payload.get("timestamp")),
    )


@app.post("/api/devices/telemetry")
def ingest_device_telemetry(payload: NormalizedDevicePayload):
    normalized = payload.model_dump()
    normalized["timestamp"] = _safe_timestamp(payload.timestamp)
    battery_state["devices"][payload.device_id] = normalized

    return {
        "status": "accepted",
        "device": normalized,
        "normalized": True,
    }


@app.post("/api/devices/ingest-raw")
def ingest_raw_device_payload(raw: RawDevicePayload):
    normalized = _normalize_vendor_payload(raw)
    battery_state["devices"][normalized.device_id] = normalized.model_dump()

    return {
        "status": "accepted",
        "vendor": raw.vendor,
        "normalized_device": normalized.model_dump(),
    }


@app.get("/api/devices")
def list_devices():
    return {
        "count": len(battery_state["devices"]),
        "devices": list(battery_state["devices"].values()),
    }


@app.post("/api/devices/load-demo")
def load_demo_devices():
    now = datetime.utcnow().isoformat() + "Z"
    demo_devices = [
        NormalizedDevicePayload(
            device_id="ev_charger_01",
            device_type="ev_charger",
            power_w=7200.0,
            state="on",
            temp_c=39.0,
            capabilities=["can_turn_off", "can_delay_start"],
            priority=4,
            comfort_weight=2.0,
            duration_h=3,
            preferred_start_h=22,
            flex_start_h=20,
            flex_end_h=6,
            timestamp=now,
        ).model_dump(),
        NormalizedDevicePayload(
            device_id="ac_unit_01",
            device_type="ac",
            power_w=3500.0,
            state="on",
            temp_c=33.0,
            capabilities=[],
            priority=5,
            comfort_weight=3.0,
            duration_h=1,
            preferred_start_h=18,
            flex_start_h=0,
            flex_end_h=23,
            timestamp=now,
        ).model_dump(),
        NormalizedDevicePayload(
            device_id="dishwasher_01",
            device_type="dishwasher",
            power_w=2000.0,
            state="on",
            temp_c=31.0,
            capabilities=["can_delay_start"],
            priority=2,
            comfort_weight=1.2,
            duration_h=2,
            preferred_start_h=21,
            flex_start_h=19,
            flex_end_h=23,
            timestamp=now,
        ).model_dump(),
    ]

    for device in demo_devices:
        battery_state["devices"][device["device_id"]] = device

    return {
        "status": "ok",
        "count": len(demo_devices),
        "devices": demo_devices,
    }


def _is_live_telemetry_fresh(timeout_seconds: int = 10) -> bool:
    last_seen = battery_state["last_telemetry_at"]
    if last_seen is None:
        return False
    return datetime.utcnow() - last_seen <= timedelta(seconds=timeout_seconds)


def _append_history(current: float, temp: float):
    battery_state["history"].append({"current": current, "temp": temp})
    battery_state["history"] = battery_state["history"][-120:]


def _predict_soh() -> float:
    # Fallback: use tracked health when model/window is unavailable.
    if model is None or len(battery_state["history"]) < 60:
        return float(battery_state["health"])

    window = np.array([[h["current"], h["temp"]] for h in battery_state["history"][-60:]], dtype=np.float32)
    window = window.reshape(1, 60, 2)
    pred = float(model.predict(window, verbose=0)[0][0])
    return max(0.0, min(1.0, pred))


def _compute_fuzzy_limit(soh_pred: float, temperature: float) -> float:
    if run_governor_test is None:
        return 40.0 if temperature > 45 else 100.0

    try:
        limit = float(run_governor_test(soh_pred, temperature))
        return max(0.0, min(100.0, limit))
    except Exception:
        return 40.0 if temperature > 45 else 100.0


def _log_action(event: str, action: str):
    battery_state["actions"].insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "event": event,
            "action": action,
        },
    )
    battery_state["actions"] = battery_state["actions"][:10]


def _maybe_log_protection_event(status: str, fuzzy_limit: float):
    prev_status = battery_state.get("last_status", "OPTIMAL")
    prev_logged_limit = battery_state.get("last_logged_protection_limit")

    entered_protected = prev_status != "PROTECTED" and status == "PROTECTED"
    limit_shifted = (
        status == "PROTECTED"
        and prev_logged_limit is not None
        and abs(float(fuzzy_limit) - float(prev_logged_limit)) > 5.0
    )

    if entered_protected or limit_shifted:
        _log_action("Thermal/Health Protection", f"Governor set limit to {fuzzy_limit:.1f}%")
        battery_state["last_logged_protection_limit"] = float(fuzzy_limit)

    if status != "PROTECTED":
        battery_state["last_logged_protection_limit"] = None

    battery_state["last_status"] = status


def _maybe_run_ga(current_limit: float):
    if ga_instance is None:
        return

    if abs(current_limit - battery_state["last_safe_limit"]) <= 5.0:
        return

    try:
        ga_instance.run()
        schedule, _, _ = ga_instance.best_solution()
        rounded = [round(float(x), 1) for x in schedule]
        battery_state["last_schedule"] = rounded
        battery_state["last_safe_limit"] = current_limit
        _log_action(
            "Re-optimization",
            f"GA schedule EV/AC/DW -> {rounded[0]}h/{rounded[1]}h/{rounded[2]}h",
        )
    except Exception:
        # Keep previous schedule if GA fails.
        pass


def _compute_control_snapshot() -> Dict[str, Any]:
    temp_offset = 15.0 if battery_state["scenario"] == "HEATWAVE" else 0.0

    use_live = battery_state["mode"] == "LIVE" and _is_live_telemetry_fresh()
    source = "LIVE" if use_live else "SIM"

    if use_live:
        telemetry = battery_state["telemetry"]
        load = max(0.0, float(telemetry.get("current", battery_state["load"])))
        temperature = float(telemetry.get("temperature", battery_state["temperature"])) + temp_offset
        external_soh = telemetry.get("soh")
    else:
        load = max(0.0, float(battery_state["load"]))
        temperature = 30.0 + (load * 0.7) + temp_offset
        external_soh = None

    battery_state["load"] = round(load, 2)
    battery_state["temperature"] = round(temperature, 1)
    _append_history(load, temperature)

    controller_mode = battery_state.get("controller_mode", "rnn_fuzzy_ga")

    if controller_mode in {"rnn_fuzzy", "rnn_fuzzy_ga"}:
        soh_pred = _predict_soh()
    else:
        soh_pred = float(battery_state["health"])

    if controller_mode == "rule_only":
        fuzzy_limit = 40.0 if temperature > 45 else 100.0
    else:
        fuzzy_limit = _compute_fuzzy_limit(soh_pred, temperature)

    status = "PROTECTED" if float(fuzzy_limit) < 80.0 else "OPTIMAL"

    stress_decay = (load * temperature) / 5000000
    if external_soh is not None:
        tracked_health = max(0.0, min(1.0, float(external_soh)))
    else:
        blended = (0.85 * float(battery_state["health"])) + (0.15 * soh_pred)
        tracked_health = max(0.0, min(1.0, blended - stress_decay))
    battery_state["health"] = round(tracked_health, 6)

    _maybe_log_protection_event(status, fuzzy_limit)
    if controller_mode == "rnn_fuzzy_ga":
        _maybe_run_ga(fuzzy_limit)

    log_control_decision(
        experiment_id=battery_state.get("active_experiment_id"),
        ts_utc=datetime.utcnow().isoformat() + "Z",
        controller_mode=controller_mode,
        runtime_mode=battery_state["mode"],
        source=source,
        telemetry_fresh=use_live,
        load_a=float(load),
        temp_c=float(temperature),
        health=float(battery_state["health"]),
        soh_pred=float(soh_pred),
        fuzzy_limit_pct=float(fuzzy_limit),
        status=status,
    )

    forecast = []
    temp_health = battery_state["health"]
    for i in range(10):
        step_decay = stress_decay * (1.08**i)
        temp_health = max(0.0, temp_health - step_decay)
        forecast.append(round(temp_health, 6))

    return {
        "source": source,
        "telemetry_fresh": use_live,
        "load": load,
        "temperature": temperature,
        "soh_pred": soh_pred,
        "fuzzy_limit": fuzzy_limit,
        "status": status,
        "forecast": forecast,
        "controller_mode": controller_mode,
    }


def _build_appliance_cards(status: str, fuzzy_limit: float, load: float) -> List[Dict[str, str]]:
    ev_on = status == "OPTIMAL"
    dishwasher_on = load < 28 and fuzzy_limit >= 60

    if battery_state["devices"]:
        cards = []
        for device in battery_state["devices"].values():
            forced_off = status == "PROTECTED" and device["power_w"] >= 5000
            device_on = device["state"] == "on" and not forced_off
            cards.append(
                {
                    "name": device["device_id"].replace("_", " ").title(),
                    "status": "ON" if device_on else "OFF (AI)",
                    "usage": f"{(device['power_w'] / 1000.0 if device_on else 0.0):.2f}",
                }
            )
        return cards[:6]

    return [
        {
            "name": "EV Charger",
            "status": "OFF (AI)" if not ev_on else "ON",
            "usage": "0.00" if not ev_on else "7.50",
        },
        {
            "name": "AC Unit",
            "status": "ON",
            "usage": "3.50",
        },
        {
            "name": "Dishwasher",
            "status": "ON" if dishwasher_on else "OFF (AI)",
            "usage": "2.00" if dishwasher_on else "0.00",
        },
    ]


def _hour_in_window(hour: int, start_h: int, end_h: int) -> bool:
    if start_h <= end_h:
        return start_h <= hour <= end_h
    # Wrap-around window, e.g., 20 -> 6
    return hour >= start_h or hour <= end_h


def _optimize_flexible_schedule(devices: List[Dict[str, Any]], fuzzy_limit: float) -> Dict[str, Any]:
    # Price proxy currently used by GA module; keep aligned for comparability.
    grid_prices = np.array(
        [
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.2, 0.2,
            0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.7, 0.4, 0.2, 0.1, 0.1,
        ],
        dtype=float,
    )
    nominal_system_kw = 10.0
    allowed_kw = nominal_system_kw * (max(0.0, min(100.0, float(fuzzy_limit))) / 100.0)

    flexible = []
    for d in devices:
        caps = set(d.get("capabilities", []))
        if d.get("state") == "on" and "can_delay_start" in caps:
            flexible.append(d)

    if not flexible:
        return {"schedule": {}, "feasible": True, "violations": 0, "objective": 0.0}

    n = len(flexible)
    pop_size = 36
    generations = 45
    mutation_prob = 0.2
    rng = np.random.default_rng()

    starts = np.array([int(d.get("flex_start_h") if d.get("flex_start_h") is not None else 0) for d in flexible])
    ends = np.array([int(d.get("flex_end_h") if d.get("flex_end_h") is not None else 23) for d in flexible])
    durations = np.array([max(1, min(24, int(d.get("duration_h", 1)))) for d in flexible])
    power_kw = np.array([max(0.0, float(d.get("power_w", 0.0)) / 1000.0) for d in flexible])
    priorities = np.array([max(1, min(5, int(d.get("priority", 3)))) for d in flexible], dtype=float)
    comfort_w = np.array([max(0.0, float(d.get("comfort_weight", 1.0))) for d in flexible], dtype=float)
    preferred = np.array([int(d.get("preferred_start_h") if d.get("preferred_start_h") is not None else 20) for d in flexible])

    def evaluate(candidate: np.ndarray) -> Dict[str, float]:
        candidate = np.clip(candidate.astype(int), 0, 23)
        infeas = 0.0
        comfort_pen = 0.0
        hour_load = np.zeros(24, dtype=float)
        energy_cost = 0.0

        for i in range(n):
            h = int(candidate[i])
            if not _hour_in_window(h, int(starts[i]), int(ends[i])):
                infeas += 1.0

            # Circular hour-distance for comfort.
            dist = min((h - preferred[i]) % 24, (preferred[i] - h) % 24)
            comfort_pen += (dist * comfort_w[i]) / priorities[i]

            for k in range(int(durations[i])):
                hh = (h + k) % 24
                hour_load[hh] += power_kw[i]
                energy_cost += power_kw[i] * grid_prices[hh]

        overload = np.maximum(0.0, hour_load - allowed_kw)
        degradation_pen = float(np.sum(overload ** 2) * 15.0)
        infeas_pen = float(infeas * 150.0)
        objective = float(energy_cost + degradation_pen + comfort_pen + infeas_pen)
        return {
            "objective": objective,
            "energy_cost": float(energy_cost),
            "degradation_penalty": float(degradation_pen),
            "comfort_penalty": float(comfort_pen),
            "infeasibility_penalty": float(infeas_pen),
            "violations": int(infeas + np.sum(overload > 0)),
        }

    # Initialize and evolve a lightweight GA-style population.
    pop = rng.integers(0, 24, size=(pop_size, n), endpoint=False)
    best_vec = pop[0].copy()
    best_eval = evaluate(best_vec)

    for _ in range(generations):
        scores = []
        evals = []
        for i in range(pop_size):
            ev = evaluate(pop[i])
            evals.append(ev)
            scores.append(ev["objective"])

        order = np.argsort(np.array(scores))
        elite = pop[order[: max(2, pop_size // 6)]]
        if scores[order[0]] < best_eval["objective"]:
            best_vec = pop[order[0]].copy()
            best_eval = evals[order[0]]

        children = []
        while len(children) < pop_size:
            p1 = elite[rng.integers(0, len(elite))]
            p2 = elite[rng.integers(0, len(elite))]
            cut = rng.integers(1, n + 1)
            child = np.concatenate([p1[:cut], p2[cut:]])
            mut_mask = rng.random(n) < mutation_prob
            if np.any(mut_mask):
                child[mut_mask] = rng.integers(0, 24, size=np.sum(mut_mask), endpoint=False)
            children.append(child)
        pop = np.array(children, dtype=int)

    schedule = {flexible[i]["device_id"]: int(best_vec[i]) for i in range(n)}
    feasible = best_eval["violations"] == 0
    return {
        "schedule": schedule,
        "feasible": bool(feasible),
        "violations": int(best_eval["violations"]),
        "objective": round(float(best_eval["objective"]), 6),
        "energy_cost": round(float(best_eval["energy_cost"]), 6),
        "degradation_penalty": round(float(best_eval["degradation_penalty"]), 6),
        "comfort_penalty": round(float(best_eval["comfort_penalty"]), 6),
        "infeasibility_penalty": round(float(best_eval["infeasibility_penalty"]), 6),
        "allowed_kw": round(float(allowed_kw), 4),
    }


def _ensure_mqtt_client() -> Optional[Any]:
    if mqtt is None:
        battery_state["mqtt_error"] = "paho-mqtt not installed"
        return None

    client = battery_state.get("mqtt_client")
    if client is not None and battery_state.get("mqtt_connected"):
        return client

    cfg = battery_state["command_adapter"]
    try:
        client = mqtt.Client()
        if cfg.get("mqtt_username"):
            client.username_pw_set(cfg["mqtt_username"], cfg.get("mqtt_password"))
        client.connect(cfg["mqtt_host"], int(cfg["mqtt_port"]), 10)
        battery_state["mqtt_client"] = client
        battery_state["mqtt_connected"] = True
        battery_state["mqtt_error"] = None
        return client
    except Exception as exc:
        battery_state["mqtt_connected"] = False
        battery_state["mqtt_error"] = str(exc)
        return None


def _command_key(device_id: str, command: str, payload: Dict[str, Any]) -> str:
    payload_str = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return f"{device_id}|{command}|{payload_str}"


def _is_duplicate_command(key: str) -> bool:
    window = int(battery_state.get("command_dedupe_window_sec", 30))
    now = datetime.utcnow()
    cache = battery_state.get("recent_command_keys", {})

    # Cleanup stale keys opportunistically.
    stale = []
    for k, ts in cache.items():
        try:
            last = datetime.fromisoformat(str(ts).replace("Z", ""))
            if (now - last).total_seconds() > window:
                stale.append(k)
        except Exception:
            stale.append(k)
    for k in stale:
        cache.pop(k, None)

    if key in cache:
        return True
    cache[key] = now.isoformat() + "Z"
    return False


def send_command(device_id: str, command: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    adapter_type = battery_state["command_adapter"]["type"]
    command_payload = payload or {}
    key = _command_key(device_id, command, command_payload)
    out = {
        "command_id": f"cmd_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
        "device_id": device_id,
        "command": command,
        "payload": command_payload,
        "adapter": adapter_type,
        "status": "queued",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "retries": 0,
    }

    if _is_duplicate_command(key):
        out["status"] = "duplicate_skipped"
        battery_state["command_outbox"].insert(0, out)
        battery_state["command_outbox"] = battery_state["command_outbox"][:100]
        return out

    if adapter_type == "mqtt":
        topic_prefix = battery_state["command_adapter"]["mqtt_topic_prefix"].rstrip("/")
        topic = f"{topic_prefix}/{device_id}"
        message = json.dumps(
            {
                "command_id": out["command_id"],
                "device_id": device_id,
                "command": command,
                "payload": command_payload,
                "timestamp": out["timestamp"],
            }
        )
        attempts = max(1, int(battery_state.get("command_retry_max", 2)) + 1)
        for attempt in range(attempts):
            client = _ensure_mqtt_client()
            out["retries"] = attempt
            if client is None:
                out["status"] = "failed"
                out["error"] = battery_state["mqtt_error"] or "mqtt unavailable"
            else:
                try:
                    result = client.publish(topic, message)
                    rc = getattr(result, "rc", 0)
                    if rc == 0:
                        out["status"] = "published"
                        out["topic"] = topic
                        out.pop("error", None)
                        break
                    out["status"] = "failed"
                    out["error"] = f"mqtt publish rc={rc}"
                except Exception as exc:
                    out["status"] = "failed"
                    out["error"] = str(exc)

            if out["status"] == "published":
                break

    battery_state["command_outbox"].insert(0, out)
    battery_state["command_outbox"] = battery_state["command_outbox"][:100]
    return out


@app.get("/api/control/outbox")
def get_control_outbox():
    return {
        "count": len(battery_state["command_outbox"]),
        "commands": battery_state["command_outbox"],
    }


@app.post("/api/control/outbox/clear")
def clear_control_outbox():
    cleared = len(battery_state["command_outbox"])
    battery_state["command_outbox"] = []
    return {"status": "ok", "cleared": cleared}


@app.post("/api/control/ack")
def ingest_command_ack(payload: CommandAckPayload):
    # Find command in outbox and update state idempotently.
    matched = None
    for c in battery_state["command_outbox"]:
        if c.get("command_id") == payload.command_id:
            matched = c
            break

    if matched is None:
        raise HTTPException(status_code=404, detail=f"Unknown command_id: {payload.command_id}")

    new_status = "acknowledged" if payload.ack_status == "ack" else "nack"
    matched["status"] = new_status
    if payload.message:
        matched["ack_message"] = payload.message

    ack_ts = payload.ack_ts_utc or (datetime.utcnow().isoformat() + "Z")
    matched["ack_ts_utc"] = ack_ts

    latency_ms = None
    try:
        sent_ts = datetime.fromisoformat(str(matched.get("timestamp", "")).replace("Z", ""))
        recv_ts = datetime.fromisoformat(str(ack_ts).replace("Z", ""))
        latency_ms = max(0.0, (recv_ts - sent_ts).total_seconds() * 1000.0)
        matched["ack_latency_ms"] = round(latency_ms, 3)
    except Exception:
        latency_ms = None

    ack_record = {
        "command_id": payload.command_id,
        "device_id": payload.device_id,
        "ack_status": payload.ack_status,
        "message": payload.message,
        "ack_ts_utc": ack_ts,
        "latency_ms": latency_ms,
    }
    battery_state["command_ack_log"].insert(0, ack_record)
    battery_state["command_ack_log"] = battery_state["command_ack_log"][:200]

    update_dispatch_command_status(payload.command_id, new_status)
    log_command_ack(
        experiment_id=battery_state.get("active_experiment_id"),
        ts_utc=datetime.utcnow().isoformat() + "Z",
        command_id=payload.command_id,
        device_id=payload.device_id,
        ack_status=payload.ack_status,
        message=payload.message,
        latency_ms=latency_ms,
        raw_payload=payload.model_dump(),
    )

    return {"status": "ok", "command_id": payload.command_id, "new_status": new_status, "latency_ms": latency_ms}


@app.get("/api/control/metrics")
def get_control_metrics():
    outbox = list(battery_state["command_outbox"])
    total = len(outbox)
    published = len([c for c in outbox if c.get("status") == "published"])
    failed = len([c for c in outbox if c.get("status") == "failed"])
    duplicates = len([c for c in outbox if c.get("status") == "duplicate_skipped"])
    acked = len([c for c in outbox if c.get("status") == "acknowledged"])
    nack = len([c for c in outbox if c.get("status") == "nack"])

    ack_latencies = [float(a.get("latency_ms")) for a in battery_state["command_ack_log"] if a.get("latency_ms") is not None]
    median_ack_ms = float(np.median(np.array(ack_latencies, dtype=float))) if ack_latencies else None

    stale_cutoff = datetime.utcnow() - timedelta(seconds=60)
    stale = 0
    for c in outbox:
        try:
            ts = datetime.fromisoformat(str(c.get("timestamp", "")).replace("Z", ""))
            if ts < stale_cutoff and c.get("status") in {"queued", "published"}:
                stale += 1
        except Exception:
            continue

    success_rate = (acked / total) if total else 0.0
    failure_rate = (failed / total) if total else 0.0
    stale_ratio = (stale / total) if total else 0.0
    safety_cmds = len([c for c in outbox if c.get("payload", {}).get("reason") == "safety_override"])
    optimization_cmds = len([c for c in outbox if c.get("payload", {}).get("reason") == "optimization"])

    return {
        "total_commands": total,
        "published": published,
        "failed": failed,
        "duplicate_skipped": duplicates,
        "acknowledged": acked,
        "nack": nack,
        "success_rate": round(success_rate, 6),
        "failure_rate": round(failure_rate, 6),
        "stale_command_ratio": round(stale_ratio, 6),
        "median_ack_latency_ms": round(median_ack_ms, 3) if median_ack_ms is not None else None,
        "safety_override_command_rate": round((safety_cmds / total), 6) if total else 0.0,
        "optimization_command_rate": round((optimization_cmds / total), 6) if total else 0.0,
    }


@app.get("/api/control/adapter")
def get_control_adapter():
    cfg = battery_state["command_adapter"].copy()
    if cfg.get("mqtt_password"):
        cfg["mqtt_password"] = "***"
    return {
        "config": cfg,
        "mqtt_connected": battery_state["mqtt_connected"],
        "mqtt_error": battery_state["mqtt_error"],
        "mqtt_library_available": mqtt is not None,
    }


@app.post("/api/control/adapter")
def set_control_adapter(payload: AdapterConfigPayload):
    cfg = battery_state["command_adapter"]
    cfg["type"] = payload.adapter_type

    if payload.mqtt_host is not None:
        cfg["mqtt_host"] = payload.mqtt_host
    if payload.mqtt_port is not None:
        cfg["mqtt_port"] = int(payload.mqtt_port)
    if payload.mqtt_username is not None:
        cfg["mqtt_username"] = payload.mqtt_username
    if payload.mqtt_password is not None:
        cfg["mqtt_password"] = payload.mqtt_password
    if payload.mqtt_topic_prefix is not None:
        cfg["mqtt_topic_prefix"] = payload.mqtt_topic_prefix

    # Force reconnect on next publish if MQTT settings changed.
    battery_state["mqtt_connected"] = False
    battery_state["mqtt_error"] = None
    battery_state["mqtt_client"] = None

    safe_cfg = cfg.copy()
    if safe_cfg.get("mqtt_password"):
        safe_cfg["mqtt_password"] = "***"
    return {"status": "ok", "config": safe_cfg}


@app.post("/api/control/dispatch")
def dispatch_control(request: DispatchRequest):
    snapshot = _compute_control_snapshot()
    status = snapshot["status"]
    fuzzy_limit = float(snapshot["fuzzy_limit"])

    devices = list(battery_state["devices"].values())
    commands: List[Dict[str, Any]] = []
    executed: List[Dict[str, Any]] = []
    optimization_result: Dict[str, Any] = {}

    # HARD SAFETY OVERRIDE has priority over optimization.
    if status == "PROTECTED":
        on_devices = [d for d in devices if d["state"] == "on"]
        total_power = sum(float(d["power_w"]) for d in on_devices)
        allowed_power = total_power * (fuzzy_limit / 100.0)

        sorted_devices = sorted(on_devices, key=lambda d: float(d["power_w"]), reverse=True)
        running_power = total_power
        for device in sorted_devices:
            if running_power <= allowed_power:
                break

            capabilities = set(device.get("capabilities", []))
            if "can_turn_off" in capabilities:
                command = {
                    "device_id": device["device_id"],
                    "command": "turn_off",
                    "reason": "safety_override",
                }
                commands.append(command)
                running_power -= float(device["power_w"])
                if not request.dry_run:
                    executed.append(send_command(device["device_id"], "turn_off", {"reason": "safety_override"}))
                    device["state"] = "off"
                    device["power_w"] = 0.0

    elif request.optimization_enabled and devices:
        optimization_result = _optimize_flexible_schedule(devices, fuzzy_limit)
        schedule = optimization_result.get("schedule", {})
        for device in devices:
            if device.get("state") != "on":
                continue
            caps = set(device.get("capabilities", []))
            if "can_delay_start" not in caps:
                continue
            target_hour = schedule.get(device["device_id"])
            if target_hour is None:
                continue

            command = {
                "device_id": device["device_id"],
                "command": "delay_start",
                "target_hour": int(target_hour),
                "reason": "optimization",
            }
            commands.append(command)
            if not request.dry_run:
                executed.append(
                    send_command(
                        device["device_id"],
                        "delay_start",
                        {"target_hour": int(target_hour), "reason": "optimization"},
                    )
                )

    battery_state["last_dispatch"] = commands

    if commands:
        summary = ", ".join([f"{c['device_id']}:{c['command']}" for c in commands[:3]])
        _log_action("Control Dispatch", f"Issued {len(commands)} command(s) [{summary}]")
    if optimization_result:
        feas = "feasible" if optimization_result.get("feasible") else "infeasible"
        _log_action(
            "Schedule Optimization",
            f"{feas} objective={optimization_result.get('objective')} violations={optimization_result.get('violations')}",
        )

    # Persist command decisions for reproducible evaluation.
    exec_lookup = {
        (e.get("device_id"), e.get("command")): e for e in executed if isinstance(e, dict)
    }
    for c in commands:
        match = exec_lookup.get((c.get("device_id"), c.get("command")))
        log_dispatch_command(
            experiment_id=battery_state.get("active_experiment_id"),
            ts_utc=datetime.utcnow().isoformat() + "Z",
            command_id=match.get("command_id") if match else None,
            device_id=str(c.get("device_id")),
            command=str(c.get("command")),
            reason=str(c.get("reason", "")),
            adapter=str(match.get("adapter") if match else battery_state["command_adapter"]["type"]),
            status=str(match.get("status") if match else ("dry_run" if request.dry_run else "planned")),
            dry_run=bool(request.dry_run),
            payload=match.get("payload") if match else c,
        )

    return {
        "status": "ok",
        "dry_run": request.dry_run,
        "safety_override_applied": status == "PROTECTED",
        "control_status": status,
        "fuzzy_limit": f"{fuzzy_limit:.1f}%",
        "commands": commands,
        "executed": executed,
        "optimization": optimization_result,
    }


@app.get("/api/system-status")
def get_system_status(manual_load: float = 22):
    # Keep SIM support for UI slider while allowing LIVE telemetry path.
    if battery_state["mode"] == "SIM":
        battery_state["load"] = max(0.0, float(manual_load))

    snapshot = _compute_control_snapshot()

    return {
        "mode": battery_state["mode"],
        "active_experiment_id": battery_state["active_experiment_id"],
        "controller_mode": snapshot["controller_mode"],
        "source": snapshot["source"],
        "telemetry_fresh": snapshot["telemetry_fresh"],
        "command_adapter": battery_state["command_adapter"]["type"],
        "mqtt_connected": battery_state["mqtt_connected"],
        "model_active": model is not None,
        "fuzzy_active": run_governor_test is not None,
        "ga_active": ga_instance is not None,
        "metrics": {
            "load": f"{snapshot['load']:.1f}A",
            "health": f"{battery_state['health']:.4f}",
            "temp": round(snapshot["temperature"], 1),
            "status": snapshot["status"],
            "limit": f"{snapshot['fuzzy_limit']:.1f}%",
        },
        "forecast": snapshot["forecast"],
        "scenario": battery_state["scenario"],
        "battery_savings": "+18%",
        "schedule": {
            "ev_hour": battery_state["last_schedule"][0],
            "ac_hour": battery_state["last_schedule"][1],
            "dishwasher_hour": battery_state["last_schedule"][2],
        },
        "devices": list(battery_state["devices"].values()),
        "appliances": _build_appliance_cards(snapshot["status"], snapshot["fuzzy_limit"], snapshot["load"]),
        "recent_actions": battery_state["actions"],
        "last_dispatch": battery_state["last_dispatch"],
    }
