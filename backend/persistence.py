import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Optional


ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("BATTERY_DB_PATH", str(ROOT_DIR / "data" / "ops" / "battery_ops.db")))

_LOCK = threading.Lock()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _LOCK:
        conn = _connect()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                started_at_utc TEXT NOT NULL,
                ended_at_utc TEXT,
                status TEXT NOT NULL,
                run_seed INTEGER,
                controller_mode TEXT,
                runtime_mode TEXT,
                scenario TEXT,
                config_snapshot_json TEXT,
                metadata_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                ts_utc TEXT NOT NULL,
                current_a REAL,
                temperature_c REAL,
                voltage_v REAL,
                soc REAL,
                soh REAL,
                mode TEXT,
                source TEXT,
                raw_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS control_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                ts_utc TEXT NOT NULL,
                controller_mode TEXT,
                runtime_mode TEXT,
                source TEXT,
                telemetry_fresh INTEGER,
                load_a REAL,
                temp_c REAL,
                health REAL,
                soh_pred REAL,
                fuzzy_limit_pct REAL,
                status TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS dispatch_commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                ts_utc TEXT NOT NULL,
                command_id TEXT,
                device_id TEXT,
                command TEXT,
                reason TEXT,
                adapter TEXT,
                status TEXT,
                dry_run INTEGER,
                payload_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS command_acks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                ts_utc TEXT NOT NULL,
                command_id TEXT,
                device_id TEXT,
                ack_status TEXT,
                message TEXT,
                latency_ms REAL,
                raw_json TEXT
            )
            """
        )
        conn.commit()
        conn.close()


def start_experiment(
    experiment_id: str,
    started_at_utc: str,
    run_seed: Optional[int],
    controller_mode: str,
    runtime_mode: str,
    scenario: str,
    config_snapshot: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    with _LOCK:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM experiments WHERE experiment_id = ?", (experiment_id,))
        if cur.fetchone():
            conn.close()
            raise ValueError(f"Experiment already exists: {experiment_id}")

        cur.execute(
            """
            INSERT INTO experiments (
                experiment_id, started_at_utc, status, run_seed, controller_mode, runtime_mode, scenario,
                config_snapshot_json, metadata_json
            )
            VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                started_at_utc,
                run_seed,
                controller_mode,
                runtime_mode,
                scenario,
                json.dumps(config_snapshot, ensure_ascii=True),
                json.dumps(metadata or {}, ensure_ascii=True),
            ),
        )
        conn.commit()
        conn.close()


def end_experiment(experiment_id: str, ended_at_utc: str, status: str = "completed") -> None:
    with _LOCK:
        conn = _connect()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE experiments
            SET ended_at_utc = ?, status = ?
            WHERE experiment_id = ?
            """,
            (ended_at_utc, status, experiment_id),
        )
        conn.commit()
        conn.close()


def log_telemetry(
    experiment_id: Optional[str],
    ts_utc: str,
    current_a: Optional[float],
    temperature_c: Optional[float],
    voltage_v: Optional[float],
    soc: Optional[float],
    soh: Optional[float],
    mode: str,
    source: str,
    raw_payload: Optional[Dict[str, Any]],
) -> None:
    with _LOCK:
        conn = _connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO telemetry (
                experiment_id, ts_utc, current_a, temperature_c, voltage_v, soc, soh, mode, source, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                ts_utc,
                current_a,
                temperature_c,
                voltage_v,
                soc,
                soh,
                mode,
                source,
                json.dumps(raw_payload or {}, ensure_ascii=True),
            ),
        )
        conn.commit()
        conn.close()


def log_control_decision(
    experiment_id: Optional[str],
    ts_utc: str,
    controller_mode: str,
    runtime_mode: str,
    source: str,
    telemetry_fresh: bool,
    load_a: float,
    temp_c: float,
    health: float,
    soh_pred: float,
    fuzzy_limit_pct: float,
    status: str,
) -> None:
    with _LOCK:
        conn = _connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO control_decisions (
                experiment_id, ts_utc, controller_mode, runtime_mode, source, telemetry_fresh,
                load_a, temp_c, health, soh_pred, fuzzy_limit_pct, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                ts_utc,
                controller_mode,
                runtime_mode,
                source,
                1 if telemetry_fresh else 0,
                load_a,
                temp_c,
                health,
                soh_pred,
                fuzzy_limit_pct,
                status,
            ),
        )
        conn.commit()
        conn.close()


def log_dispatch_command(
    experiment_id: Optional[str],
    ts_utc: str,
    command_id: Optional[str],
    device_id: str,
    command: str,
    reason: str,
    adapter: str,
    status: str,
    dry_run: bool,
    payload: Optional[Dict[str, Any]],
) -> None:
    with _LOCK:
        conn = _connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO dispatch_commands (
                experiment_id, ts_utc, command_id, device_id, command, reason, adapter, status, dry_run, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                ts_utc,
                command_id,
                device_id,
                command,
                reason,
                adapter,
                status,
                1 if dry_run else 0,
                json.dumps(payload or {}, ensure_ascii=True),
            ),
        )
        conn.commit()
        conn.close()


def update_dispatch_command_status(command_id: str, status: str) -> None:
    with _LOCK:
        conn = _connect()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE dispatch_commands
            SET status = ?
            WHERE command_id = ?
            """,
            (status, command_id),
        )
        conn.commit()
        conn.close()


def log_command_ack(
    experiment_id: Optional[str],
    ts_utc: str,
    command_id: str,
    device_id: str,
    ack_status: str,
    message: Optional[str],
    latency_ms: Optional[float],
    raw_payload: Optional[Dict[str, Any]],
) -> None:
    with _LOCK:
        conn = _connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO command_acks (
                experiment_id, ts_utc, command_id, device_id, ack_status, message, latency_ms, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                ts_utc,
                command_id,
                device_id,
                ack_status,
                message,
                latency_ms,
                json.dumps(raw_payload or {}, ensure_ascii=True),
            ),
        )
        conn.commit()
        conn.close()
