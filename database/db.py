from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Iterable

from config import DB_PATH, K_WINDOW, THRESHOLD


SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    score REAL NOT NULL,
    mean_score REAL NOT NULL,
    capture_path TEXT,
    source TEXT
);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: str = DB_PATH) -> None:
    with get_connection(db_path) as connection:
        connection.executescript(SCHEMA)
        connection.commit()


def add_incident(
    score: float,
    mean_score: float,
    capture_path: str | None = None,
    source: str | None = None,
    db_path: str = DB_PATH,
) -> int:
    init_db(db_path)
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO incidents (created_at, score, mean_score, capture_path, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                float(score),
                float(mean_score),
                capture_path,
                source,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def list_incidents(db_path: str = DB_PATH, limit: int = 100) -> list[sqlite3.Row]:
    init_db(db_path)
    with get_connection(db_path) as connection:
        return list(
            connection.execute(
                """
                SELECT id, created_at, score, mean_score, capture_path, source
                FROM incidents
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            )
        )


def export_rows(db_path: str = DB_PATH) -> Iterable[sqlite3.Row]:
    init_db(db_path)
    with get_connection(db_path) as connection:
        yield from connection.execute(
            """
            SELECT id, created_at, score, mean_score, capture_path, source
            FROM incidents
            ORDER BY id DESC
            """
        )


def save_settings(threshold: float, k_window: int, db_path: str = DB_PATH) -> None:
    init_db(db_path)
    with get_connection(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO app_settings (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            [
                ("threshold", str(float(threshold))),
                ("k_window", str(int(k_window))),
            ],
        )
        connection.commit()


def get_settings(db_path: str = DB_PATH) -> dict[str, float | int]:
    init_db(db_path)
    settings: dict[str, float | int] = {
        "threshold": float(THRESHOLD),
        "k_window": int(K_WINDOW),
    }
    with get_connection(db_path) as connection:
        rows = connection.execute("SELECT key, value FROM app_settings")
        for row in rows:
            if row["key"] == "threshold":
                settings["threshold"] = float(row["value"])
            if row["key"] == "k_window":
                settings["k_window"] = int(row["value"])
    return settings
