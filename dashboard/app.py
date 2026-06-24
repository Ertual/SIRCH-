from __future__ import annotations

import csv
import io
from pathlib import Path

from flask import Flask, Response, redirect, render_template, request, send_from_directory, url_for

import config
from database.db import export_rows, get_settings, init_db, list_incidents, save_settings


app = Flask(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPTURES_DIR = (PROJECT_ROOT / config.CAPTURES_FOLDER).resolve()


@app.route("/")
def index():
    incidents = list_incidents(limit=10)
    settings = get_settings()
    return render_template(
        "index.html",
        incidents=incidents,
        threshold=settings["threshold"],
        k_window=settings["k_window"],
        cooldown=config.COOLDOWN_SECONDS,
    )


@app.route("/incidents")
def incidents():
    return render_template("incidents.html", incidents=list_incidents(limit=200))


@app.route("/settings", methods=["POST"])
def settings():
    threshold = float(request.form.get("threshold", config.THRESHOLD))
    k_window = int(request.form.get("k_window", config.K_WINDOW))
    threshold = min(max(threshold, 0.0), 1.0)
    k_window = max(k_window, 1)
    save_settings(threshold, k_window)
    return redirect(url_for("index"))


@app.route("/captures/<path:filename>")
def capture(filename):
    return send_from_directory(CAPTURES_DIR, Path(filename).name)


@app.route("/export.csv")
def export_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "created_at", "score", "mean_score", "capture_path", "source"])
    for row in export_rows():
        writer.writerow([row["id"], row["created_at"], row["score"], row["mean_score"], row["capture_path"], row["source"]])
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=sirch_incidents.csv"},
    )


def run_dashboard() -> None:
    init_db()
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)


if __name__ == "__main__":
    run_dashboard()
