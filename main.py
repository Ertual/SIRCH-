from __future__ import annotations

import argparse
import os
from datetime import datetime

import cv2

from alerts.alertmanager import AlertManager, AlertPayload
from config import CAPTURES_FOLDER
from core.acquisition import VideoSource
from core.decision import DecisionEngine
from core.inference import ViolenceInference
from core.preprocessing import FrameSequenceBuffer
from database.db import add_incident, get_settings, init_db


WINDOW_NAME = "SIRCH - Camera"


def save_capture(frame) -> str:
    os.makedirs(CAPTURES_FOLDER, exist_ok=True)
    filename = datetime.now().strftime("capture_%Y%m%d_%H%M%S.jpg")
    path = os.path.join(CAPTURES_FOLDER, filename)
    cv2.imwrite(path, frame)
    return path


def draw_status_overlay(
    frame,
    score: float | None,
    mean_score: float | None,
    threshold: float,
    k_window: int,
    infer_every: int,
    is_violence: bool,
    buffer_count: int,
    n_frames: int,
    last_capture_path: str | None,
    last_capture_time: datetime | None,
):
    overlay = frame.copy()
    height, width = frame.shape[:2]
    box_width = min(620, width - 20)
    box_height = 150
    cv2.rectangle(overlay, (10, 10), (10 + box_width, 10 + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    status_text = "VIOLENCE DETECTEE" if is_violence else "OK - surveillance"
    status_color = (0, 0, 255) if is_violence else (0, 220, 0)

    lines = [
        ("SIRCH temps reel - appuie sur q dans cette fenetre pour quitter", (255, 255, 255)),
        (status_text, status_color),
        (f"Score: {score:.3f}" if score is not None else f"Preparation sequence: {buffer_count}/{n_frames} frames", (255, 255, 255)),
        (f"Moyenne K: {mean_score:.3f} | Seuil: {threshold:.2f} | K: {k_window}" if mean_score is not None else f"Seuil: {threshold:.2f} | K: {k_window}", (255, 255, 255)),
        (f"Mode faible latence: prediction toutes les {infer_every} frames", (180, 220, 255)),
    ]

    if last_capture_path and last_capture_time:
        capture_name = os.path.basename(last_capture_path)
        lines.append((f"Derniere capture: {capture_name} a {last_capture_time.strftime('%H:%M:%S')}", (0, 255, 255)))
    else:
        lines.append(("Aucune capture enregistree pour l'instant", (180, 180, 180)))

    y = 35
    for text, color in lines:
        cv2.putText(frame, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        y += 25

    if is_violence:
        cv2.rectangle(frame, (4, 4), (width - 5, height - 5), (0, 0, 255), 5)

    return frame


def run(
    source: str = "0",
    show: bool = True,
    infer_every: int = 5,
    camera_width: int = 640,
    camera_height: int = 480,
    camera_fps: int = 15,
) -> None:
    init_db()
    infer_every = max(1, int(infer_every))
    buffer = FrameSequenceBuffer()
    inference = ViolenceInference()
    decision = DecisionEngine()
    alerts = AlertManager()
    settings = get_settings()
    threshold = float(settings["threshold"])
    k_window = int(settings["k_window"])
    last_score: float | None = None
    last_mean_score: float | None = None
    last_is_violence = False
    last_capture_path: str | None = None
    last_capture_time: datetime | None = None

    if show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 960, 540)
        cv2.moveWindow(WINDOW_NAME, 20, 80)

    with VideoSource(source, width=camera_width, height=camera_height, fps=camera_fps, buffer_size=1) as video:
        print("SIRCH demarre. Appuie sur q dans la fenetre camera pour quitter.")
        print(f"Mode faible latence : camera {camera_width}x{camera_height}@{camera_fps}fps, prediction toutes les {infer_every} frames.")
        frame_index = 0
        while True:
            frame = video.read()
            if frame is None:
                break
            frame_index += 1

            sequence = buffer.add_frame(frame)
            should_predict = sequence is not None and frame_index % infer_every == 0
            if should_predict:
                score = float(inference.predict(sequence))
                settings = get_settings()
                threshold = float(settings["threshold"])
                k_window = int(settings["k_window"])
                decision.configure(threshold, k_window)
                is_violence, mean_score = decision.update(score)
                last_score = score
                last_mean_score = mean_score
                last_is_violence = is_violence
                if is_violence:
                    capture_path = save_capture(frame)
                    last_capture_path = capture_path
                    last_capture_time = datetime.now()
                    add_incident(score, mean_score, capture_path, str(source))
                    alerts.send_alerts(AlertPayload(score=mean_score, capture_path=capture_path))

            if show:
                display_frame = draw_status_overlay(
                    frame.copy(),
                    last_score,
                    last_mean_score,
                    threshold,
                    k_window,
                    infer_every,
                    last_is_violence,
                    len(buffer.frames),
                    buffer.n_frames,
                    last_capture_path,
                    last_capture_time,
                )
                cv2.imshow(WINDOW_NAME, display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIRCH - detection de violence en temps reel")
    parser.add_argument("--source", default="0", help="Webcam index, flux RTSP/IP ou fichier video")
    parser.add_argument("--no-window", action="store_true", help="Desactive la fenetre OpenCV")
    parser.add_argument("--infer-every", type=int, default=5, help="Nombre de frames entre deux predictions IA")
    parser.add_argument("--camera-width", type=int, default=640, help="Largeur demandee pour la webcam")
    parser.add_argument("--camera-height", type=int, default=480, help="Hauteur demandee pour la webcam")
    parser.add_argument("--camera-fps", type=int, default=15, help="FPS demandes pour la webcam")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        source=args.source,
        show=not args.no_window,
        infer_every=args.infer_every,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
    )
