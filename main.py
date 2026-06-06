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


def save_capture(frame) -> str:
    os.makedirs(CAPTURES_FOLDER, exist_ok=True)
    filename = datetime.now().strftime("capture_%Y%m%d_%H%M%S.jpg")
    path = os.path.join(CAPTURES_FOLDER, filename)
    cv2.imwrite(path, frame)
    return path


def run(source: str = "0", show: bool = True) -> None:
    init_db()
    buffer = FrameSequenceBuffer()
    inference = ViolenceInference()
    decision = DecisionEngine()
    alerts = AlertManager()

    with VideoSource(source) as video:
        print("SIRCH demarre. Appuie sur q pour quitter.")
        while True:
            frame = video.read()
            if frame is None:
                break

            sequence = buffer.add_frame(frame)
            if sequence is not None:
                score = inference.predict(sequence)
                settings = get_settings()
                decision.configure(settings["threshold"], settings["k_window"])
                is_violence, mean_score = decision.update(score)
                if is_violence:
                    capture_path = save_capture(frame)
                    add_incident(score, mean_score, capture_path, str(source))
                    alerts.send_alerts(AlertPayload(score=mean_score, capture_path=capture_path))

            if show:
                cv2.imshow("SIRCH", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIRCH - detection de violence en temps reel")
    parser.add_argument("--source", default="0", help="Webcam index, flux RTSP/IP ou fichier video")
    parser.add_argument("--no-window", action="store_true", help="Desactive la fenetre OpenCV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(source=args.source, show=not args.no_window)
