from __future__ import annotations

import asyncio
import os
import smtplib
import time
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Optional

from telegram import Bot

from config import (
    COOLDOWN_SECONDS,
    EMAIL_PASSWORD,
    EMAIL_RECEIVER,
    EMAIL_SENDER,
    EMAIL_SMTP_PORT,
    EMAIL_SMTP_SERVER,
    TELEGRAM_CHAT_ID,
    TELEGRAM_TOKEN,
)


@dataclass
class AlertPayload:
    score: float
    capture_path: Optional[str] = None
    message: str = "Alerte SIRCH : comportement violent detecte"


class AlertManager:
    def __init__(self, cooldown_seconds: int = COOLDOWN_SECONDS) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time = 0.0

    def can_send(self) -> bool:
        return (time.time() - self.last_alert_time) >= self.cooldown_seconds

    def send_alerts(self, payload: AlertPayload) -> bool:
        if not self.can_send():
            return False
        self.last_alert_time = time.time()
        self.play_sound()
        self.send_email(payload)
        self.send_telegram(payload)
        return True

    def play_sound(self) -> None:
        try:
            import winsound

            winsound.Beep(1200, 500)
        except Exception:
            pass

    def send_email(self, payload: AlertPayload) -> None:
        if not EMAIL_PASSWORD or EMAIL_PASSWORD == "ton_mot_de_passe_application_gmail":
            return

        msg = EmailMessage()
        msg["Subject"] = "Alerte SIRCH"
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg.set_content(f"{payload.message}\nScore : {payload.score:.3f}")

        if payload.capture_path and os.path.exists(payload.capture_path):
            with open(payload.capture_path, "rb") as capture_file:
                msg.add_attachment(
                    capture_file.read(),
                    maintype="image",
                    subtype="jpeg",
                    filename=os.path.basename(payload.capture_path),
                )

        with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)

    def send_telegram(self, payload: AlertPayload) -> None:
        token = (TELEGRAM_TOKEN or "").strip()
        chat_id = (TELEGRAM_CHAT_ID or "").strip()
        if token == "" or chat_id == "":
            return
        if token == "METS_TON_TOKEN_ICI" or chat_id == "METS_TON_CHAT_ID_ICI":
            return
        try:
            asyncio.run(self._send_telegram_async(payload))
        except Exception:
            return

    async def _send_telegram_async(self, payload: AlertPayload) -> None:
        bot = Bot(token=TELEGRAM_TOKEN)
        text = f"{payload.message}\nScore : {payload.score:.3f}"
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
        if payload.capture_path and os.path.exists(payload.capture_path):
            with open(payload.capture_path, "rb") as capture_file:
                await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=capture_file)
