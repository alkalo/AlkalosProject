"""Notification helpers for Telegram and email.

Messages are sent using credentials defined in a `.env` file.  If
credentials are missing the message is silently ignored."""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables from the project `.env`
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


def notify(message: str) -> None:
    """Send ``message`` via Telegram and/or email.

    The function checks for the presence of credentials in the environment
    and sends the notification through any configured channel.  Missing
    credentials are ignored without raising an exception.
    """

    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:  # pragma: no cover - best effort notification
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat_id, "text": message},
                timeout=10,
            )
        except Exception:
            pass

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "0"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    email_to = os.getenv("EMAIL_TO")
    if all([smtp_host, smtp_port, smtp_user, smtp_pass, email_to]):
        msg = EmailMessage()
        msg["Subject"] = "Bot notification"
        msg["From"] = smtp_user
        msg["To"] = email_to
        msg.set_content(message)
        try:  # pragma: no cover - best effort notification
            with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
        except Exception:
            pass
