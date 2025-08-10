from src.utils.notify import notify


def test_notify_no_credentials(monkeypatch):
    """notify should not raise when credentials are missing"""
    for var in [
        "TELEGRAM_TOKEN",
        "TELEGRAM_CHAT_ID",
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_USER",
        "SMTP_PASS",
        "EMAIL_TO",
    ]:
        monkeypatch.delenv(var, raising=False)

    notify("test message")
