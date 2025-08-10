"""Minimal client for Coinbase Exchange sandbox API.

This module wraps a subset of the Coinbase Exchange REST API suitable for
simple trading bots.  It handles authentication using credentials stored in a
`.env` file and performs automatic retries when the exchange enforces rate
limits.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional

import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Ensure environment variables from `.env` are loaded
from src.utils import env as env_utils  # noqa: F401  # pragma: no cover


class ExchangeClient:
    """Client wrapper for the Coinbase Exchange sandbox API.

    Parameters
    ----------
    api_key, api_secret, passphrase:
        API credentials.  If omitted, the values are loaded from the
        environment variables ``COINBASE_API_KEY``, ``COINBASE_API_SECRET`` and
        ``COINBASE_PASSPHRASE``.
    base_url:
        Base URL of the API.  Defaults to Coinbase's public sandbox endpoint.
    max_retries:
        Number of retry attempts for rate limiting or transient errors.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        passphrase: Optional[str] = None,
        *,
        base_url: str | None = None,
        max_retries: int = 5,
    ) -> None:
        self.api_key = api_key or os.getenv("COINBASE_API_KEY")
        self.api_secret = api_secret or os.getenv("COINBASE_API_SECRET")
        self.passphrase = passphrase or os.getenv("COINBASE_PASSPHRASE")
        if not all([self.api_key, self.api_secret, self.passphrase]):
            raise ValueError("Missing Coinbase API credentials")

        self.base_url = base_url or os.getenv(
            "COINBASE_API_URL", "https://api-public.sandbox.exchange.coinbase.com"
        )
        self.max_retries = max_retries

        self.session: Session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            raise_on_status=False,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sign(self, method: str, request_path: str, body: str = "") -> Dict[str, str]:
        """Return authentication headers for ``method`` and ``request_path``."""

        timestamp = str(time.time())
        message = timestamp + method.upper() + request_path + body
        secret = base64.b64decode(self.api_secret)
        signature = hmac.new(secret, message.encode(), hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest()).decode()
        return {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature_b64,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Send a request to the Coinbase API with retry handling."""

        url = self.base_url + path
        body = json.dumps(data) if data else ""
        headers = self._sign(method, path, body)

        for attempt in range(self.max_retries):
            response = self.session.request(
                method, url, params=params, data=body or None, headers=headers
            )
            if response.status_code == 429:  # Rate limited
                wait = float(response.headers.get("Retry-After", "1"))
                time.sleep(wait)
                continue
            if response.status_code >= 500:  # transient server error
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            if response.content:
                return response.json()
            return None

        response.raise_for_status()
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_price(self, product_id: str) -> float:
        """Return the latest trade price for ``product_id``."""

        data = self._request("GET", f"/products/{product_id}/ticker")
        return float(data["price"])

    def get_balance(self, currency: Optional[str] = None) -> Any:
        """Return account balances.

        If ``currency`` is provided, only the balance for that currency is
        returned.  Otherwise a dictionary mapping currencies to balances is
        produced.
        """

        accounts = self._request("GET", "/accounts")
        if currency is not None:
            for acc in accounts:
                if acc["currency"] == currency:
                    return float(acc["balance"])
            return 0.0
        return {acc["currency"]: float(acc["balance"]) for acc in accounts}

    def place_order(
        self,
        product_id: str,
        side: str,
        size: float,
        *,
        order_type: str = "market",
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Place an order on the exchange.

        Parameters
        ----------
        product_id:
            Trading pair identifier, e.g. ``"BTC-USD"``.
        side:
            ``"buy"`` or ``"sell"``.
        size:
            Order size in base currency.
        order_type:
            ``"market"`` or ``"limit"``.
        price:
            Required when ``order_type`` is ``"limit"``.
        """

        order: Dict[str, Any] = {
            "product_id": product_id,
            "side": side,
            "type": order_type,
            "size": str(size),
        }
        if order_type == "limit" and price is not None:
            order["price"] = str(price)
            order["time_in_force"] = "GTC"
        return self._request("POST", "/orders", data=order)


__all__ = ["ExchangeClient"]
