import base64
import hmac
import hashlib
import json
import logging
import os
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class CoinbaseClient:
    """Simple REST client for Coinbase Advanced Trade sandbox."""

    def __init__(self) -> None:
        load_dotenv()
        self.api_key = os.getenv("CB_API_KEY")
        self.api_secret = os.getenv("CB_API_SECRET")
        self.passphrase = os.getenv("CB_PASSPHRASE")
        self.base_url = os.getenv(
            "CB_BASE_URL", "https://api-public.sandbox.exchange.coinbase.com"
        )

        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.session.headers.update(
            {
                "CB-ACCESS-KEY": self.api_key or "",
                "CB-ACCESS-PASSPHRASE": self.passphrase or "",
                "Content-Type": "application/json",
            }
        )

        if not all([self.api_key, self.api_secret, self.passphrase]):
            logger.warning("Coinbase API credentials are not fully set.")

    # ------------------------------------------------------------------
    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Create CB-ACCESS-SIGN header value."""
        secret = (self.api_secret or "").encode()
        key = base64.b64decode(secret)
        message = f"{timestamp}{method.upper()}{path}{body}"
        hmac_digest = hmac.new(key, message.encode(), hashlib.sha256).digest()
        return base64.b64encode(hmac_digest).decode()

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
        auth: bool = True,
    ):
        url = f"{self.base_url}{path}"
        data = json.dumps(body) if body else ""
        headers = self.session.headers.copy()

        if auth:
            timestamp = str(time.time())
            headers.update(
                {
                    "CB-ACCESS-TIMESTAMP": timestamp,
                    "CB-ACCESS-SIGN": self._sign(timestamp, method, path, data),
                }
            )

        try:
            response = self.session.request(
                method,
                url,
                params=params,
                data=data if body else None,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger.error("Request failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    def get_ticker(self, product_id: str = "BTC-USD"):
        """Get ticker information for a product."""
        path = f"/products/{product_id}/ticker"
        return self._request("GET", path, auth=True)

    def place_market_order(
        self,
        product_id: str,
        side: str,
        size_usd: Optional[float] = None,
        size_asset: Optional[float] = None,
    ):
        """Place a market order using size in USD or asset units."""
        if (size_usd is None) == (size_asset is None):
            raise ValueError("Specify exactly one of size_usd or size_asset")

        body = {"product_id": product_id, "side": side, "type": "market"}
        if size_usd is not None:
            body["funds"] = str(size_usd)
        if size_asset is not None:
            body["size"] = str(size_asset)

        return self._request("POST", "/orders", body=body, auth=True)

    def get_accounts(self):
        """Retrieve all accounts."""
        return self._request("GET", "/accounts", auth=True)


__all__ = ["CoinbaseClient"]
