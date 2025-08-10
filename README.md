# AlkalosProject

Simple Coinbase Advanced Trade sandbox REST client.

## Setup

1. Create a `.env` file using `.env.example` as a template.
2. Install dependencies:
   ```bash
   pip install requests python-dotenv
   ```

## Usage

```python
from coinbase_client import CoinbaseClient

client = CoinbaseClient()
print(client.get_ticker("BTC-USD"))
```
