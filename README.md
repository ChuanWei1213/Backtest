# Backtest
`BactestServer` simulates Bybit API by replacing real price data with historical klines data, and use method `next()` to simulate time movement.
Cross margin is used, USDT perpeptual contract (`'linear'`) and inverse perpeptual contract (`'inverse'`) are supported.

# `BacktestServer`
- `__init__()`:
  - `data`: Klines data.  
    Example:
    ```Python
    data = {
        'linear': {
            'BTCUSDT': DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume']),
            'ETHUSDT': DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        },
        'inverse': {
            'BTCUSD': DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume']),
            'ETHUSD': DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        }
    }
    ```
  - `balances`: Initial balance for all coins.
    Example:
    ```Python
    balances = {
        'USDT': 10000,
        'BTC': 1,
    }
    ```
  - `cross_leverages`: The leverage for all symbols. Optional. If not specified, `BacktestServer.MAX_LEVERAGE` is used.
    Example:
    ```Python
    cross_leverage = {
        'linear': {
            'BTCUSDT': 100,
            'BNBUSDT': 50,
    }
    ```
  - `data_utcoffset`: The UTC offset the user would like the times to be showned.
 
- `place_order()`: Order creation.
  Example:
  ```Python
  place_order('linear', 'BTCUSDT', 'Buy', 'Market', 0.01, triggerDirection=1, triggerPrice=125000, reduceOnly=True)
  ```

- `amend_order()`: Amend active order.
- `cancel_order()`: Cancel active order.
- `get_open_orders()`: Primarily query for active orders but also support recent 500 closed status orders.
- `cancel_all_orders()`: Cancell all active orders.
- `close_position()`: Close position with market order.
- `next()`: Moves to the next kline.
- `run_to_end()`: Run through all the klines.
- `show_klines()`: Displays a klines plot with trading signals, active order levels, and return curve. This gives user a visual understanding about the trading process.

# To Run
## Check Python installation
```
python3 --version
```

## Create a virtual environment
```
python3 -m venv venv
```

## Activate the environment
- macOS/Linux:
  ```
  source venv/bin/activate
  ```
- Windows (PowerShell):
  ```
  venv\Scripts\Activate.ps1
  ```

- Windows (cmd.exe):
  ```
  venv\Scripts\activate.bat
  ```

## Install packages
```
pip install -r requirements.txt
```

## Run
Modify the code in `backtest.ipynb` and run.
