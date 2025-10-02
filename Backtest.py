# Data Processing
import math
from decimal import Decimal
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from numpy import ndarray, dtype, float64, datetime64
from datetime import datetime

# Randomness
import random
random.seed(42)
import uuid

# Data Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 6]
import seaborn as sns
import mplfinance as mpf

# Data Handling
from heapq import heappush, heappop
from collections import defaultdict, deque

# Type Annotations
from typing import List, Tuple, Dict, Set, Optional, Literal, Deque, Any

# Data Classes
from dataclasses import dataclass, field, asdict


TAKER_FEE_RATE = 5.5e-4
MAKER_FEE_RATE = 2e-4

OrderStatus = Literal[
    # Active
    'New', # Limit order
    'Untriggered', # Conditional order
    'PartiallyFilled', 

    # Closed
    'Filled', 
    'Cancelled', 
    'Deactivated', # Cancelled before being triggered
    'Triggered' # Instantaneous state after 'Untriggered', before 'New' or 'Filled'
    'Amended' # Order has been amended and is no longer valid
]


def qty_to_value(category: str, avgPrice: float, qty: float) -> float:
    """
    Convert quantity to value based on the category and average price.
    """
    return qty * avgPrice if category == 'linear' else qty / avgPrice


def get_settleCoin(category: str, symbol: str) -> str:
    """
    Get the settlement coin based on the category and symbol.
    """
    if category == 'linear':
        return 'USDT'
    elif category == 'inverse':
        return symbol[:-3]  # Remove 'USD' from the end of the symbol
    else:
        raise ValueError(f"Invalid category: {category}. Must be 'linear' or 'inverse'.")
    
    
def get_IM_rate(leverage: float) -> float:
    return round(1 / leverage, 3)


def get_MM_rate(leverage: float) -> float:
    return round(get_IM_rate(leverage) / 2, 3)
    
    
def get_estimated_fee(value: float, category: Literal['linear', 'inverse'], side: Literal['Buy', 'Sell'], leverage: float, include_open_position: bool):
    sign = 1 if (category == 'linear') ^ (side == 'Buy') else -1
    return value * (int(include_open_position) + 1 + sign / leverage) * TAKER_FEE_RATE


def get_orderIM(order: "Order", leverage: float) -> float:
    value = order.leavesValue
    return value * get_IM_rate(leverage) + get_estimated_fee(value, order.category, order.side, leverage, True)


def increase_position(order: "Order", position: "Position") -> bool:
    return order.orderStatus != 'Untriggered' and (position.side == '' or order.side == position.side)


class LiquidationError(Exception):
    """
    Exception raised when a position is liquidated due to insufficient margin or other liquidation conditions.
    """
    def __init__(self) -> None:
        super().__init__('Liquidated.')
    


@dataclass
class Order:
    category: Literal['linear', 'inverse']
    symbol: str
    orderId: str
    side: Literal['Buy', 'Sell']
    orderType: Literal['Market', 'Limit']
    qty: float
    lastPriceOnCreated: float
    price: Optional[float] = None
    orderLinkId: str = ''
    triggerDirection: Optional[Literal[1, 2]] = None
    triggerPrice: Optional[float] = None
    reduceOnly: bool = False
    orderStatus: Optional[str] = None
    leavesQty: Optional[float] = None
    settleCoin: Optional[str] = None
    avgPrice: Optional[float] = None

    def __post_init__(self):
        if self.category not in ['linear', 'inverse']:
            raise ValueError(f"Invalid category: {self.category}. Must be 'linear' or 'inverse'.")

        if not isinstance(self.symbol, str):
            raise TypeError(f"symbol must be a string, got {type(self.symbol).__name__}")
        
        if not (self.category == 'linear' and self.symbol.endswith('USDT')) and not (self.category == 'inverse' and self.symbol.endswith('USD')):
            raise ValueError(f"Invalid symbol for category {self.category}: {self.symbol}. Must end with 'USDT' for linear or 'USD' for inverse.")

        if not isinstance(self.orderId, str):
            raise TypeError(f"orderId must be a string, got {type(self.orderId).__name__}")
            
        if self.side not in ['Buy', 'Sell']:
            raise ValueError(f"Invalid order side: {self.side}. Order side must be 'Buy' or 'Sell'.")

        if self.orderType not in ['Market', 'Limit']:
            raise ValueError(f"Invalid order type: {self.orderType}. Order type must be 'Market' or 'Limit'.")

        if not isinstance(self.qty, (int, float)):
            raise TypeError(f"qty must be a number, got {type(self.qty).__name__}")
        if self.qty <= 0:
            raise ValueError(f"Invalid order quantity: {self.qty}. Order quantity must be greater than 0")

        if self.orderType == 'Limit' and self.price is None:
            raise ValueError("Limit orders must have a price specified.")
        
        if not isinstance(self.orderLinkId, str):
            raise TypeError(f"orderLinkId must be a string, got {type(self.orderLinkId).__name__}")

        if not isinstance(self.orderLinkId, str):
            raise TypeError(f"orderLinkId must be a string, got {type(self.orderLinkId).__name__}")
        if len(self.orderLinkId) > 45:
            raise ValueError(f"Order link ID is too long: {len(self.orderLinkId)} characters. Maximum length is 45 characters.")
        
        if self.triggerDirection is not None and self.triggerDirection not in [1, 2]:
            raise ValueError("triggerDirection must be 1 (price rise to `triggerPrice`) or 2 (price falls to `triggerPrice`).")
        
        if (self.triggerDirection is None) ^ (self.triggerPrice is None):
            raise ValueError("Both `triggerDirection` and `triggerPrice` must be set or unset together.")
        
        for attr in ['orderStatus', 'leavesQty', 'settleCoin', 'avgPrice']:
            if getattr(self, attr) is not None:
                raise ValueError(f"{attr} should not be provided during initialization.")
            
        self.orderStatus: OrderStatus = 'Untriggered' if self.triggerPrice is not None else 'New'
        
        self.leavesQty = self.qty
        
        self.settleCoin = get_settleCoin(self.category, self.symbol)
        
        self.avgPrice: float = 0.0
        self.node: Optional["OrderNode"] = None
        self.ob: Optional["OrderBook"] = None
        
    def id_dict(self) -> Dict[str, str]:
        return {'orderId': self.orderId, 'orderLinkId': self.orderLinkId}
    
    def asdict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d
    
    @property
    def is_taker(self) -> bool:
        if self.orderStatus == 'Untriggered':
            return False
        curPrice = self.triggerPrice if self.orderStatus == 'Triggered' else self.lastPriceOnCreated
        if (
            self.orderType == 'Market'
            or (self.side == 'Buy' and self.price > curPrice)
            or (self.side == 'Sell' and self.price < curPrice)
        ):
            return True
        return False
    
    
    @property
    def estimated_next_update_price(self) -> float:
        if self.orderStatus == 'Untriggered':
            return self.triggerPrice
        curPrice = self.triggerPrice if self.orderStatus == 'Triggered' else self.lastPriceOnCreated
        if (
            self.orderType == 'Market'
            or (self.side == 'Buy' and self.price > curPrice)
            or (self.side == 'Sell' and self.price < curPrice)
        ):
            return curPrice
        if self.orderStatus in ['New', 'Triggered']:
            return self.price
        # Closed
        return 0.0
        
        
    def _value(self, qty: float) -> float:
        if self.orderStatus == 'Untriggered':
            return 0.0
        return qty_to_value(self.category, self.estimated_next_update_price, qty)
        
        
    @property
    def value(self) -> float:
        return self._value(self.qty)
    
    
    @property
    def leavesValue(self) -> float:
        return self._value(self.leavesQty)
    
    
    @property
    def heap_key(self) -> float:
        """
        Returns:
            float: 
                - -inf for closed orders. This ensures they are always popped first, 
                    which prevents the heap from being clogged with closed orders, 
                    since we are using lazy deletion.
                - negative for bids (max-heap) 
                - positive for asks (min-heap)
        """
        if self.orderStatus in ['Cancelled', 'Deactivated', 'Amended']:
            return -math.inf
        
        p = self.estimated_next_update_price
        if self.orderStatus == 'Untriggered':
            return p if self.triggerDirection == 1 else -p
        return -p if self.side == 'Buy' else p
             

    # Used for heap operations
    def __lt__(self, other: "Order") -> bool:
        return self.heap_key < other.heap_key
    
    
    @property
    def ap_key(self) -> Tuple[float, str, str]:
        """
        Returns:
            Tuple:
                - price: The y value on klines plot
                - side: The color of the line
                - orderStatus: The line style
        """
        return (self.estimated_next_update_price, self.side, self.orderStatus)
    
    
@dataclass
class OrderNode:
    val: Optional[Order] = None
    prev: Optional["OrderNode"] = None
    next: Optional["OrderNode"] = None

    
class Position:
    def __init__(self, category: Literal['linear', 'inverse'], symbol: str, leverage: float):
        self.category = category
        self.symbol = symbol
        self.leverage = leverage
        
        self.side: Literal['Buy', 'Sell', ''] = ''
        self._size: float = 0.0
        self.avgPrice: float = 0.0
        self.markPrice: float = 0.0
        
        self.curRealisedPnl: float = 0.0
        self.cumRealisedPnl: float = 0.0
        
        self.settleCoin: str = get_settleCoin(category, symbol)


    @property
    def size(self) -> float:
        return self._size
    

    @size.setter
    def size(self, new_size: float) -> None:
        if new_size < self._size:
            # Reducing size, realising profits and losses
            self.curRealisedPnl += self.unrealisedPnl * (self._size - new_size) / self._size
        if new_size == 0:
            # Potition closed
            self.side = ''
            self.avgPrice = 0.0
            self.cumRealisedPnl += self.curRealisedPnl
            self.curRealisedPnl = 0.0
        self._size = new_size
    
        
    @property
    def unrealisedPnl(self) -> float:
        return (self.markPrice - self.avgPrice) * self.size * (1 if self.side == 'Buy' else -1)
    
    
    @property
    def realisedPnl(self) -> float:
        return self.curRealisedPnl + self.cumRealisedPnl
        
        
    @property
    def pnl(self) -> float:
        return self.unrealisedPnl + self.realisedPnl
    
    
    @property
    def positionValue(self) -> float:
        if self.category == 'linear':
            return self.size * self.markPrice
        return self.size / self.markPrice
    
    
    @property
    def entryValue(self) -> float:
        if self.avgPrice == 0:
            return 0.0
        if self.category == 'linear':
            return self.size * self.avgPrice
        return self.size / self.avgPrice
    

    @property
    def positionIM(self) -> float:
        return self.positionValue * get_IM_rate(self.leverage) + get_estimated_fee(self.entryValue, self.category, self.side, self.leverage, False)
    
    
    @property
    def positionMM(self) -> float:
        return self.positionValue * get_MM_rate(self.leverage) + get_estimated_fee(self.entryValue, self.category, self.side, self.leverage, False)
    
    
    def fill_order(self, order: Order, time_idx: int) -> List["Transaction"]:
        if order.avgPrice == 0:
            raise ValueError("Order must have a valid average price to be filled.")
        
        self.markPrice = order.avgPrice
        qty = order.leavesQty
        fee_rate = TAKER_FEE_RATE if order.is_taker else MAKER_FEE_RATE
        self.curRealisedPnl -= order.avgPrice * qty * fee_rate
        
        if increase_position(order, self):
            self.avgPrice = (self.avgPrice * self.size + order.avgPrice * qty) / (self.size + qty)
            self.size += qty
            self.side = order.side
            isReduce = False
        else:
            if order.leavesQty >= self.size:
                # Close the position first
                qty = self.size
            self.size -= qty
            isReduce = True
            
        transactions = [Transaction(
            orderId=order.orderId,
            orderLinkId=order.orderLinkId,
            side=order.side,
            qty=qty,
            price=order.avgPrice,
            isReduce=isReduce,
            time_idx=time_idx
        )]
            
        order.leavesQty -= qty
        if order.leavesQty > 0:
            # Fill the rest qty after closing the position
            order.orderStatus = 'PartiallyFilled'
            transactions.extend(self.fill_order(order, time_idx))
            
        order.orderStatus = 'Filled'
        
        return transactions
                
    
@dataclass
class Transaction:
    orderId: str
    side: Literal['Buy', 'Sell']
    qty: float
    price: float
    isReduce: bool
    time_idx: int
    orderLinkId: Optional[str] = None
    
    
class OrderBook:
    def __init__(self, ohlcv: ndarray, leverage: float):
        self.ohlcv = ohlcv # OHLCV data for the order book, shape: (n, 5), dtype: float64
        self.leverage = leverage
        self.time_idx: int = 0
        self.asks: List[Order] = [] # Limit sell orders (min-heap)
        self.bids: List[Order] = [] # Limit buy orders (max-heap)
        self.addplots: Dict[Tuple[float, str, str], ndarray[dtype[float64]]] = {} # Dict[(price, side, orderStatus), ndarray[price | nan]]
        self.orderIds: Set[str] = set() # Set[orderId]
        
        self.totalOrderIM: float = 0.0

    def place_order(self, order: Order) -> None:
        if order.heap_key < 0:
            heappush(self.bids, order)
        else:
            heappush(self.asks, order)
            
        self.orderIds.add(order.orderId)
        self.totalOrderIM += get_orderIM(order, self.leverage)
            
        key = order.ap_key
        if key not in self.addplots:
            self.addplots[key] = np.r_[
                np.full(self.time_idx, np.nan, dtype=float64), 
                np.full(self.ohlcv.shape[0]-self.time_idx, key[0], dtype=float64)
            ]
            
    def _mark_removed(self, order: Order) -> None:
        self.orderIds.remove(order.orderId)
        self.totalOrderIM -= get_orderIM(order, self.leverage)
        self.addplots[order.ap_key][self.time_idx:] = np.nan
    
    def cancel_order(self, order: Order) -> None:
        if order.orderId not in self.orderIds:
            raise ValueError(f"Order {order.orderId} not found in the order book.")
        self._mark_removed(order)
            
    def update(self) -> List[Order]:
        """
        Update the order book by removing orders that are filled at the given price.
        Returns a list of orders that were removed.
        """
        def check_bids(low: float) -> List[Order]:
            removed_orders = []
            # Remove bids (buy orders) that are below the current price
            while self.bids and self.bids[0].heap_key < -low:
                order = heappop(self.bids)
                if order.orderStatus in ['Untriggered', 'New']:
                    removed_orders.append(order)
            return removed_orders
        
        def check_asks(high: float) -> List[Order]:
            removed_orders = []
            # Remove asks (sell orders) that are above the current price
            while self.asks and self.asks[0].heap_key < high:
                order = heappop(self.asks)
                if order.orderStatus in ['Untriggered', 'New']:
                    removed_orders.append(order)
            return removed_orders
        
        removed_orders: List[Order] = []
        high, low, close = self.ohlcv[self.time_idx, 1:4]  # low, high, close prices for the current time index
        if high - close < close - low:
            removed_orders.extend(check_bids(low))
            removed_orders.extend(check_asks(high))
        else:
            removed_orders.extend(check_asks(high))
            removed_orders.extend(check_bids(low))
        
        self.time_idx += 1
        
        for order in removed_orders:
            self._mark_removed(order)
            
        return removed_orders
    
    
    def get_addplots(self) -> List[Dict[str, Series | DataFrame | ndarray | List]]:
        addplots = []
        for (_, side, orderStatus), arr in self.addplots.items():
            color = 'green' if side == 'Buy' else 'red'
            if orderStatus == 'New':
                linestyle = '-'
                orderType = 'Limit'
            else:
                linestyle = '--'
                orderType = 'Conditional'
                
            addplots.append(mpf.make_addplot(arr, panel=0, secondary_y=False, color=color, label=f'{orderType} {side}', linestyle=linestyle))

        return addplots
    
    
@dataclass
class CoinBalance:
    coin: str
    initialBalance: float
    positions: List[Position] = field(default_factory=list)
    orderbooks: List[OrderBook] = field(default_factory=list)
    
    @property
    def totalOrderIM(self) -> float:
        return sum(orderbook.totalOrderIM for orderbook in self.orderbooks)
    
    @property
    def totalPositionIM(self) -> float:
        return sum(position.positionIM for position in self.positions)
    
    @property
    def totalPositionMM(self) -> float:
        return sum(position.positionMM for position in self.positions)
    
    @property
    def unrealisedPnl(self) -> float:
        return sum(position.unrealisedPnl for position in self.positions)
    
    @property
    def cumRealisedPnl(self) -> float:
        return sum(position.realisedPnl for position in self.positions)
    
    @property
    def walletBalance(self) -> float:
        return self.initialBalance + self.cumRealisedPnl
    
    @property
    def equity(self) -> float:
        return self.walletBalance + self.unrealisedPnl
    
    @property
    def availableBalance(self) -> float:
        return self.equity - self.totalOrderIM - self.totalPositionIM - self.totalPositionMM
    
class BacktestServer:
    MAX_LEVERAGE = 100.0
    MAX_CLOSED_ORDERS = 500
    
    def __init__(
        self, 
        data:               Dict[str, Dict[str, DataFrame]], 
        balances:           Dict[str, float], 
        cross_leverages:    Optional[Dict[str, Dict[str, float]]] = None,
        data_utcoffset:     int = 0
    ) -> None:
        
        # (Deprecated) Data structure for storing OHLCV data
        self.data = data
        
        # Open times
        self.times: ndarray[dtype[datetime64]] = None

        # Category -> symbol -> ndarray[open, high, low, close, volume]
        self.ohlcv: Dict[str, Dict[str, ndarray[dtype[float64]]]] = defaultdict(dict)

        # Category -> symbol -> ATR values
        self.atrs:  Dict[str, Dict[str, float]]                   = defaultdict(dict)
        
        # List[Tuple[category, symbol]]
        self.cat_sym: List[Tuple[str, str]] = []
        
        if cross_leverages is None:
            cross_leverages = defaultdict(lambda: defaultdict(lambda: self.MAX_LEVERAGE))
        self.cross_leverages = cross_leverages

        for category in data:
            for symbol in data[category]:
                df = data[category][symbol]
                
                # Validate DataFrame structure
                if list(df.columns) != ['time', 'open', 'high', 'low', 'close', 'volume']:
                    raise ValueError(f"DataFrame for {category} - {symbol} must have columns: ['time', 'open', 'high', 'low', 'close', 'volume']")

                if self.times is None:
                    self.times = df['time'].values
                else:
                    # Ensure all times are the same
                    if not np.array_equal(self.times, df['time'].values):
                        raise ValueError(f"'time' mismatch in {category} - {symbol}")
                    
                # Convert DataFrame to ndarray
                ohlcv = df.values[:, 1:]
                self.ohlcv[category][symbol] = ohlcv
                
                # Compute ATR for signal position on klines plot
                high = ohlcv[:, 1]
                low = ohlcv[:, 2]
                close = ohlcv[:, 3]
                tr = np.maximum(high[1:] - low[1:], 
                                np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
                atr = np.mean(tr)
                self.atrs[category][symbol] = atr

                # Storing category and symbol combinations for easy access
                self.cat_sym.append((category, symbol))
                
                # Validate balances
                settleCoin = get_settleCoin(category, symbol)
                if settleCoin not in balances:
                    raise ValueError(f"Balance for settlement coin '{settleCoin}' not found in balances")
                
        # Adjust times to current UTC offset
        # Get current UTC offset in hours
        offset_hours = datetime.now().astimezone().utcoffset().total_seconds() / 3600 - data_utcoffset
        if offset_hours != 0:
            # times is datetime64, so convert offset to timedelta64
            offset = np.timedelta64(int(offset_hours * 3600), 's')
            self.times = self.times + offset
            
            for c, s in self.cat_sym:
                df = self.data[c][s]
                df['time'] = self.times
            
                
        # Data length
        self.n = len(self.times)
                
        # Store initial balances for reset and return computation
        self.initial_balances = balances
        
        # Time interval
        self.time_delta = self.times[1] - self.times[0]

        # Dummy node for linked list head
        self.dummy_node = OrderNode()
        self.dummy_node.next = self.dummy_node
        self.dummy_node.prev = self.dummy_node
        
        self.reset()
        
    def __len__(self) -> int:
        return self.n
        
    def reset(self) -> None:
        self.time_idx: int = 0
        self.at_close: bool = False
        
        # Coin -> CoinBalance
        self.balances: Dict[str, CoinBalance] = {}
        for coin, balance in self.initial_balances.items():
            self.balances[coin] = CoinBalance(
                coin=coin,
                initialBalance=balance,
                positions=[],
                orderbooks=[]
            )
        
        # Category -> symbol -> Position
        self.positions:         Dict[str, Dict[str, Position]]                      = defaultdict(dict)

        # Category -> symbol -> OrderBook
        self.orderbooks:        Dict[str, Dict[str, OrderBook]]                     = defaultdict(dict)

        # Category -> symbol -> OrderNode (linked list head)
        self.opened_order_head: Dict[str, Dict[str, OrderNode]]                     = defaultdict(dict)

        # Category -> symbol -> Deque[closed Orders]
        self.closed_orders:     Dict[str, Dict[str, Deque[Order]]]                  = defaultdict(dict)
        
        # Category -> symbol -> List[Transaction]
        self.transactions:      Dict[str, Dict[str, List[Transaction]]]             = defaultdict(dict)
        
        # Category -> symbol -> ndarray[return]
        self.cum_returns:       Dict[str, Dict[str, ndarray[dtype[float64]]]]       = defaultdict(dict)
        
        # Initialize
        for c, s in self.cat_sym:
            leverage = self.cross_leverages[c][s]
            self.positions[c][s] = Position(c, s, leverage=leverage)
            self.orderbooks[c][s] = OrderBook(ohlcv=self.ohlcv[c][s], leverage=leverage)
            self.opened_order_head[c][s] = self.dummy_node
            self.closed_orders[c][s] = deque(maxlen=self.MAX_CLOSED_ORDERS)
            self.transactions[c][s] = []
            self.cum_returns[c][s] = np.full(self.n, np.nan, dtype=float64)
            
            coin = self.positions[c][s].settleCoin
            self.balances[coin].positions.append(self.positions[c][s])
            self.balances[coin].orderbooks.append(self.orderbooks[c][s])
        
        # Id -> Order
        self.id_to_order: Dict[str, Order] = {}

        # Link Id -> Id
        self.linkId_to_id: Dict[str, str] = {}
        
    def create_order_id(self) -> str:
        # Random UUID with 36 characters
        return str(uuid.uuid4())
    
    def place_order(
        self, 
        category: Literal['linear', 'inverse'], 
        symbol: str, 
        side: Literal['Buy', 'Sell'],
        orderType: Literal['Market', 'Limit'],
        qty: float, 
        price: Optional[float] = None,
        triggerDirection: Optional[Literal[1, 2]] = None,
        triggerPrice: Optional[float] = None,
        orderLinkId: str = '',
        reduceOnly: Optional[bool] = False,
    ) -> Dict[str, Any]:
        """
        Args:
            category (Literal['linear', 'inverse']): `linear` for USDT perpetual, `inverse` for inverse perpetual
            symbol (str): Symbol name, like `BTCUSDT`, uppercase only
            side (Literal['Buy', 'Sell']): `Buy`, `Sell`
            orderType (Literal['Market', 'Limit']): `Market`, `Limit`
            qty (float): Order quantity
            price (Optional[float], optional): Order price. Market order will ignore this field.
            triggerDirection (Optional[Literal[1, 2]], optional): 
                Conditional order param. Used to identify the expected direction of the conditional order.
                - `1`: triggered when market price rises to `triggerPrice`
                - `2`: triggered when market price falls to `triggerPrice`
            triggerPrice (Optional[float], optional): Conditional order trigger price.
            orderLinkId (str, optional): User customised order ID. A max of 45 characters. Always unique.
            reduceOnly (Optional[bool], optional): `True` means your position can only reduce in size if this order is triggered.

        Returns:
            Dict[str, Any]:
                - orderId (str): System generated order ID
                - orderLinkId (str): User customised order ID
                - category (Literal['linear', 'inverse']): `linear` for USDT perpetual, `inverse` for inverse perpetual
                - symbol (str): Symbol name, like `BTCUSDT`, uppercase only
                - side (Literal['Buy', 'Sell']): `Buy`, `Sell`
                - orderType (Literal['Market', 'Limit']): `Market`, `Limit`
                - qty (float): Order quantity
                - price (Optional[float]): Order price. Market order will ignore this field.
                - triggerDirection (Optional[Literal[1, 2]]): 
                    Conditional order param. Used to identify the expected direction of the conditional order.
                    - `1`: triggered when market price rises to `triggerPrice`
                    - `2`: triggered when market price falls to `triggerPrice`
                - triggerPrice (Optional[float]): Conditional order trigger price.
                - reduceOnly (bool): `True` means your position can only reduce in size if this order is triggered.
                - lastPriceOnCreated (float): The last price when the order is created.
                - orderStatus (OrderStatus): 
                    - Active: 'New' for limit orders, 'Untriggered' for conditional orders, 'PartiallyFilled'
                    - Closed: 'Filled', 'Cancelled', 'Deactivated' for conditional orders cancelled before being triggered, 'Triggered' for instantaneous state after 'Untriggered' before 'New' or 'Filled', 'Amended' for orders that have been amended and are no longer valid
                - leavesQty (float): Remaining quantity to be filled
                - avgPrice (float): Average fill price
                - settleCoin (str): Settlement coin, like `USDT`, `BTC`
        """
        order = Order(
            category=category,
            symbol=symbol,
            orderId=self.create_order_id(),
            side=side,
            orderType=orderType,
            qty=qty,
            lastPriceOnCreated=self.get_price(category, symbol),
            price=price,
            orderLinkId=orderLinkId,
            triggerDirection=triggerDirection,
            triggerPrice=triggerPrice,
            reduceOnly=reduceOnly
        )
        
        node = OrderNode(val=order, prev=self.dummy_node)
        order.node = node
        
        self._place_order(order)
        return order.asdict()

    def _check_balance(self, coin: str, amount: float) -> None:
        available_balance = self.balances[coin].availableBalance
        if available_balance < amount:
            raise ValueError(f"Insufficient balance for {coin}. Required: {amount}, Available: {available_balance}")
    
    def _place_order(self, order: Order) -> None:
        """
        Place an order into the order book after validating it.
        """
        category = order.category
        symbol = order.symbol
        orderId = order.orderId
        orderLinkId = order.orderLinkId
        
        position = self.positions[category][symbol]
        orderbook = self.orderbooks[category][symbol]
        
        # Open price
        curPrice = self.get_price(category, symbol)
        position.markPrice = curPrice
        
        if order.reduceOnly and increase_position(order, position):
            raise ValueError(f"`reduceOnly` is True but order ({orderId=}, {orderLinkId=}) will increase position size.")
        
        # Validate triggerDirection
        if order.orderStatus == 'Untriggered':
            triggerDirection = order.triggerDirection
            triggerPrice = order.triggerPrice
            if (
                triggerDirection == 1 and triggerPrice < curPrice
                or triggerDirection == 2 and triggerPrice > curPrice
            ):
                raise ValueError(f'{triggerDirection=} but {triggerPrice=} {"<" if triggerDirection == 1 else ">"} {curPrice}.')
        
        # Check if there is enough balance to open the order
        if increase_position(order, position):
            settleCoin = order.settleCoin
            orderIM = get_orderIM(order, position.leverage)
            self._check_balance(settleCoin, orderIM)

        if order.orderStatus in ['Untriggered', 'New']:
            # Save orderId and orderLinkId
            if orderLinkId != '':
                if orderLinkId in self.linkId_to_id:
                    raise ValueError(f"Order link ID {orderLinkId} already exists.")
                self.linkId_to_id[orderLinkId] = orderId
            self.id_to_order[orderId] = order
            
            # Store the order in the link list
            self._add_node(order.node)

        if order.is_taker:
            self._fill_order(order)
        else:
            if order.orderStatus == 'Triggered':
                order.orderStatus = 'New'
            orderbook.place_order(order)
    
    def _fill_order(self, order: Order) -> None:
        """Fill the order immediately at the estimated next update price."""
        category = order.category
        symbol = order.symbol
        
        order.avgPrice = order.estimated_next_update_price
            
        position = self.positions[category][symbol]

        # Update mark price to the fill price
        position.markPrice = order.avgPrice
        
        if increase_position(order, position):
            positionMM = order.leavesValue * get_MM_rate(position.leverage) + get_estimated_fee(order.leavesValue, category, order.side, position.leverage, False)
            self._check_balance(order.settleCoin, positionMM)
            
        transactions = position.fill_order(order, self.time_idx)

        self.transactions[category][symbol].extend(transactions)
        
        self._remove_node(order.node)
                        
    def _add_node(self, node: OrderNode) -> None:
        """ Add a node to the head of the linked list."""
        order = node.val
        category = order.category
        symbol = order.symbol
        node.next = self.opened_order_head[category][symbol]
        node.next.prev = node
        self.opened_order_head[category][symbol] = node
               
    def _remove_node(self, node: OrderNode) -> None:
        """ Remove a node from the linked list and add to closed orders if not amended."""
        if node is self.dummy_node:
            return
        
        order = node.val
        if node is self.opened_order_head[order.category][order.symbol]:
            self.opened_order_head[order.category][order.symbol] = node.next
        node.prev.next = node.next
        node.next.prev = node.prev
        node = self.dummy_node  # Clear the node reference to help garbage collection
        if order.orderStatus != 'Amended':
            self.closed_orders[order.category][order.symbol].append(order)
        
    def _node_to_list(self, category: str, symbol: str) -> List[Order]:
        """ Convert the linked list to a list of orders."""
        node = self.opened_order_head[category][symbol]
        orders = []
        while node is not self.dummy_node:
            orders.append(node.val)
            node = node.next
        return orders
    
    def amend_order(
        self, 
        category: str,
        symbol: str,
        orderId: Optional[str] = None,
        orderLinkId: Optional[str] = None,
        triggerPrice: Optional[float] = None,
        qty: Optional[float] = None,
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        if orderId is None and orderLinkId is None:
            raise ValueError('Either `orderId` or `orderLinkId` is required.')
        
        old_order = self._get_open_orders(category, symbol, orderId, orderLinkId)[0]
        if old_order.orderStatus not in ['New', 'Untriggered']:
            raise ValueError(f"Only unfilled order can be amended. Current status: {old_order.orderStatus}")
        
        if (
            (triggerPrice is None or triggerPrice == old_order.triggerPrice) 
            and (qty is None or qty == old_order.qty)
            and (price is None or price == old_order.price)
        ):
            raise ValueError('Order is not amended.')
        
        # New order with the same id and other args except the amended ones
        new_order = Order(
            category=category,
            symbol=symbol,
            orderId=old_order.orderId,
            side=old_order.side,
            orderType=old_order.orderType,
            qty=qty if qty is not None else old_order.qty,
            lastPriceOnCreated=self.get_price(category, symbol),
            price=price if price is not None else old_order.price,
            orderLinkId=old_order.orderLinkId,
            triggerDirection=old_order.triggerDirection,
            triggerPrice=triggerPrice if triggerPrice is not None else old_order.triggerPrice,
            reduceOnly=old_order.reduceOnly
        )
        node = OrderNode(val=new_order, prev=self.dummy_node)
        new_order.node = node
    
        self.orderbooks[category][symbol].cancel_order(old_order)
        old_order.orderStatus = 'Amended'
        self._remove_node(old_order.node)
        del self.id_to_order[old_order.orderId]
        if old_order.orderLinkId != '':
            del self.linkId_to_id[old_order.orderLinkId]
        
        self._place_order(new_order)
        return new_order.asdict()
    
    def cancel_order(
        self, 
        category: str, 
        symbol: str, 
        orderId: Optional[str] = None, 
        orderLinkId: Optional[str] = None
    ) -> Dict[str, Any]:
        order = self._get_open_orders(category, symbol, orderId, orderLinkId)[0]
        self.orderbooks[category][symbol].cancel_order(order)
        order.orderStatus = 'Deactivated' if order.orderStatus == 'Untriggered' else 'Cancelled'
        self._remove_node(order.node)
        return order.asdict()
    
    def _get_open_orders(
        self, 
        category: str, 
        symbol: str, 
        orderId: Optional[str] = None, 
        orderLinkId: Optional[str] = None, 
        openOnly: int = 0
    ) -> List[Order]:
        
        if orderId is not None:
            if orderId in self.id_to_order:
                return [self.id_to_order[orderId]]
            else:
                return []
        
        if orderLinkId is not None:
            if orderLinkId in self.linkId_to_id:
                orderId = self.linkId_to_id[orderLinkId]
                return [self.id_to_order[orderId]]
            else:
                return []
            
        if openOnly == 0:
            return self._node_to_list(category, symbol)
        return list(self.closed_orders[category][symbol])[::-1]
    
    def get_open_orders(
        self, 
        category: str, 
        symbol: str, 
        orderId: Optional[str] = None, 
        orderLinkId: Optional[str] = None, 
        openOnly: int = 0
    ) -> List[Dict[str, Any]]:
        orders = self._get_open_orders(category, symbol, orderId, orderLinkId, openOnly)
        return [order.asdict() for order in orders]
    
    def get_leverage(self, category: str, symbol: str) -> float:
        return self.positions[category][symbol].leverage
    
    def get_server_time(self) -> dtype[datetime64]:
        return self.times[self.time_idx] + int(self.at_close) * self.time_delta
    
    def get_price(self, category: str, symbol: str) -> float:
        return self.ohlcv[category][symbol][self.time_idx, 0 if not self.at_close else 3]
                
    def cancel_all_orders(self, category: str, symbol: str) -> None:
        node = self.opened_order_head[category][symbol]
        while node != self.dummy_node:
            order = node.val
            next_node = node.next
            self.cancel_order(category, symbol, orderId=order.orderId)
            node = next_node
            
    def close_position(self, category: str, symbol: str) -> Dict[str, str]:
        position = self.positions[category][symbol]
        if position.size == 0:
            return
        
        side = 'Sell' if position.side == 'Buy' else 'Buy'
        return self.place_order(
            category=category,
            symbol=symbol,
            side=side,
            orderType='Market',
            qty=position.size,
            reduceOnly=True
        )
    
    def _update_orders(self) -> None:
        for c, s in self.cat_sym:
            # Update order book with current price
            removed_orders = self.orderbooks[c][s].update()
            for order in removed_orders:
                if order.orderStatus == 'Untriggered':
                    order.orderStatus = 'Triggered'
                    self._place_order(order)
                else: # New
                    self._fill_order(order)
                    
    def _update_positions(self) -> None:
        for c, s in self.cat_sym:
            position = self.positions[c][s]
            position.markPrice = self.get_price(c, s)
            settleCoin = position.settleCoin
            if self.balances[settleCoin].availableBalance < 0:
                raise LiquidationError()
            self.cum_returns[c][s][self.time_idx] = position.pnl / self.initial_balances[settleCoin]

    def next(self) -> bool:
        self.at_close = True
        self._update_orders()
        self._update_positions()
        
        # Reach the end of the data
        if self.time_idx >= self.n - 1:
            for category, symbol in self.cat_sym:
                self.cancel_all_orders(category, symbol)
                self.close_position(category, symbol)
            self._update_positions()
                
            return False
        
        self.time_idx += 1
        self.at_close = False
        return True
        
    def run_to_end(self) -> None:
        for _ in range(self.n - self.time_idx - int(self.at_close)):
            self.next()
    
    def show_klines(self) -> None:
        """Show the klines plot with buy/sell markers, conditional/limit orders, and cumulative returns."""
        for category, symbol in self.cat_sym:
            df = self.data[category][symbol].set_index('time')
            atr = self.atrs[category][symbol]
            highs = df['high'].values
            lows = df['low'].values
            
            addplots = self.orderbooks[category][symbol].get_addplots()
            
            buy_arr = np.full(self.n, np.nan, dtype=float64)
            sell_arr = np.full(self.n, np.nan, dtype=float64)
            
            mul = 0.5
            for transaction in self.transactions[category][symbol]:
                if transaction.side == 'Buy':
                    buy_arr[transaction.time_idx] = lows[transaction.time_idx] - atr * mul
                else:
                    sell_arr[transaction.time_idx] = highs[transaction.time_idx] + atr * mul
                    
            addplots.extend([
                mpf.make_addplot(buy_arr, type='scatter', markersize=100, marker='^', color='green', label='Buy', panel=0, secondary_y=False),
                mpf.make_addplot(sell_arr, type='scatter', markersize=100, marker='v', color='red', label='Sell', panel=0, secondary_y=False)
            ])
            
            cum_returns = self.cum_returns[category][symbol]
            addplots.append(mpf.make_addplot(cum_returns, panel=1, ylabel='Return'))
            
            fig, axlist = mpf.plot(
                df, type='candle', style='yahoo', panel_ratios=(2, 1), returnfig=True, 
                xlabel=f'Time ({df.index[0].year})', ylabel='Price ($)', volume=False,
                figsize=(16, 9), addplot=addplots
            )
            # Add legends to the main price panel (panel 0)
            handles, labels = axlist[0].get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            handles = list(unique.values())
            labels = list(unique.keys())
            if handles:
                axlist[0].legend(handles, labels, loc='best')
            
            fig.suptitle(f"{symbol}", fontsize=20, x=0.55)
            # Make panels visually separated and add thin borders
            for ax in axlist:
                ax.spines['top'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.spines['right'].set_visible(True)
                for spine in ax.spines.values():
                    spine.set_linewidth(1.2)
                    spine.set_color('black')
                ax.set_facecolor('white')
            # Add space between panels
            fig.subplots_adjust(hspace=0.25)
            mpf.show()
            
