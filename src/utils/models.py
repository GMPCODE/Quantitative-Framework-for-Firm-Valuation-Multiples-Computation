from dataclasses import dataclass
from enum import IntEnum

from utils.utils import EnhancedStrEnum


class DiscreteNumericAction(IntEnum):
    BUY: int = 1
    SELL: int = 0


class DisceteAction(EnhancedStrEnum):
    BUY: str = "buy"
    SELL: str = "sell"


class Position(IntEnum):
    LONG: int = 1
    SHORT: int = 0


@dataclass
class TradingEnvironmentSettings:
    initial_capital: float
    cost_commission: float
    tickers: list[str]
    source: ...
