from dataclasses import dataclass
from datetime import datetime
from typing import Any

from utils.models import Position
from utils.utils import EnhancedStrEnum


class TradeResult(EnhancedStrEnum):
    PROFIT: str = "profit"
    LOSS: str = "loss"
    PENDING: str = "pending"


class TradeStatus(EnhancedStrEnum):

    CLOSED: str = "closed"
    PENDING: str = "pending"


@dataclass
class BacktestRecord:
    direction: Position
    entry_point: float
    entry_point_index: int
    take_profit: float
    stop_loss: float
    price_close: float
    price_close_index: int
    time_opening: datetime
    time_closing: datetime
    status: TradeStatus
    result: TradeResult


@dataclass
class BacktestRequest:
    starting_indexes: list[int]
    direction: list[Position]
    take_profits: list[float]
    stop_losses: list[float]
    dataset: Any
