from dataclasses import dataclass, field
from datetime import date

from utils.utils import EnhancedStrEnum


class AvailableTechincalIndicators(EnhancedStrEnum):
    SMA: str = "SMA"
    WMA: str = "WMA"
    EMA: str = "EMA"


class TimeFrames(EnhancedStrEnum):
    MINUTES_1: str = "1m"
    MINUTES_2: str = "2m"
    MINUTES_5: str = "5m"
    MINUTES_15: str = "15m"
    MINUTES_30: str = "30m"
    MINUTES_60: str = "60m"
    MINUTES_90: str = "90m"
    HOUR_1: str = "1h"
    DAY_1: str = "1d"
    DAY_5: str = "5d"
    WEEK_1: str = "1wk"
    MONTH_1: str = "1mo"
    MONTH_3: str = "3mo"
    YEAR_1: str = "1y"


class DataSourceType(EnhancedStrEnum):
    BLOOMBERG: str = "BLOOMBERG"
    YAHOO_FINANCE: str = "YAHOO_FINANCE"


@dataclass
class DatasetMetadata:
    Ticker: str
    Origin: str
    FormatDate: str = field(default="%Y-%m-%d")

    Period: str = field(default=TimeFrames.YEAR_1)
    Interval: str = field(default=TimeFrames.DAY_1)

    SourceName: str | None = field(default=None)
    FileFormat: str | None = field(default=None)

    StartDate: str | date | None = field(default=None)
    EndDate: str | date | None = field(default=None)
