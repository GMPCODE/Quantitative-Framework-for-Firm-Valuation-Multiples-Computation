from dataclasses import dataclass, field
from datetime import date


@dataclass
class PortfolioPerformance:
    OptimizationModel: str = field(default="NA")
    OptimizationMode: str = field(default="NA")
    Tickers: list[str] = field(default="NA")
    Weights: list[float] = field(default="NA")
    Mean: float = field(default=0.0)
    Mode: float = field(default=0.0)
    Min: float = field(default=0.0)
    FirstQuartile: float = field(default=0.0)
    Median: float = field(default=0.0)
    ThirdQuartile: float = field(default=0.0)
    Max: float = field(default=0.0)
    Var: float = field(default=0.0)
    StdDev: float = field(default=0.0)
    UpsideStdDev: float = field(default=0.0)
    DownsideStdDev: float = field(default=0.0)
    SharpeRatio: float = field(default=0.0)
    SortinoRatio: float = field(default=0.0)
    Skewness: float = field(default=0.0)
    Kurtosis: float = field(default=0.0)
    StartDate: date = field(default=date(1900, 1, 1))
    EndTime: date = field(default=date(1900, 1, 1))
    ActivityTime: date = field(default=date(1900, 1, 1))
