from typing import Any

import pandas as pd
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

from log.log import Logger
from portfolio.datasets.bloomberg.schemas import (
    AvailableRawColumns as BAvailableRawColumns,
)
from portfolio.datasets.yahoo_finance.schemas import (
    AvailableRawColumns as YFAvailableRawColumns,
)
from portfolio.optimisation.schemas import OptimizationOptions


class MarkowitzOptimization:
    def __init__(self, tickers: list[str], data_frames: list[pd.DataFrame]) -> None:
        self.logger = Logger("markowitz_optimization")
        self.dataset: pd.DataFrame = pd.DataFrame()

        self.efficient_frontier: EfficientFrontier | None = None

        for i in range(len(tickers)):
            columns = data_frames[i].columns
            if YFAvailableRawColumns.CLOSE_PRICE in columns:
                self.dataset[tickers[i]] = data_frames[i][
                    YFAvailableRawColumns.CLOSE_PRICE
                ]
            elif BAvailableRawColumns.CLOSE_PRICE in columns:
                self.dataset[tickers[i]] = data_frames[i][
                    BAvailableRawColumns.CLOSE_PRICE
                ]
            else:
                self.logger.warn(
                    "Unable to properly identify the Close Price Column in the current"
                    " Data Frame"
                )
                continue

        self.compute_efficient_frontier()

    def compute_efficient_frontier(
        self,
    ) -> tuple[Any]:
        self.mean = expected_returns.mean_historical_return(self.dataset)
        self.covarariance_matrix = risk_models.sample_cov(self.dataset)

        self.efficient_frontier = EfficientFrontier(self.mean, self.covarariance_matrix)

    def optimize(
        self,
        optimization_mode: OptimizationOptions = OptimizationOptions.SHARPE_RATIO,
    ) -> dict:
        match optimization_mode:
            case OptimizationOptions.SHARPE_RATIO:
                return self.efficient_frontier.max_sharpe()
            case OptimizationOptions.QUADRATIC_UTILITY:
                return self.efficient_frontier.max_quadratic_utility()
            case _:
                return {}

    def clean(
        self,
    ) -> None:
        if self.efficient_frontier:
            self.efficient_frontier.clean_weights()
