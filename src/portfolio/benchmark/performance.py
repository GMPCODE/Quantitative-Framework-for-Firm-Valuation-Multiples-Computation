import numpy as np
from numpy.typing import NDArray
from scipy import stats


class PortfolioPerformer:
    def __init__(self, returns: NDArray) -> None:
        self.returns = returns

    def mean(
        self,
    ) -> float:
        return self.returns.mean()

    def mode(
        self,
    ) -> float:
        return stats.mode(self.returns)

    def min(
        self,
    ) -> float:
        return self.returns.min()

    def first_quartile(
        self,
    ) -> float:
        return np.percentile(self.returns, 25)

    def median(
        self,
    ) -> float:
        return np.median(self.returns)

    def third_quartile(
        self,
    ) -> float:
        return np.percentile(self.returns, 75)

    def max(
        self,
    ) -> float:
        return self.returns.max()

    def variance(
        self,
    ) -> float:
        return self.returns.var()

    def standard_deviation(
        self,
    ) -> float:
        return self.returns.std()

    def upside_standard_deviation(
        self,
    ) -> float:
        return self.returns[self.returns > 0].std()

    def downside_standard_deviation(
        self,
    ) -> float:
        return self.returns[self.returns < 0].std()

    def sharpe_ratio(
        self,
        risk_free: float = 0.0,
    ) -> float:
        return (self.returns.mean() - risk_free) / (self.returns - risk_free).std()

    def sortino_ratio(
        self,
        risk_free: float = 0.0,
    ) -> float:
        return (self.returns.mean() - risk_free) / self.downside_standard_deviation()

    def skewness(
        self,
    ) -> float:
        return stats.skew(self.returns)

    def kurtosis(
        self,
    ) -> float:
        return stats.kurtosis(self.returns)
