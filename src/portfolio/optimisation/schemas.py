from utils.utils import EnhancedStrEnum


class OptimizationOptions(EnhancedStrEnum):
    SHARPE_RATIO: str = "SHARPE_RATIO"
    QUADRATIC_UTILITY: str = "QUADRATIC_UTILITY"
