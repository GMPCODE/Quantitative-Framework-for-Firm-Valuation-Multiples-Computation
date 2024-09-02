from utils.utils import EnhancedStrEnum


class AvailableRawColumns(EnhancedStrEnum):
    DATES: str = "Dates"
    OPEN_PRICE: str = "Open"
    HIGH_PRICE: str = "High"
    LOW_PRICE: str = "Low"
    CLOSE_PRICE: str = "Close"
    RETURNS: str = "Returns"
