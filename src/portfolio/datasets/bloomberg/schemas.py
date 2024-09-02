from utils.utils import EnhancedStrEnum


class AvailableRawColumns(EnhancedStrEnum):
    DATES: str = "Date"
    OPEN_PRICE: str = "PX_OPEN"
    HIGH_PRICE: str = "PX_HIGH"
    LOW_PRICE: str = "PX_LOW"
    CLOSE_PRICE: str = "PX_LAST"
    VOLUME: str = "PX_VOLUME"
    PUT_CALL_VOLUME_RATIO_CUR_DAY: str = "PUT_CALL_VOLUME_RATIO_CUR_DAY"
    VOLUME_TOTAL_CALL: str = "VOLUME_TOTAL_CALL"
    VOLUME_TOTAL_PUT: str = "VOLUME_TOTAL_PUT"
    OPEN_INT_TOTAL_CALL: str = "OPEN_INT_TOTAL_CALL"
    OPEN_INT_TOTAL_PUT: str = "OPEN_INT_TOTAL_PUT"
    RETURNS: str = "RETURNS"
