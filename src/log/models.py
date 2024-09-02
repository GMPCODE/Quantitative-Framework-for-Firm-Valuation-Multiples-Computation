from dataclasses import dataclass

from utils.utils import EnhancedStrEnum


class LogType(EnhancedStrEnum):
    CRITICAL: str = "CRITICAL"
    DEBUG: str = "DEBUG"
    ERROR: str = "ERROR"
    INFO: str = "INFORMATION"
    WARN: str = "WARNING"


@dataclass
class Log:
    PackageName: str
    ModuleName: str
    Level: LogType
    Message: str
