from dataclasses import dataclass
from typing import TypeAlias

from utils.utils import EnhancedStrEnum

ComparisonElement: TypeAlias = dict[str, int | float | str]


class Elements(EnhancedStrEnum):
    ELEMENT: str = "ELEMENT"
    PERIOD: str = "PERIOD"

    CLOSE_PRICE: str = "Close"


class Relation(EnhancedStrEnum):
    GREATER: str = ">"
    GREATER_THAN: str = ">="
    LESS: str = "<"
    LESS_THAN: str = "<="
    EQUAL: str = "=="


class SamplingMethods(EnhancedStrEnum):
    NORMAL_DISTRIBUTION: str = "Normal Sampling"
    LAPLACE_DISTRIBUTION: str = "Laplace Sampling"
    UNIFORM_DISTRIBUTION: str = "Uniform Sampling"


@dataclass
class Filter:
    first_element: ComparisonElement
    second_element: ComparisonElement
    relation: Relation | None
