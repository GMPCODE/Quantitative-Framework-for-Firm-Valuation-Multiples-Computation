from enum import StrEnum

import tensorflow as tf
import torch
from torch.backends import mps


class EnhancedStrEnum(StrEnum):
    @classmethod
    def has(
        cls,
        key: str,
    ) -> bool:
        return key in cls.__members__.values()

    @classmethod
    def list(cls) -> list[str]:
        return list(cls.__members__.values())


def is_cuda_available():
    return torch.cuda.is_available()


def is_mps_available():
    return mps.is_available()


def get_device(torch: bool = False):
    if is_cuda_available():
        return torch.device(
            "cuda",
        )
    elif is_mps_available():
        return torch.device(
            "mps",
        )
    else:
        return torch.device(
            "cpu",
        )


def get_gpu_devices() -> list:
    return tf.config.experimental.list_physical_devices("GPU")
