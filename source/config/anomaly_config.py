from abc import ABC
from dataclasses import dataclass


@dataclass
class AnomalyConfig(ABC):
    model_path: str


@dataclass
class VideoMAEAnomalyConfig(AnomalyConfig):
    sequence_length: int = 16
    stride: int = 1
