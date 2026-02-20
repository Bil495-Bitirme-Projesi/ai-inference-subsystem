from abc import ABC
from dataclasses import dataclass

@dataclass
class AnomalyConfig(ABC):
    model_path: str

@dataclass
class LSTMAnomalyConfig(AnomalyConfig):
    sequence_length: int = 10
    stride: int = 1