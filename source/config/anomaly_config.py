from abc import ABC
from dataclasses import dataclass


@dataclass
class AnomalyConfig(ABC):
    model_path: str


@dataclass
class VideoMAEAnomalyConfig(AnomalyConfig):
    sequence_length: int = 16
    stride: int = 1

@dataclass
class VLMConfig(AnomalyConfig):
    sequence_length: int = 16
    stride: int = 4
    use_potential_filter: bool = True
    use_memory: bool = True
    enable_caching: bool = False
    log_level: str = "INFO"
