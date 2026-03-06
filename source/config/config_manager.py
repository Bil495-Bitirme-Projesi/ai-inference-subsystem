import json
from .anomaly_config import AnomalyConfig

class ConfigManager:
    @staticmethod
    def load(path: str, config_class) -> AnomalyConfig:
        with open(path, 'r') as f:
            data = json.load(f)
        return config_class(**data)