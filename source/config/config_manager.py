import json

class ConfigManager:
    @staticmethod
    def load(path: str, config_class):
        with open(path, 'r') as f:
            data = json.load(f)
        return config_class(**data)