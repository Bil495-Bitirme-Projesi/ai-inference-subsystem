from engine.lstm_engine import LSTMAnomalyEngine
from config.config_manager import ConfigManager
from config.anomaly_config import LSTMAnomalyConfig

class InferenceFactory:
    @staticmethod
    def create(model_type: str, config_path: str):
        if model_type == "LSTM":
            config = ConfigManager.load(config_path, LSTMAnomalyConfig)
            engine = LSTMAnomalyEngine()
            engine.init(config)
            return engine
        raise ValueError("Unknown model type")