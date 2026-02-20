from .interfaces import IInferenceEngine

class LSTMAnomalyEngine(IInferenceEngine):
    def __init__(self):
        self._config = None
        self._model = None

    def init(self, config):
        self._config = config
        # Model yükleme mantığı buraya gelir
        print(f"LSTM Model loaded from {config.model_path}")

    def predict(self, sequence_tensor):
        # Gerçek çıkarım (inference) kodu
        return [0.05] # Örnek anomali skoru