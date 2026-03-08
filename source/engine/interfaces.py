from abc import ABC, abstractmethod


class IInferenceEngine(ABC):
    @abstractmethod
    def init(self, config):
        pass

    @abstractmethod
    def predict(self, sequence_tensor):
        pass
