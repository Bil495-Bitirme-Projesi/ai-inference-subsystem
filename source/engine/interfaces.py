from abc import ABC, abstractmethod


class IInferenceEngine(ABC):
    @abstractmethod
    def predict(self, sequence_tensor):
        pass
