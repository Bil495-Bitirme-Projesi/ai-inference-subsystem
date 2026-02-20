from abc import ABC, abstractmethod

class IDispatcher(ABC):
    @abstractmethod
    def dispatch(self, detections, info, frame): pass