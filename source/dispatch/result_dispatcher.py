from engine.interfaces import IDispatcher


class ResultDispatcher(IDispatcher):
    def dispatch(self, detections, info, frame):
        # Sonuçları CommunicationModule'a veya loglara aktar
        print(f"Dispatching results: {detections}")
