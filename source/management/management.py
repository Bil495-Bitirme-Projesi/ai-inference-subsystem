from factory.inference_factory import InferenceFactory


class IngestorManager:
    def __init__(self):
        self.active_ingestors = {}
        self.factory = InferenceFactory()

    def start_thread(self, camera_id, config_path):
        engine = self.factory.create("LSTM", config_path)
        # Ingestor oluştur ve thread başlat
        print(f"Started monitor for {camera_id}")
