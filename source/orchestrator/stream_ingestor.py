from queue import Queue
from .streamer import Streamer
from threading import Thread   
import torch

class StreamIngestor(Thread):
    def __init__(self, url, preprocessor, seq_buffer, engine, dispatcher, buffer_size=10):
        super().__init__(daemon=True)
        self.url = url
        self.streamer = Streamer(url=url, buffer_size=buffer_size)
        self.preprocessor = preprocessor
        self.sequence_buffer = seq_buffer
        self.inference_engine = engine
        self.dispatcher = dispatcher

    def run(self):
        self.streamer.start()
        self._process_loop()

    def start_capture(self):
        self.start()
    
    def stop_capture(self):
        self.streamer.stop()

    def get_stats(self):
        return self.streamer.get_stats()

    def _process_loop(self):
        while self.streamer.running or not self.streamer.frame_queue.empty():
            raw_frame = self.streamer.read_frame()
            if raw_frame is None:
                # Eğer streamer durduysa ve kuyruk boşsa (read_frame timeout olduysa) çık
                if not self.streamer.running:
                    break
                continue
                
            processed = self.preprocessor.process(raw_frame)
            self.sequence_buffer.add_frame(processed)

            if self.sequence_buffer.is_ready():
                seq_list = self.sequence_buffer.get_sequence()
                
                # List[np.ndarray] -> torch.Tensor (T, H, W, C)
                seq_tensor = torch.stack([torch.from_numpy(f) for f in seq_list])
                # (T, H, W, C) -> (T, C, H, W) -> (B, T, C, H, W)
                seq_tensor = seq_tensor.permute(0, 3, 1, 2).unsqueeze(0)
                
                results = self.inference_engine.predict(seq_tensor)
                self.dispatcher.dispatch(results, {}, raw_frame)
