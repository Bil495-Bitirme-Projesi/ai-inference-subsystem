from .streamer import Streamer
from threading import Thread
import torch


class StreamIngestor(Thread):
    def __init__(
        self, url, preprocessor, seq_buffer, engine, dispatcher, buffer_size=10
    ):
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
        fps = self.streamer.fps if self.streamer.fps > 0 else 30.0
        
        while self.streamer.running or not self.streamer.frame_queue.empty():
            data = self.streamer.read_frame()
            if data is None:
                # Gecikmeli bağlantı veya bitiş kontrolü
                if not self.streamer.running:
                    break
                continue

            frame_id, raw_frame = data
            
            processed = self.preprocessor.process(raw_frame)
            self.sequence_buffer.add_frame(frame_id, processed)

            if self.sequence_buffer.is_ready():
                seq_data = self.sequence_buffer.get_sequence()
                
                # İndeksleri ve kareleri ayır
                indices = [item[0] for item in seq_data]
                frames = [item[1] for item in seq_data]
                
                # Zaman aralığını hesapla
                start_sec = indices[0] / fps
                end_sec = indices[-1] / fps

                # List[np.ndarray] -> torch.Tensor (T, H, W, C)
                seq_tensor = torch.stack([torch.from_numpy(f) for f in frames])
                # (T, H, W, C) -> (T, C, H, W) -> (B, T, C, H, W)
                seq_tensor = seq_tensor.permute(0, 3, 1, 2).unsqueeze(0)

                results = self.inference_engine.predict(seq_tensor)
                
                meta_info = {
                    "start_sec": round(start_sec, 2),
                    "end_sec": round(end_sec, 2),
                    "start_frame": indices[0],
                    "end_frame": indices[-1]
                }
                
                self.dispatcher.dispatch(results, meta_info, raw_frame)
