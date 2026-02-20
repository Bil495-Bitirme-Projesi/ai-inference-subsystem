from queue import Queue
from .streamer import Streamer


class StreamIngestor:
    def __init__(self, preprocessor, seq_buffer, engine, dispatcher):
        self.streamer = Streamer()
        self.frame_buffer = Queue()
        self.preprocessor = preprocessor
        self.sequence_buffer = seq_buffer
        self.inference_engine = engine
        self.dispatcher = dispatcher
        self._running = False

    def start_capture(self):
        self._running = True
        self.streamer.run()

    def _process_loop(self):
        while self._running:
            raw_frame = self.frame_buffer.get()
            processed = self.preprocessor.process(raw_frame)
            self.sequence_buffer.add_frame(processed)

            if self.sequence_buffer.is_ready():
                seq = self.sequence_buffer.get_sequence()
                results = self.inference_engine.predict(seq)
                self.dispatcher.dispatch(results, {}, raw_frame)
