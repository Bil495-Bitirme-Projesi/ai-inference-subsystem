import sys
import os
import time
from unittest.mock import MagicMock

# Mock torch and transformers if not present
try:
    import torch
except ImportError:
    mock_torch = MagicMock()
    mock_torch.from_numpy = lambda x: MagicMock()
    mock_torch.stack = lambda x: MagicMock()
    mock_torch.no_grad = MagicMock
    sys.modules["torch"] = mock_torch

try:
    import transformers
except ImportError:
    mock_transformers = MagicMock()
    sys.modules["transformers"] = mock_transformers

import numpy as np

# Add source to path
sys.path.append(os.path.abspath("source"))

from orchestrator.streamer import Streamer
from orchestrator.stream_ingestor import StreamIngestor
from proc.preproc import Preprocessor
from proc.sequence_buf import SequenceBuffer
from dispatch.result_dispatcher import ResultDispatcher

def test_pipeline():
    print("Starting pipeline test...")
    
    # Mock dependencies
    preprocessor = Preprocessor()
    seq_buffer = SequenceBuffer(sequence_length=5)
    
    engine = MagicMock()
    engine.predict.return_value = 0.95
    
    dispatcher = MagicMock()
    
    # Initialize Ingestor
    ingestor = StreamIngestor(preprocessor, seq_buffer, engine, dispatcher)
    
    # Manually push some frames to streamer's queue since we don't have a real camera
    for _ in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ingestor.streamer._frame_to_queue(frame)
    
    # We don't want to start the streamer thread as it will try to connect to None
    ingestor.streamer.running = True
    
    print("Running process loop for a few iterations...")
    # Run process loop in a way we can stop it
    import threading
    
    def run_ingestor():
        try:
            ingestor._process_loop()
        except Exception as e:
            print(f"Error in process loop: {e}")

    t = threading.Thread(target=run_ingestor, daemon=True)
    t.start()
    
    time.sleep(2)
    ingestor.streamer.running = False
    t.join(timeout=1)
    
    print(f"Engine predict call count: {engine.predict.call_count}")
    print(f"Dispatcher dispatch call count: {dispatcher.dispatch.call_count}")
    
    if engine.predict.call_count > 0:
        print("Verification SUCCESS: Frames processed and predicted.")
    else:
        print("Verification FAILED: No frames processed.")

if __name__ == "__main__":
    test_pipeline()
