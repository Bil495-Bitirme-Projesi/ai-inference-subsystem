import sys
import os
import time
import argparse
from unittest.mock import MagicMock

# Add source to path
sys.path.append(os.path.abspath("source"))

from orchestrator.stream_ingestor import StreamIngestor
from proc.preproc import Preprocessor
from proc.sequence_buf import SequenceBuffer
from dispatch.result_dispatcher import ResultDispatcher

def main():
    parser = argparse.ArgumentParser(description="StreamIngestor Video File Test")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence length for buffer")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
        sys.exit(1)

    print(f"Starting test with video: {args.video_path}")

    # Initialize components
    preprocessor = Preprocessor()
    seq_buffer = SequenceBuffer(sequence_length=args.seq_len)
    
    # Mock Engine to avoid heavy dependencies in a simple orchestration test
    engine = MagicMock()
    engine.predict.side_effect = lambda x: print(f"  [Engine] Inferred sequence: {x.shape}") or 0.5
    
    dispatcher = ResultDispatcher()
    
    # Initialize Ingestor
    # Streamer is initialized inside StreamIngestor
    ingestor = StreamIngestor(preprocessor, seq_buffer, engine, dispatcher)
    
    # Set the video URL in the streamer
    ingestor.streamer.url = args.video_path
    ingestor.streamer.name = "VideoTestStreamer"

    print("Starting ingestor thread...")
    ingestor.start_capture()

    start_time = time.time()
    try:
        while ingestor.is_alive():
            stats = ingestor.get_stats()
            print(f"Status: Connected={stats['connected']}, Frames={stats['frames']}, FPS={stats['fps']}", end="\r")
            time.sleep(1)
            
            # Optional: Timeout for very long videos if desired
            # if time.time() - start_time > 60: break
            
    except KeyboardInterrupt:
        print("\nStopping test...")
    finally:
        ingestor.stop_capture()
        ingestor.join(timeout=5)
        print("\nTest completed.")
        final_stats = ingestor.get_stats()
        print(f"Final Statistics: {final_stats}")

if __name__ == "__main__":
    main()
