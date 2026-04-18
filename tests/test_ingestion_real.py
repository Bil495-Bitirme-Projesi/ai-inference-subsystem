import sys
import os
import argparse
import time
from source.orchestrator.stream_ingestor import StreamIngestor
from source.proc.preproc import Preprocessor
from source.proc.sequence_buf import SequenceBuffer
from source.engine.inference_factory import InferenceFactory
from source.dispatch.result_dispatcher import ResultDispatcher

# Add the project root and source to sys.path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root)
sys.path.append(os.path.join(root, "source"))


def test_real_ingestion(url: str):
    print("--- Starting Real Ingestion Test ---")
    print(f"Stream URL: {url}")
    
    # 1. Initialize Components
    preprocessor = Preprocessor()
    
    # Let's use 16 frames as per VideoMAE default
    seq_buffer = SequenceBuffer(sequence_length=16)
    
    # Load Engine via Factory (uses config/videomae_cfg.json by default or provided path)
    config_path = "config/videomae_cfg.json"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found!")
        return
        
    print("Loading Inference Engine (this might take a few seconds)...")
    engine = InferenceFactory.create("VideoMAE", config_path)
    
    dispatcher = ResultDispatcher()
    
    # 2. Create StreamIngestor
    ingestor = StreamIngestor(
        url=url,
        preprocessor=preprocessor,
        seq_buffer=seq_buffer,
        engine=engine,
        dispatcher=dispatcher,
        buffer_size=16
    )
    
    # 3. Start Capture
    print("Starting capture thread...")
    ingestor.start_capture()
    
    # Wait for streamer to actually start
    print("Waiting for stream to connect...")
    max_wait = 5
    start_wait = time.time()
    while not ingestor.streamer.running and (time.time() - start_wait) < max_wait:
        time.sleep(0.1)

    if not ingestor.streamer.running:
        print("\nError: Failed to start stream within timeout.")
        ingestor.stop_capture()
        return

    # 4. Monitor & Display (Optional display if needed, but ingestor runs in thread)
    try:
        while ingestor.is_alive():
            stats = ingestor.get_stats()
            print(f"\r- Frames: {stats['frames']} | FPS: {stats['fps']} | Connected: {stats['connected']}   ", end="")
            time.sleep(1)
            
            # Check if streamer finished but ingestor still working (normal for files)
            if not ingestor.streamer.running and ingestor.is_alive():
                # Fine, draining queue...
                pass
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("\nStopping capture...")
        ingestor.stop_capture()
        # Give some time for threads to clean up
        time.sleep(1)
        print("Test finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Stream Ingestion Test")
    parser.add_argument("url", type=str, help="Path to video file or RTSP stream URL")
    args = parser.parse_args()
    
    test_real_ingestion(args.url)
