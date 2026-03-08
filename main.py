import argparse
import time
import os
import sys

# Proje kök dizinini ve 'source' dizinini path'e ekleyelim
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "source"))

# Engine'lerin register olması için import edilmesi gerekir
try:
    from engine.videomae_engine import VideoMAEAnomalyEngine
except ImportError as e:
    print(f"[WARNING] VideoMAE engine could not be imported: {e}")

from orchestrator.stream_ingestor import StreamIngestor
from proc.preproc import Preprocessor
from proc.sequence_buf import SequenceBuffer
from engine.inference_factory import InferenceFactory
from dispatch.result_dispatcher import ResultDispatcher

def main():
    parser = argparse.ArgumentParser(description="AI Inference Subsystem")
    parser.add_argument("--url", type=str, default="sample_data/Assault001_x264.mp4", 
                        help="Path to video file or RTSP stream URL")
    parser.add_argument("--config", type=str, default="config/videomae_cfg.json", 
                        help="Path to inference configuration JSON")
    parser.add_argument("--model", type=str, default="VideoMAE", 
                        help="Model type to use (default: VideoMAE)")
    parser.add_argument("--window", type=int, default=16, 
                        help="Sequence window length (default: 16)")
    parser.add_argument("--stride", type=int, default=16, 
                        help="Inference stride (default: 16, means non-overlapping)")
    parser.add_argument("--buffer", type=int, default=32, 
                        help="Streamer buffer size")
    args = parser.parse_args()

    # Config kontrolü
    if not os.path.exists(args.config):
        print(f"[ERROR] Configuration file not found: {args.config}")
        return

    print("="*50)
    print("   AI INFERENCE SUBSYSTEM   ")
    print("="*50)
    print(f"[*] Stream Source : {args.url}")
    print(f"[*] Model Type    : {args.model}")
    print(f"[*] Config Path   : {args.config}")
    print(f"[*] Window/Stride : {args.window} / {args.stride}")
    
    # 1. Bileşenlerin Başlatılması
    print("[*] Initializing components...")
    preprocessor = Preprocessor()
    
    # Pencere uzunluğu ve atlama (stride) değerlerini ayarla
    seq_buffer = SequenceBuffer(sequence_length=args.window, stride=args.stride)
    
    print("[*] Loading inference engine (this may take a while)...")
    try:
        engine = InferenceFactory.create(args.model, args.config)
    except Exception as e:
        print(f"[ERROR] Failed to create inference engine: {e}")
        return
    
    dispatcher = ResultDispatcher()
    
    # 2. Ingestor Oluşturulması
    ingestor = StreamIngestor(
        url=args.url,
        preprocessor=preprocessor,
        seq_buffer=seq_buffer,
        engine=engine,
        dispatcher=dispatcher,
        buffer_size=args.buffer
    )
    
    # 3. Yakalama Başlatılması
    print("[*] Starting capture thread...")
    ingestor.start_capture()
    
    # 4. Ana Döngü - İstatistikleri İzle
    try:
        print("[*] System is running. Press Ctrl+C to stop.")
        while ingestor.is_alive():
            stats = ingestor.get_stats()
            if stats['connected']:
                status = "CONNECTED"
                # FPS bilgisi 0 gelirse (henüz kare okunmadıysa) bekle
                fps_info = f"FPS: {stats['fps']:.2f}" if stats['fps'] > 0 else "FPS: --"
                print(f"\r[{status}] {fps_info} | Total Frames: {stats['frames']}   ", end="")
            else:
                if stats['frames'] > 0:
                   print(f"\r[DRAINING] Stream ended. Processing remaining frames... ({stats['frames']})   ", end="")
                else:
                   # Henüz bağlanamadıysa veya kapandıysa
                   print(f"\r[WAITING] Connecting to stream...   ", end="")
            
            time.sleep(1)
            
            # Eğer streamer kapandıysa ve ingestor hala alive ise kuyruk bitince çıkacak zaten
            # Ama manuel bir kontrol gerekirse buraya eklenebilir
            
    except KeyboardInterrupt:
        print("\n\n[!] Stopping system...")
    finally:
        ingestor.stop_capture()
        # İpliklerin düzgünce kapanması için kısa bir bekleme
        time.sleep(0.5)
        print("\n[+] System shutdown complete.")
        print("="*50)

if __name__ == "__main__":
    main()