import logging
import cv2
from source.orchestrator.streamer import Streamer

import argparse

def test_streamer(url: str):
    logging.basicConfig(level=logging.INFO)

    # 1. Başlatma
    stream = Streamer(url=url, buffer_size=5)
    stream.start()

    # 2. Kareleri Okuma
    try:
        while True:
            frame = stream.read_frame()
            if frame is not None:
                cv2.imshow("Canli Yayin", frame)
            elif not stream.is_alive():
                break

            wait_time = max(1, int(1000 / stream.fps))
            if cv2.waitKey(wait_time) & 0xFF == ord("q"):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StreamIngestor Video File Test")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    args = parser.parse_args()

    test_streamer(args.video_path)
