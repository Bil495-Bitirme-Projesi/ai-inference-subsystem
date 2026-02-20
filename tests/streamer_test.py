import logging
import cv2
from source.orchestrator.streamer import Streamer


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

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    url = "rtsp://localhost:8554/test_yayin"
    test_streamer(url)
