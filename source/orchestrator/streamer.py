import numpy as np
import cv2
from queue import Queue, Empty, Full
from typing import Callable, Optional
import threading
import logging
import time


class Streamer:
    def __init__(
        self,
        name: str = "DefaultStreamer",
        url: str = None,
        max_retries: int = 5,
        frame_callback: Optional[Callable] = None,
        buffer_size: int = 10,
    ):
        super().__init__(name=name, daemon=True)
        self.url = url
        self.max_retries = max_retries
        self.frame_callback = frame_callback
        self.buffer_size = buffer_size

        # Akış Kontrolü
        self.cap = None
        self.frame_queue = Queue(maxsize=buffer_size)
        self.running = False
        self.connected = False

        self.logger = logging.getLogger(self.name)

        # İstatistikler
        self.stats = {"frames_received": 0, "errors": 0, "start_time": None}

    def run(self):
        self.logger.info(f"Stream thread starting: {self.url}")
        self.running = True
        self.stats["start_time"] = time.time()

        retry_count = 0
        while self.running and retry_count < self.max_retries:
            if self._connect_stream():
                retry_count = 0  # Başarılı bağlantıda sayacı sıfırla
                self._stream_loop()
            else:
                retry_count += 1
                self.logger.warning(
                    f"Bağlantı denemesi {retry_count}/{self.max_retries}"
                )
                time.sleep(2)

        self._clean_up()
        self.logger.info("Stream thread stopped.")

    def _connect_stream(self) -> bool:
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            self.connected = False
            return False

        self.connected = True
        self.logger.info(f"Bağlantı başarılı: {self.url}")
        return True

    def _stream_loop(self):
        """Görüntü yakalama döngüsü."""
        while self.running and self.connected:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error("Kare okunamadı, bağlantı kopmuş olabilir.")
                self.connected = False
                break

            self._frame_to_queue(frame)
            self.stats["frames_received"] += 1

            if self.frame_callback:
                try:
                    self.frame_callback(frame)
                except Exception as e:
                    self.logger.error(f"Callback hatası: {e}")

    def _frame_to_queue(self, frame):
        """Kuyruk doluysa en eski kareyi atıp yenisini ekler (Real-time önceliği)."""
        try:
            if self.frame_queue.full():
                self.frame_queue.get_nowait()  # Eski kareyi çıkar
            self.frame_queue.put_nowait(frame)
        except (Empty, Full):
            pass

    def read_frame(self) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get(timeout=1.0)
        except Empty:
            return None

    def stop(self):
        self.running = False
        self.join(timeout=2)

    def _clean_up(self):
        if self.cap:
            self.cap.release()
        self.connected = False

        # Kuyruğu temizle
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break

    def get_stats(self):
        uptime = (
            time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
        )
        return {
            "uptime": round(uptime, 2),
            "frames": self.stats["frames_received"],
            "fps": round(self.stats["frames_received"] / uptime, 2)
            if uptime > 0
            else 0,
            "connected": self.connected,
        }
