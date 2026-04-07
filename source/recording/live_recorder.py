"""
LiveRecorder — Anomali süresi boyunca kareleri diske canlı yazan kaydedici.

RecordBuffer'dan bağımsız olarak çalışır:
  1. Anomali başladığında → pre-event kareleri RecordBuffer'dan alınıp dosyaya yazılır
  2. Olay süresince     → gelen her kare doğrudan dosyaya eklenir
  3. Olay bittiğinde    → dosya kapatılır, klip hazırdır

Bu sayede anomali ne kadar uzun sürerse sürsün kare kaybı olmaz.
Ring buffer kapasitesine bağımlılık ortadan kalkar.

Kullanım:
    recorder = LiveRecorder(output_dir="clips", fps=30.0)
    recorder.start(pre_frames, filename="cam5_evt_100.mp4")
    recorder.add_frame(frame_id, raw_frame)   # her kare için
    clip_path = recorder.finalize()           # olay bittiğinde
"""

import os
import logging
import tempfile
from typing import List, Optional, Tuple

import cv2
import numpy as np


class LiveRecorder:
    """
    Canlı video klip kaydedici.

    start() ile kayda başlar, add_frame() ile kare ekler,
    finalize() ile kaydı bitirir ve dosya yolunu döndürür.
    """

    def __init__(self, output_dir: Optional[str] = None, fps: float = 30.0):
        """
        Args:
            output_dir: Kliplerin yazılacağı dizin.
                        Verilmezse sistem temp dizini kullanılır.
            fps:        Varsayılan kare hızı.
        """
        self.output_dir = output_dir or os.path.join(tempfile.gettempdir(), "ais_clips")
        self.fps = fps
        self.logger = logging.getLogger("LiveRecorder")

        os.makedirs(self.output_dir, exist_ok=True)

        # Aktif kayıt durumu
        self._writer: Optional[cv2.VideoWriter] = None
        self._filepath: Optional[str] = None
        self._frame_count: int = 0
        self._recording: bool = False

    # ------------------------------------------------------------------ #
    #  Durum
    # ------------------------------------------------------------------ #

    @property
    def is_recording(self) -> bool:
        """Canlı kayıt aktif mi kontrol eder."""
        return self._recording

    # ------------------------------------------------------------------ #
    #  Yaşam döngüsü: start → add_frame → finalize
    # ------------------------------------------------------------------ #

    def start(
        self,
        pre_frames: List[Tuple[int, np.ndarray]],
        filename: Optional[str] = None,
        fps: Optional[float] = None,
    ) -> bool:
        """
        Pre-event kareleri yazarak canlı kaydı başlatır.

        Args:
            pre_frames: RecordBuffer'dan alınan [(frame_id, raw_frame), ...] listesi.
            filename:   Klip dosya adı (opsiyonel).
            fps:        Bu klip için FPS (opsiyonel).

        Returns:
            Başarılı ise True.
        """
        if self._recording:
            self.logger.warning("Recording already in progress, ignoring start().")
            return False

        clip_fps = fps or self.fps

        # Dosya adı belirle
        if filename is None:
            tag = pre_frames[0][0] if pre_frames else 0
            filename = f"live_{tag}.mp4"
        self._filepath = os.path.join(self.output_dir, filename)

        # Pre-frame yoksa kayıt durumunu aç, writer'ı ilk add_frame'de oluştur
        if not pre_frames:
            self.logger.warning("No pre-event frames; writer deferred to first add_frame().")
            self._recording = True
            self._frame_count = 0
            return True

        # İlk kareden boyut bilgisi al ve VideoWriter oluştur
        _, sample = pre_frames[0]
        success = self._open_writer(sample, clip_fps)
        if not success:
            return False

        # Pre-event karelerini yaz
        for _, frame in pre_frames:
            self._writer.write(frame)

        self._frame_count = len(pre_frames)
        self._recording = True
        self.logger.info(
            f"Live recording started: {self._filepath} "
            f"({self._frame_count} pre-event frames written)"
        )
        return True

    def add_frame(self, frame_id: int, raw_frame: np.ndarray) -> None:
        """
        Canlı kaydedilen klibe yeni kare ekler.
        Olay ve post-event süresince her kare için çağrılmalıdır.

        Args:
            frame_id:  Kare sıra numarası.
            raw_frame: Ham (BGR, uint8) numpy dizisi.
        """
        if not self._recording:
            return

        # Writer henüz yoksa (pre_frames boştu), şimdi oluştur
        if self._writer is None:
            if not self._open_writer(raw_frame, self.fps):
                self._recording = False
                return

        self._writer.write(raw_frame)
        self._frame_count += 1

    def finalize(self) -> Optional[str]:
        """
        Kaydı bitirir ve dosya yolunu döndürür.

        Returns:
            Klip dosya yolu (str), veya aktif kayıt yoksa None.
        """
        if not self._recording:
            self.logger.warning("No active recording to finalize.")
            return None

        self._close_writer()
        self._recording = False

        duration = self._frame_count / self.fps if self.fps > 0 else 0
        self.logger.info(
            f"Live recording finalized: {self._filepath} "
            f"({self._frame_count} frames, {duration:.1f}s)"
        )

        path = self._filepath
        self._filepath = None
        self._frame_count = 0
        return path

    def abort(self) -> None:
        """Aktif kaydı iptal eder ve geçici dosyayı siler."""
        self._close_writer()

        if self._filepath and os.path.exists(self._filepath):
            try:
                os.remove(self._filepath)
                self.logger.info(f"Aborted recording removed: {self._filepath}")
            except OSError:
                pass

        self._recording = False
        self._filepath = None
        self._frame_count = 0

    # ------------------------------------------------------------------ #
    #  Dahili yardımcılar
    # ------------------------------------------------------------------ #

    def _open_writer(self, sample_frame: np.ndarray, fps: float) -> bool:
        """VideoWriter'ı açar. Başarı durumunu döndürür."""
        h, w = sample_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self._filepath, fourcc, fps, (w, h))

        if not self._writer.isOpened():
            self.logger.error(f"Failed to open VideoWriter: {self._filepath}")
            self._writer = None
            return False
        return True

    def _close_writer(self) -> None:
        """VideoWriter'ı güvenli şekilde kapatır."""
        if self._writer:
            self._writer.release()
            self._writer = None
