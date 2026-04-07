"""
ClipRecorder — Kare listesinden MP4 video klip dosyası oluşturur.

RecordBuffer'dan çıkarılan ham kareler bu sınıfa verilerek
diske .mp4 olarak yazılır, ardından MinIO'ya yüklenebilir.

Kullanım:
    recorder = ClipRecorder(output_dir="clips", fps=30.0)
    path = recorder.record(frames, filename="cam5_evt42.mp4")
"""

import os
import logging
import tempfile
from typing import List, Optional, Tuple

import cv2
import numpy as np


class ClipRecorder:
    """
    Ham kare dizisinden MP4 video klip dosyası yazar.

    Her çağrı bağımsızdır; aynı anda birden fazla klip yazılabilir
    (farklı dosya adlarıyla).
    """

    def __init__(self, output_dir: Optional[str] = None, fps: float = 30.0):
        """
        Args:
            output_dir: Kliplerin yazılacağı dizin.
                        Verilmezse sistem temp dizini kullanılır.
            fps:        Varsayılan kare hızı (VideoWriter için).
        """
        self.output_dir = output_dir or os.path.join(tempfile.gettempdir(), "ais_clips")
        self.fps = fps
        self.logger = logging.getLogger("ClipRecorder")

        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def record(
        self,
        frames: List[Tuple[int, np.ndarray]],
        filename: Optional[str] = None,
        fps: Optional[float] = None,
    ) -> Optional[str]:
        """
        Verilen kare listesini MP4 dosyasına yazar.

        Args:
            frames:   [(frame_id, raw_frame), ...] — RecordBuffer çıktısı.
            filename: Dosya adı (opsiyonel). Verilmezse otomatik oluşturulur.
            fps:      Bu klip için FPS (opsiyonel). Verilmezse self.fps kullanılır.

        Returns:
            Başarılı ise dosya yolu (str), başarısız ise None.
        """
        if not frames:
            self.logger.warning("No frames provided, clip creation skipped.")
            return None

        clip_fps = fps or self.fps

        # Dosya adı
        if filename is None:
            first_id = frames[0][0]
            last_id = frames[-1][0]
            filename = f"clip_{first_id}_{last_id}.mp4"

        filepath = os.path.join(self.output_dir, filename)

        # İlk kareden boyut bilgisi
        _, sample = frames[0]
        h, w = sample.shape[:2]

        # VideoWriter oluştur
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filepath, fourcc, clip_fps, (w, h))

        if not writer.isOpened():
            self.logger.error(f"Failed to open VideoWriter for: {filepath}")
            return None

        try:
            for _, frame in frames:
                writer.write(frame)
            return filepath
        except Exception as e:
            self.logger.error(f"Error writing clip: {e}")
            return None
        finally:
            writer.release()
            frame_count = len(frames)
            duration = frame_count / clip_fps if clip_fps > 0 else 0
            self.logger.info(
                f"Clip saved: {filepath} "
                f"({frame_count} frames, {duration:.1f}s, {clip_fps:.0f}fps)"
            )
