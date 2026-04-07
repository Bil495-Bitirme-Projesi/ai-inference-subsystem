"""
RecordBuffer — Ham video karelerini sürekli saklayan thread-safe ring buffer.

Anomali tespit edildiğinde, bu buffer'dan son N kareyi çıkararak
video klip oluşturmak için kullanılır.

Pre-event / Post-event mantığı:
    - Buffer sürekli olarak ham kareleri biriktirir (ring buffer).
    - Anomali anında get_clip_frames() ile pre-event kareler alınır.
    - Post-event kareler için: anomali anından sonra da buffer'a kare eklenir,
      belirli bir süre sonra tekrar get_clip_frames() çağrılarak
      hem pre hem post event kareleri kapsayan klip oluşturulabilir.
    - Ancak bu zamanlama orkestra katmanında (StreamIngestor/ResultDispatcher)
      yönetilmelidir; buffer kendisi bekleme yapmaz.

Kullanım:
    buffer = RecordBuffer(max_frames=900)   # ~30sn @ 30fps
    buffer.add_frame(frame_id=0, raw_frame=frame)
    clip_frames = buffer.get_clip_frames(num_frames=300)  # son ~10sn
"""

from collections import deque
from threading import Lock
from typing import List, Tuple

import numpy as np


class RecordBuffer:
    """
    Sabit kapasiteli, thread-safe ring buffer.

    En eski kareler otomatik olarak düşürülür (FIFO).
    Tüm kareler ham (raw, işlenmemiş) olarak saklanır;
    böylece video klip doğrudan bu karelerden yazılabilir.
    """

    def __init__(self, max_frames: int = 900):
        """
        Args:
            max_frames: Buffer'da tutulacak maksimum kare sayısı.
                        Örnek: 30 fps × 30 saniye = 900 kare.
        """
        self._buffer: deque[Tuple[int, np.ndarray]] = deque(maxlen=max_frames)
        self._lock = Lock()

    # ------------------------------------------------------------------ #
    #  Yazma
    # ------------------------------------------------------------------ #

    def add_frame(self, frame_id: int, raw_frame: np.ndarray) -> None:
        """
        Buffer'a yeni bir kare ekler.

        Kare, `.copy()` ile kopyalanır; böylece dış referans değişse bile
        buffer içindeki veri bozulmaz.

        Args:
            frame_id:  Kare sıra numarası (Streamer'dan gelen indeks).
            raw_frame: Ham (BGR, uint8) numpy dizisi.
        """
        with self._lock:
            self._buffer.append((frame_id, raw_frame.copy()))

    # ------------------------------------------------------------------ #
    #  Okuma
    # ------------------------------------------------------------------ #

    def get_clip_frames(self, num_frames: int = 300) -> List[Tuple[int, np.ndarray]]:
        """
        Buffer'ın sonundan (en güncel) *num_frames* kadar kareyi döndürür.

        Buffer'da istenen miktardan az kare varsa, mevcut tüm kareler döner.

        Args:
            num_frames: İstenen kare sayısı.
                        Örnek: 30 fps × 10 saniye = 300 kare.

        Returns:
            [(frame_id, raw_frame), ...] listesi — zaman sıralı.
        """
        with self._lock:
            frames = list(self._buffer)

        if len(frames) > num_frames:
            return frames[-num_frames:]
        return frames

    # ------------------------------------------------------------------ #
    #  Yardımcı
    # ------------------------------------------------------------------ #

    def clear(self) -> None:
        """Buffer'ı tamamen temizler."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def max_frames(self) -> int:
        """Buffer kapasitesi."""
        return self._buffer.maxlen
