"""
InferenceService — Thread-safe inference sarmalayıcı.

Tek bir model instance'ı üzerinden sıralı (serialized) inference yapar.
Çoklu StreamIngestor thread'leri güvenli şekilde predict() çağırabilir.

GPU bellek çakışması ve CUDA thread safety sorunlarını önler.

Kullanım:
    engine = InferenceFactory.create("VideoMAE", "config/videomae_cfg.json")
    service = InferenceService(engine)   # thread-safe sarmalama
"""

from threading import Lock
import logging


class InferenceService:
    """
    IInferenceEngine etrafında thread-safe sarmalayıcı.

    Lock tabanlı sıralama ile aynı anda sadece bir inference
    çalışmasını garanti eder. predict() imzası IInferenceEngine
    ile aynıdır, bu yüzden drop-in replacement olarak kullanılabilir.
    """

    def __init__(self, engine):
        """
        Args:
            engine: IInferenceEngine implementasyonu
                    (örn. VideoMAEAnomalyEngine)
        """
        self._engine = engine
        self._lock = Lock()
        self.logger = logging.getLogger("InferenceService")
        self.logger.info(
            f"Inference service initialized with {type(engine).__name__} "
            f"on device={getattr(engine, 'device', 'unknown')}"
        )

    def predict(self, sequence_tensor):
        """
        Thread-safe inference çağrısı.

        Aynı anda birden fazla thread çağırsa bile,
        model'e sırayla (serialized) erişim sağlanır.

        Args:
            sequence_tensor: Model girdisi (torch.Tensor).

        Returns:
            Model çıktısı (dict) — alttaki engine'in predict() dönüş değeri.
        """
        with self._lock:
            return self._engine.predict(sequence_tensor)

    @property
    def device(self):
        """Alttaki engine'in cihaz bilgisi."""
        return getattr(self._engine, "device", "unknown")
