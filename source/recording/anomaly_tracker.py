"""
AnomalyTracker — Anomali olaylarının yaşam döngüsünü yöneten state machine.

Sürekli gelen inference sonuçlarını tek bir olay olarak gruplar.
Her tespit için ayrı ingest isteği atılmasını önler.

State machine:
    IDLE       → Anomali yok, bekleme modunda
    ACTIVE     → Anomali devam ediyor, aynı olay içindeyiz
    POST_WAIT  → Anomali sinyali durdu, post-event kaydı bekleniyor

Akış:
    1. IDLE'da anomali tespit → ACTIVE'e geç (olay başlangıcı)
    2. ACTIVE'de anomali devam → aynı olayda kal, skor/tip güncelle
    3. ACTIVE'de normal tespit → POST_WAIT'e geç
    4. POST_WAIT'de anomali geri aklı → ACTIVE'e dön (aynı olay)
    5. POST_WAIT süresi doldu → EventClipRequest üret, IDLE'a dön

Kullanım:
    tracker = AnomalyTracker(pre_event_seconds=5, post_event_seconds=5)
    for each inference result:
        clip_req = tracker.update(prediction, frame_id)
        if clip_req:
            # Olay tamamlandı — klip oluştur ve CMS'e gönder
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional
import logging


@dataclass
class EventClipRequest:
    """
    Tamamlanmış bir anomali olayı için klip çıkarma isteği.
    AnomalyTracker tarafından üretilir, StreamIngestor tarafından tüketilir.
    """
    event_type: str           # En sık tespit edilen anomali tipi
    max_score: float          # Olay boyunca görülen en yüksek skor
    description: str          # En yüksek skorlu tespite ait açıklama
    start_frame: int          # Anomalinin başladığı kare
    end_frame: int            # Post-event dahil bitiş karesi
    pre_event_frames: int     # Klibe dahil edilecek ön-olay kare sayısı
    total_clip_frames: int    # Klip için gereken toplam kare sayısı
    timestamp: str            # Olay başlangıç zamanı (ISO-8601 UTC)
    detection_count: int      # Olay boyunca toplam anomali tespit sayısı


class AnomalyTracker:
    """
    Sürekli gelen inference sonuçlarını analiz ederek anomali olaylarını
    gruplayıp tek bir olay olarak raporlar.

    Amaçlar:
        1. Her inference sonucu için ayrı ingest isteği atılmasını önler
        2. Pre-event + event + post-event kare penceresi belirler
        3. Cooldown ile ardışık olayların birleşmesini engeller
    """

    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    POST_WAIT = "POST_WAIT"

    def __init__(
        self,
        normal_label: str = "Normal Videos",
        threshold: float = 0.0,
        pre_event_seconds: float = 5.0,
        post_event_seconds: float = 5.0,
        cooldown_seconds: float = 10.0,
        fps: float = 30.0,
    ):
        """
        Args:
            normal_label:       Model çıktısında "anomali yok" anlamına gelen etiket.
            threshold:          Minimum skor eşiği. Sadece score >= threshold olan
                                tespitler olay olarak kabul edilir.
            pre_event_seconds:  Anomali öncesi klibe dahil edilecek süre (saniye).
            post_event_seconds: Anomali sonrası klibe dahil edilecek süre (saniye).
            cooldown_seconds:   İki olay arasındaki minimum bekleme süresi (saniye).
            fps:                Videonun kare hızı (zamanlama hesapları için).
        """
        self.normal_label = normal_label
        self.threshold = threshold
        self.fps = fps
        self.pre_event_frames = int(pre_event_seconds * fps)
        self.post_event_frames = int(post_event_seconds * fps)
        self.cooldown_frames = int(cooldown_seconds * fps)

        self.state: str = self.IDLE
        self.logger = logging.getLogger("AnomalyTracker")

        # Aktif olay durumu
        self._event_start_frame: Optional[int] = None
        self._last_anomaly_frame: Optional[int] = None
        self._max_score: float = 0.0
        self._description: str = ""
        self._type_counts: Dict[str, int] = {}
        self._detection_count: int = 0
        self._event_timestamp: Optional[str] = None

        # Cooldown takibi
        self._last_event_end_frame: Optional[int] = None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def update(self, prediction: dict, frame_id: int) -> Optional[EventClipRequest]:
        """
        Her inference sonucu ile çağrılır.

        Args:
            prediction: Engine çıktısı — {"predicted_label": str, "probs": str/float, ...}
            frame_id:   Inference penceresinin son kare indeksi.

        Returns:
            Olay tamamlandıysa EventClipRequest, devam ediyorsa/yoksa None.
        """
        label = prediction.get("predicted_label", self.normal_label)
        score = float(prediction.get("probs", 0))
        description = prediction.get("description", "")
        is_anomaly = label != self.normal_label

        if self.state == self.IDLE:
            return self._handle_idle(is_anomaly, label, score, description, frame_id)
        elif self.state == self.ACTIVE:
            return self._handle_active(is_anomaly, label, score, description, frame_id)
        elif self.state == self.POST_WAIT:
            return self._handle_post_wait(is_anomaly, label, score, description, frame_id)

        return None

    def update_fps(self, fps: float) -> None:
        """
        FPS değiştiğinde zamanlama parametrelerini günceller.
        Olay başlamadan önce çağrılmalıdır.
        """
        ratio = fps / self.fps if self.fps > 0 else 1.0
        self.fps = fps
        self.pre_event_frames = int(self.pre_event_frames * ratio)
        self.post_event_frames = int(self.post_event_frames * ratio)
        self.cooldown_frames = int(self.cooldown_frames * ratio)

    # ------------------------------------------------------------------ #
    #  State handlers
    # ------------------------------------------------------------------ #

    def _handle_idle(self, is_anomaly, label, score, description, frame_id):
        """IDLE durumunda: yeni anomali bekler. Sadece score >= threshold ise olay başlar."""
        if is_anomaly and score >= self.threshold and self._cooldown_ok(frame_id):
            # Yeni olay başladı
            self.state = self.ACTIVE
            self._event_start_frame = frame_id
            self._last_anomaly_frame = frame_id
            self._max_score = score
            self._description = description
            self._type_counts = {label: 1}
            self._detection_count = 1
            self._event_timestamp = datetime.now(timezone.utc).isoformat()
            self.logger.info(
                f"Anomaly event STARTED: {label} "
                f"(score={score:.2f}, threshold={self.threshold}) at frame {frame_id}"
            )
        return None

    def _handle_active(self, is_anomaly, label, score, description, frame_id):
        """ACTIVE durumunda: anomali devam ediyor veya durdu."""
        if is_anomaly:
            # Olay devam ediyor — istatistikleri güncelle
            self._last_anomaly_frame = frame_id
            if score > self._max_score:
                self._max_score = score
                self._description = description
            self._type_counts[label] = self._type_counts.get(label, 0) + 1
            self._detection_count += 1
        else:
            # Normal tespit geldi — post-event beklemeye geç
            self.state = self.POST_WAIT
            self.logger.debug(
                f"Anomaly signal stopped at frame {frame_id}, "
                f"entering post-event wait ({self.post_event_frames} frames)."
            )
        return None

    def _handle_post_wait(self, is_anomaly, label, score, description, frame_id):
        """POST_WAIT durumunda: post-event süresi doluyor veya anomali geri dönüyor."""
        if is_anomaly:
            # Anomali geri döndü — aynı olay devam ediyor
            self.state = self.ACTIVE
            self._last_anomaly_frame = frame_id
            if score > self._max_score:
                self._max_score = score
                self._description = description
            self._type_counts[label] = self._type_counts.get(label, 0) + 1
            self._detection_count += 1
            self.logger.debug(
                f"Anomaly resumed at frame {frame_id}, back to ACTIVE."
            )
            return None

        # Post-event süresi doldu mu kontrol et
        frames_since_last_anomaly = frame_id - self._last_anomaly_frame
        if frames_since_last_anomaly >= self.post_event_frames:
            return self._finalize_event(frame_id)

        return None

    # ------------------------------------------------------------------ #
    #  Yardımcı
    # ------------------------------------------------------------------ #

    def _finalize_event(self, current_frame: int) -> EventClipRequest:
        """Olay tamamlandı — klip isteği oluştur ve state'i sıfırla."""
        # En sık görülen anomali tipini bul
        dominant_type = max(self._type_counts, key=self._type_counts.get)

        # Toplam klip kare sayısı: pre-event + olay süresi + post-event
        total_clip = self.pre_event_frames + (current_frame - self._event_start_frame)

        request = EventClipRequest(
            event_type=dominant_type,
            max_score=self._max_score,
            description=self._description,
            start_frame=self._event_start_frame,
            end_frame=current_frame,
            pre_event_frames=self.pre_event_frames,
            total_clip_frames=total_clip,
            timestamp=self._event_timestamp,
            detection_count=self._detection_count,
        )

        self.logger.info(
            f"Anomaly event FINALIZED: type={dominant_type}, "
            f"max_score={self._max_score:.2f}, detections={self._detection_count}, "
            f"frames=[{self._event_start_frame}..{current_frame}], "
            f"clip_frames={total_clip}"
        )

        # Cooldown başlangıcını kaydet ve state'i sıfırla
        self._last_event_end_frame = current_frame
        self._reset()
        return request

    def _cooldown_ok(self, frame_id: int) -> bool:
        """Son olaydan bu yana cooldown süresi dolmuş mu kontrol eder."""
        if self._last_event_end_frame is None:
            return True
        return (frame_id - self._last_event_end_frame) >= self.cooldown_frames

    def _reset(self):
        """Aktif olay durumunu sıfırlar, cooldown bilgisi korunur."""
        self.state = self.IDLE
        self._event_start_frame = None
        self._last_anomaly_frame = None
        self._max_score = 0.0
        self._description = ""
        self._type_counts = {}
        self._detection_count = 0
        self._event_timestamp = None
