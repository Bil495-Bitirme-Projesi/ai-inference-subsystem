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
        normal_label: str = "Normal",
        threshold: float = 0.0,
        pre_event_seconds: float = 0.0,
        post_event_seconds: float = 0.0,
        cooldown_seconds: float = 0.0,
        max_event_seconds: float = 20.0,
        fps: float = 30.0,
        smoothing_alpha: float = 0.8,
    ):
        """
        Args:
            normal_label:       Model çıktısında "anomali yok" anlamına gelen etiket.
            threshold:          Minimum skor eşiği. Sadece score >= threshold olan
                                tespitler olay olarak kabul edilir.
            pre_event_seconds:  Anomali öncesi klibe dahil edilecek süre (saniye).
            post_event_seconds: Anomali sonrası klibe dahil edilecek süre (saniye).
            cooldown_seconds:   İki olay arasındaki minimum bekleme süresi (saniye).
            max_event_seconds:  Bir olayın ACTIVE state'te kalabileceği maksimum süre.
                                Limit dolunca olay finalize edilir; anomali devam
                                ediyorsa yeni olay otomatik chain edilir.
            fps:                Videonun kare hızı (zamanlama hesapları için).
        """
        self.normal_label = normal_label
        self.threshold = threshold
        self.fps = fps
        self.pre_event_frames = int(pre_event_seconds * fps)
        self.post_event_frames = int(post_event_seconds * fps)
        self.cooldown_frames = int(cooldown_seconds * fps)
        self.max_event_frames = int(max_event_seconds * fps)

        # Exponential moving average for smoothing anomaly scores (to avoid jitter)
        # ema_{t} = alpha * raw_score_t + (1-alpha) * ema_{t-1}
        # raw_score_t is score when label != normal_label, otherwise 0.0
        self.smoothing_alpha = float(smoothing_alpha)
        self._ema_score: float = 0.0
        self._last_non_normal_label: Optional[str] = None

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
        # support both numeric and string probability formats, and both "prob" and "probs" keys
        try:
            val = prediction.get("prob") or prediction.get("probs") or 0.0
            score = float(val)
        except Exception:
            score = 0.0
        description = prediction.get("description", "")

        # raw indicator: current frame's raw anomaly score (0.0 if label == normal)
        raw_score = score if label != self.normal_label else 0.0

        # update last seen non-normal label to use when EMA triggers anomaly
        if label != self.normal_label:
            self._last_non_normal_label = label

        # update EMA of anomaly score
        alpha = max(0.0, min(1.0, self.smoothing_alpha))
        self._ema_score = alpha * raw_score + (1.0 - alpha) * self._ema_score

        # smoothed anomaly decision (use > to exclude threshold boundary)
        is_anomaly = self._ema_score > self.threshold

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
        self.max_event_frames = int(self.max_event_frames * ratio)

    # ------------------------------------------------------------------ #
    #  State handlers
    # ------------------------------------------------------------------ #

    def _handle_idle(self, is_anomaly, label, score, description, frame_id):
        """IDLE durumunda: yeni anomali bekler. EMA tabanlı karar verilir ve cooldown kontrolü yapılır."""
        # Use the smoothed EMA decision. If EMA indicates anomaly, pick a label
        # from the latest non-normal observation (if available).
        if is_anomaly and self._cooldown_ok(frame_id):
            chosen_label = self._last_non_normal_label or label
            
            # Yeni olay başladı
            self.state = self.ACTIVE
            self._event_start_frame = frame_id
            self._last_anomaly_frame = frame_id
            
            # Only set max_score and description from non-normal frames
            self._max_score = score if label != self.normal_label else 0.0
            self._description = description if label != self.normal_label else ""
            self._type_counts = {chosen_label: 1}
            self._detection_count = 1
            self._event_timestamp = datetime.now(timezone.utc).isoformat()
            self.logger.info(
                f"Anomaly event STARTED: {chosen_label} "
                f"(ema_score={self._ema_score:.3f}, threshold={self.threshold}) at frame {frame_id}"
            )
        return None

    def _handle_active(self, is_anomaly, label, score, description, frame_id):
        """ACTIVE durumunda: anomali devam ediyor veya durdu (EMA bazlı)."""
        # ── Max-duration kontrolü ──────────────────────────────────────
        # Sürekli anomalilerde kaydın sınırsız büyümesini engeller.
        # Limit dolunca mevcut olay finalize edilir; anomali hâlâ
        # devam ediyorsa yeni bir olay otomatik olarak chain edilir.
        # Note: Total clip = pre_event + event + post_event.
        # We contr
        # ol event portion only, but total should stay within reasonable bounds.
        start = self._event_start_frame if self._event_start_frame is not None else frame_id
        event_duration = frame_id - start
        # Effective max includes pre_event buffer that will be added when recording starts
        effective_max_frames = self.max_event_frames + self.pre_event_frames
        if event_duration >= effective_max_frames:
            self.logger.info(
                f"Max event duration reached at frame {frame_id} "
                f"(event_duration={event_duration} frames, "
                f"effective_max={effective_max_frames} frames)."
            
            )
            
            clip_request = self._finalize_event(frame_id)

            # Anomali devam ediyorsa → yeni olay chain et (gap-free)
            # Only chain if current frame has non-normal label (not just EMA smoothing)
            if is_anomaly and label != self.normal_label:
                chosen_label = label
                self.state = self.ACTIVE
                self._event_start_frame = frame_id
                self._last_anomaly_frame = frame_id
                # Only set max_score and description from non-normal frames
                self._max_score = score
                self._description = description
                self._type_counts = {chosen_label: 1}
                self._detection_count = 1
                self._event_timestamp = datetime.now(timezone.utc).isoformat()
                self.logger.info(
                    f"Max duration reached — event chained at frame {frame_id} (seconds={frame_id/self.fps:.2f})"
                )

            return clip_request

        # ── Normal akış ────────────────────────────────────────────────
        if is_anomaly:
            # Olay devam ediyor — istatistikleri güncelle
            self._last_anomaly_frame = frame_id
            # Only update max_score from non-normal frames (exclude normal label scores)
            if label != self.normal_label and score > self._max_score:
                self._max_score = score
                self._description = description
            # Update counts using raw label if available, otherwise use last_non_normal
            used_label = label if label != self.normal_label else (self._last_non_normal_label or label)
            self._type_counts[used_label] = self._type_counts.get(used_label, 0) + 1
            self._detection_count += 1
        else:
            # EMA indicates anomaly has stopped — enter post-event wait
            self.state = self.POST_WAIT
            self.logger.debug(
                f"Anomaly signal stopped (EMA) at frame {frame_id}, "
                f"entering post-event wait ({self.post_event_frames} frames)."
            )
        return None

    def _handle_post_wait(self, is_anomaly, label, score, description, frame_id):
        """POST_WAIT durumunda: post-event süresi doluyor veya anomali geri dönüyor."""
        if is_anomaly:
            # Anomali geri döndü — aynı olay devam ediyor
            self.state = self.ACTIVE
            self._last_anomaly_frame = frame_id
            # Only update max_score from non-normal frames (exclude normal label scores)
            if label != self.normal_label and score > self._max_score:
                self._max_score = score
                self._description = description
            used_label = label if label != self.normal_label else (self._last_non_normal_label or label)
            self._type_counts[used_label] = self._type_counts.get(used_label, 0) + 1
            self._detection_count += 1
            self.logger.debug(
                f"Anomaly resumed at frame {frame_id}, back to ACTIVE."
            )
            return None

        # Post-event süresi doldu mu kontrol et (use last anomaly frame index)
        last = self._last_anomaly_frame if self._last_anomaly_frame is not None else frame_id
        frames_since_last_anomaly = frame_id - last
        
        # Also apply max total duration constraint: prevent event from running indefinitely
        # even during post-wait (total = event + post should not exceed max + post)
        start = self._event_start_frame if self._event_start_frame is not None else frame_id
        total_duration = frame_id - start
        effective_max_total = self.max_event_frames + self.post_event_frames
        
        if frames_since_last_anomaly >= self.post_event_frames or total_duration >= effective_max_total:
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
            pre_event_frames=0,
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
