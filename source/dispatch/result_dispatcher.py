# source/dispatch/result_dispatcher.py
from .interfaces import IDispatcher
import uuid
from datetime import datetime, timezone
import logging

from source.recording.clip_uploader import ClipUploader
from source.recording.anomaly_tracker import EventClipRequest


class ResultDispatcher(IDispatcher):
    def __init__(self, api_client=None, clip_uploader=None):
        """
        Args:
            api_client:    CMSApiClient nesnesi (Opsiyonel).
                           Verilirse anomali durumunda CMS'e bildirim gönderir.
            clip_uploader: ClipUploader nesnesi (Opsiyonel).
                           Verilirse klip dosyasını MinIO'ya yükler.
        """
        self.api_client = api_client
        self.clip_uploader = clip_uploader or ClipUploader()
        self.logger = logging.getLogger("ResultDispatcher")

    def dispatch(self, detections, info, frame):
        """
        Her inference sonucu için çağrılır — sadece loglama yapar.
        """
        time_range = f"[{info.get('start_sec')}s - {info.get('end_sec')}s]"
        print(f"\n[DISPATCH] {time_range} Results: {detections}")

    def dispatch_event(self, clip_request: EventClipRequest, clip_path: str) -> bool:
        """
        Tamamlanmış bir anomali olayını CMS'e bildirir ve klibini yükler.

        AnomalyTracker tarafından TEK bir olay olarak gruplanmış sinyallerin
        sonucunda çağrılır. Akış:
          1. Event payload oluştur
          2. POST /api/events/ingest → clipUploadUrl al
          3. PUT clipUploadUrl → klip dosyasını MinIO'ya yükle

        Args:
            clip_request: AnomalyTracker'dan gelen EventClipRequest.
            clip_path:    LiveRecorder tarafından oluşturulan klip dosyası yolu.

        Returns:
            Başarılı ise True, değilse False.
        """
        if not self.api_client:
            self.logger.warning("No API client configured, event dispatch skipped.")
            return False

        event_payload = {
            "sourceEventId": str(uuid.uuid4()),
            "cameraId": clip_request.start_frame,  # Placeholder — sonra override edilir
            "timestamp": clip_request.timestamp,
            "score": round(clip_request.max_score, 4),
            "type": clip_request.event_type,
        }

        return self._ingest_and_upload(event_payload, clip_path)

    def dispatch_event_for_camera(
        self,
        clip_request: EventClipRequest,
        clip_path: str,
        camera_id,
    ) -> bool:
        """
        Kamera ID'si ile birlikte tamamlanmış olayı CMS'e bildirir.

        Args:
            clip_request: AnomalyTracker'dan gelen EventClipRequest.
            clip_path:    Klip dosyası yolu.
            camera_id:    Olayın ait olduğu kamera ID'si.

        Returns:
            Başarılı ise True.
        """
        event_payload = {
            "sourceEventId": str(uuid.uuid4()),
            "cameraId": camera_id,
            "timestamp": clip_request.timestamp,
            "score": round(clip_request.max_score, 4),
            "type": clip_request.event_type,
        }

        return self._ingest_and_upload(event_payload, clip_path)

    def _ingest_and_upload(self, event_payload: dict, clip_path: str) -> bool:
        """
        Event ingest + klip yükleme ortak akışı.

        1. POST /api/events/ingest → response'dan clipUploadUrl al
        2. PUT clipUploadUrl → klip dosyasını MinIO'ya yükle
        """
        # CMS'e event bildir
        ingest_result = self.api_client.ingest_event(event_payload)

        if ingest_result is None:
            self.logger.error(
                f"Event ingestion failed for sourceEventId={event_payload['sourceEventId']}"
            )
            return False

        self.logger.info(
            f"Event reported to CMS: "
            f"eventId={ingest_result.get('eventId')}, "
            f"type={event_payload['type']}, "
            f"score={event_payload['score']}"
        )

        # Klip dosyasını MinIO'ya yükle
        clip_upload_url = ingest_result.get("clipUploadUrl")
        if not clip_upload_url:
            self.logger.warning("No clipUploadUrl in ingest response, clip upload skipped.")
            return True  # Ingest başarılı, sadece upload atlandı

        upload_success = self.clip_uploader.upload(clip_path, clip_upload_url)
        if upload_success:
            self.logger.info(
                f"Clip uploaded for eventId={ingest_result.get('eventId')}"
            )
        else:
            self.logger.error(
                f"Clip upload failed for eventId={ingest_result.get('eventId')}"
            )

        return upload_success