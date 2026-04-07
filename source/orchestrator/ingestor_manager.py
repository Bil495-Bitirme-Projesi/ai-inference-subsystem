# orchestrator/ingestor_manager.py
import logging
from typing import Dict, Optional
from .stream_ingestor import StreamIngestor
from source.proc.sequence_buf import SequenceBuffer
from source.recording import RecordBuffer, AnomalyTracker, LiveRecorder


class IngestorManager:
    def __init__(
        self,
        preprocessor,
        engine,
        dispatcher,
        ws_client=None,
        sequence_length=16,
        stride=1,
    ):
        """
        Args:
            preprocessor:    Paylaşımlı Preprocessor instance'ı (stateless, güvenli).
            engine:          Paylaşımlı InferenceService instance'ı (thread-safe).
            dispatcher:      Paylaşımlı ResultDispatcher instance'ı.
            ws_client:       CMSWebSocketClient — kamera durumu bildirimi için.
            sequence_length: Her kameranın SequenceBuffer'ı için pencere boyutu.
            stride:          İki inference arası kare atlama sayısı.
        """
        self.ingestors: Dict[int, StreamIngestor] = {}
        self.preprocessor = preprocessor
        self.engine = engine
        self.dispatcher = dispatcher
        self.ws_client = ws_client
        self.sequence_length = sequence_length
        self.stride = stride
        self.logger = logging.getLogger("IngestorManager")

    # ------------------------------------------------------------------ #
    #  SNAPSHOT — tam senkronizasyon
    # ------------------------------------------------------------------ #

    def sync_from_snapshot(self, snapshot_data: dict):
        """
        SNAPSHOT mesajı geldiğinde tüm kameraları senkronize eder.
        snapshot_data: {"type": "SNAPSHOT", "cameras": [...]}
        """
        cameras_list = snapshot_data.get("cameras", [])
        active_ids = {cam["cameraId"] for cam in cameras_list if cam.get("detectionEnabled")}

        # Artık listede olmayan veya devre dışı bırakılanları durdur
        for cam_id in list(self.ingestors.keys()):
            if cam_id not in active_ids:
                self.stop_ingestor(cam_id)

        # Yeni veya güncellenenleri başlat
        for cam in cameras_list:
            if cam.get("detectionEnabled"):
                self.update_or_start_ingestor(cam)

    # ------------------------------------------------------------------ #
    #  CAMERA_DELTA — tekil güncelleme
    # ------------------------------------------------------------------ #

    def handle_camera_delta(self, delta_data: dict):
        """
        CameraDelta mesajını işler.

        delta_data formatı:
            UPSERT: {"type": "CAMERA_DELTA", "changeType": "UPSERT",
                     "cameraId": 5, "camera": {rtspUrl, detectionEnabled, threshold, ...}}
            DELETE: {"type": "CAMERA_DELTA", "changeType": "DELETE", "cameraId": 5}
        """
        change_type = delta_data.get("changeType")
        cam_id = delta_data.get("cameraId")

        if change_type == "UPSERT":
            cam_config = delta_data.get("camera", {})
            cam_config["cameraId"] = cam_id  # ID'yi config'e ekle

            if cam_config.get("detectionEnabled"):
                self.update_or_start_ingestor(cam_config)
            else:
                # detectionEnabled=false → durdur
                self.stop_ingestor(cam_id)

            self.logger.info(f"Camera delta UPSERT processed: camera {cam_id}")

        elif change_type == "DELETE":
            self.stop_ingestor(cam_id)
            self.logger.info(f"Camera delta DELETE processed: camera {cam_id}")

        else:
            self.logger.warning(f"Unknown delta changeType: {change_type}")

    # ------------------------------------------------------------------ #
    #  Ingestor CRUD
    # ------------------------------------------------------------------ #

    def update_or_start_ingestor(self, cam_config: dict):
        """Tekil kamera güncelleme (UPSERT)"""
        cam_id = cam_config["cameraId"]
        url = cam_config["rtspUrl"]

        if cam_id in self.ingestors:
            # URL değiştiyse yeniden başlat
            if self.ingestors[cam_id].url != url:
                self.stop_ingestor(cam_id)
                self._start_new(cam_config)
        else:
            self._start_new(cam_config)

    def _start_new(self, cfg):
        cam_id = cfg["cameraId"]
        threshold = cfg.get("threshold", 0.0)

        # Her kamera kendi bileşenlerine sahip (mutable state paylaşılmaz)
        cam_seq_buffer = SequenceBuffer(
            sequence_length=self.sequence_length,
            stride=self.stride,
        )
        cam_anomaly_tracker = AnomalyTracker(threshold=threshold)

        ingestor = StreamIngestor(
            url=cfg["rtspUrl"],
            preprocessor=self.preprocessor,
            seq_buffer=cam_seq_buffer,
            engine=self.engine,
            dispatcher=self.dispatcher,
            anomaly_tracker=cam_anomaly_tracker,
        )
        ingestor.cameraId = cam_id
        ingestor.start_capture()
        self.ingestors[cam_id] = ingestor

        # Kamera durumunu CMS'e bildir
        if self.ws_client:
            self.ws_client.register_camera(cam_id, "ONLINE")

        self.logger.info(
            f"Camera {cam_id} started (threshold={threshold})."
        )

    def stop_ingestor(self, cam_id):
        if cam_id in self.ingestors:
            self.ingestors[cam_id].stop_capture()
            del self.ingestors[cam_id]

            # Kamera durumunu CMS'e bildir
            if self.ws_client:
                self.ws_client.unregister_camera(cam_id)

            self.logger.info(f"Camera {cam_id} stopped.")

    def stop_all(self):
        """Tüm kameraları durdurur."""
        for cam_id in list(self.ingestors.keys()):
            self.stop_ingestor(cam_id)
        self.logger.info("All cameras stopped.")