from .streamer import Streamer
from threading import Thread
import torch
import logging

from source.recording import RecordBuffer, AnomalyTracker, LiveRecorder
from source.recording.anomaly_tracker import EventClipRequest


class StreamIngestor(Thread):
    """
    Tek bir kameranın video akışını işleyen orchestrator.

    Akış:
        1. Streamer'dan ham kare oku
        2. Ham kareyi RecordBuffer'a ekle (pre-event ring buffer)
        3. Anomali kaydı aktifse, kareyi LiveRecorder'a yaz (diske)
        4. İşlenmiş kareyi SequenceBuffer'a ekle (inference için)
        5. Sequence hazır olduğunda inference çalıştır (InferenceService üzerinden)
        6. AnomalyTracker'a bildir — state geçişlerine göre kayıt başlat/bitir
    """

    def __init__(
        self,
        url,
        preprocessor,
        seq_buffer,
        engine,
        dispatcher,
        buffer_size=10,
        record_buffer=None,
        anomaly_tracker=None,
        live_recorder=None,
    ):
        super().__init__(daemon=True)
        self.url = url
        self.streamer = Streamer(url=url, buffer_size=buffer_size)
        self.preprocessor = preprocessor
        self.sequence_buffer = seq_buffer
        self.inference_engine = engine
        self.dispatcher = dispatcher
        self.logger = logging.getLogger("StreamIngestor")

        # Her kamera kendi kayıt bileşenlerine sahiptir
        self.record_buffer = record_buffer or RecordBuffer()
        self.anomaly_tracker = anomaly_tracker or AnomalyTracker()
        self.live_recorder = live_recorder or LiveRecorder()

        # Kamera metadata (IngestorManager tarafından atanır)
        self.cameraId = None

    def run(self):
        self.streamer.start()
        self._process_loop()

    def start_capture(self):
        self.start()

    def stop_capture(self):
        self.streamer.stop()
        # Aktif kayıt varsa iptal et
        if self.live_recorder.is_recording:
            self.live_recorder.abort()

    def get_stats(self):
        return self.streamer.get_stats()

    # ------------------------------------------------------------------ #
    #  Ana işleme döngüsü
    # ------------------------------------------------------------------ #

    def _process_loop(self):
        fps = self.streamer.fps if self.streamer.fps > 0 else 30.0

        while self.streamer.running or not self.streamer.frame_queue.empty():
            data = self.streamer.read_frame()
            if data is None:
                # Gecikmeli bağlantı veya bitiş kontrolü
                if not self.streamer.running:
                    break
                continue

            frame_id, raw_frame = data

            # 1) Ham kareyi RecordBuffer'a ekle (pre-event ring buffer)
            self.record_buffer.add_frame(frame_id, raw_frame)

            # 2) Anomali kaydı aktifse, kareyi canlı diske yaz
            if self.live_recorder.is_recording:
                self.live_recorder.add_frame(frame_id, raw_frame)

            # 3) İşlenmiş kareyi SequenceBuffer'a ekle (inference için)
            processed = self.preprocessor.process(raw_frame)
            self.sequence_buffer.add_frame(frame_id, processed)

            if self.sequence_buffer.is_ready():
                seq_data = self.sequence_buffer.get_sequence()

                # İndeksleri ve kareleri ayır
                indices = [item[0] for item in seq_data]
                frames = [item[1] for item in seq_data]

                # Zaman aralığını hesapla
                start_sec = indices[0] / fps
                end_sec = indices[-1] / fps

                # List[np.ndarray] -> torch.Tensor (T, H, W, C)
                seq_tensor = torch.stack([torch.from_numpy(f) for f in frames])
                # (T, H, W, C) -> (T, C, H, W) -> (B, T, C, H, W)
                seq_tensor = seq_tensor.permute(0, 3, 1, 2).unsqueeze(0)

                # Inference (InferenceService üzerinden thread-safe)
                results = self.inference_engine.predict(seq_tensor)

                meta_info = {
                    "start_sec": round(start_sec, 2),
                    "end_sec": round(end_sec, 2),
                    "start_frame": indices[0],
                    "end_frame": indices[-1],
                    "cameraId": self.cameraId,
                }

                self.dispatcher.dispatch(results, meta_info, raw_frame)

                # Anomali state machine'i güncelle
                prev_state = self.anomaly_tracker.state
                clip_request = self.anomaly_tracker.update(results, frame_id)

                # IDLE → ACTIVE geçişi: canlı kayda başla
                if (
                    prev_state == AnomalyTracker.IDLE
                    and self.anomaly_tracker.state == AnomalyTracker.ACTIVE
                ):
                    self._start_live_recording(frame_id, fps)

                # Olay tamamlandı: kaydı bitir
                if clip_request:
                    self._handle_completed_event(clip_request)

    # ------------------------------------------------------------------ #
    #  Kayıt yönetimi
    # ------------------------------------------------------------------ #

    def _start_live_recording(self, event_start_frame: int, fps: float):
        """
        Anomali başladığında canlı kaydı başlatır.

        RecordBuffer'dan pre-event kareleri alınır ve dosyaya yazılır.
        Sonraki kareler _process_loop'ta add_frame() ile eklenir.
        """
        pre_frames = self.record_buffer.get_clip_frames(
            num_frames=self.anomaly_tracker.pre_event_frames
        )

        cam_id = self.cameraId or "unknown"
        filename = f"cam{cam_id}_evt_{event_start_frame}.mp4"

        self.live_recorder.start(pre_frames, filename=filename, fps=fps)

    def _handle_completed_event(self, clip_request: EventClipRequest):
        """
        Tamamlanmış anomali olayı için kaydı bitirir,
        ardından CMS ingest + MinIO upload akışını başlatır.
        """
        clip_path = self.live_recorder.finalize()

        if not clip_path:
            self.logger.error("Live recording finalization failed.")
            return

        cam_id = self.cameraId or "unknown"
        self.logger.info(
            f"Event clip ready for camera {cam_id}: "
            f"type={clip_request.event_type}, "
            f"score={clip_request.max_score:.2f}, "
            f"detections={clip_request.detection_count}, "
            f"path={clip_path}"
        )

        # CMS'e event bildir + klip yükle (dispatch_event_for_camera)
        self.dispatcher.dispatch_event_for_camera(
            clip_request=clip_request,
            clip_path=clip_path,
            camera_id=cam_id,
        )
