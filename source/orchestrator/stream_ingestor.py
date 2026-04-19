from source.proc import sequence_buf
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
        frame_count = 0
        local_frame_id = 0  # Local counter for consistent frame indexing

        self.logger.info(f"Processing loop started for camera {self.cameraId} (fps={fps})")

        while self.streamer.running or not self.streamer.frame_queue.empty():
            try:
                data = self.streamer.read_frame()
                if data is None:
                    # Gecikmeli bağlantı veya bitiş kontrolü
                    if not self.streamer.running:
                        break
                    continue

                stream_frame_id, raw_frame = data
                frame_count += 1

                if frame_count == 1:
                    self.logger.info(
                        f"Camera {self.cameraId}: First frame received "
                        f"(shape={raw_frame.shape}, dtype={raw_frame.dtype})"
                    )

                # 1) Ham kareyi RecordBuffer'a ekle (pre-event ring buffer)
                self.record_buffer.add_frame(local_frame_id, raw_frame)

                # 2) Anomali kaydı aktifse, kareyi canlı diske yaz
                if self.live_recorder.is_recording:
                    self.live_recorder.add_frame(local_frame_id, raw_frame)

                # 3) İşlenmiş kareyi SequenceBuffer'a ekle (stride sampling yapılır)
                processed = self.preprocessor.process(raw_frame)
                was_added = self.sequence_buffer.add_frame(local_frame_id, processed)
                
                # Sadece stride sampling sonrasında sequence kontrol et
                if was_added and self.sequence_buffer.is_ready():
                    seq_data = self.sequence_buffer.get_sequence()

                    # İndeksleri ve kareleri ayır
                    indices = [item[0] for item in seq_data]
                    frames = [item[1] for item in seq_data]

                    self.logger.info(f"Sequence indices (stride-sampled): {indices}")

                    # Zaman aralığını hesapla
                    start_sec = indices[0] / fps
                    end_sec = indices[-1] / fps

                    self.logger.debug(
                        f"Camera {self.cameraId}: Sequence ready "
                        f"[{start_sec:.1f}s - {end_sec:.1f}s], running inference..."
                    )

                    # List[np.ndarray] -> torch.Tensor (T, H, W, C)
                    seq_tensor = torch.stack([torch.from_numpy(f) for f in frames])
                    self.logger.info(f"Camera {self.cameraId}: Sequence tensor shape: {seq_tensor.shape}")
                    
                    # Inference (InferenceService üzerinden thread-safe)
                    results = self.inference_engine.predict(seq_tensor)

                    self.sequence_buffer.flush()   
                    self.logger.info(f"Camera {self.cameraId}: Cleared frame queue.")

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
                    clip_request = self.anomaly_tracker.update(results, local_frame_id)

                    # IDLE → ACTIVE geçişi: canlı kayda başla
                    if (
                        prev_state == self.anomaly_tracker.IDLE
                        and self.anomaly_tracker.state == self.anomaly_tracker.ACTIVE
                    ):
                        self._start_live_recording(local_frame_id, fps)

                    # Olay tamamlandı: kaydı bitir
                    if clip_request is not None:
                        was_chained = self.anomaly_tracker.state == self.anomaly_tracker.ACTIVE        
                        self._handle_completed_event(clip_request)
                        
                        # Chained event (max duration) ise hemen yeni segment kaydını başlat
                        if was_chained:
                            self.logger.info(f"Camera {self.cameraId}: Starting next segment (chained).")
                            self._start_live_recording(local_frame_id, fps)

                local_frame_id += 1  # Increment after processing

            except Exception as e:
                self.logger.error(
                    f"Camera {self.cameraId}: Error in processing loop "
                    f"(frame_count={frame_count}): {e}",
                    exc_info=True,
                )

        self.logger.info(
            f"Camera {self.cameraId}: Processing loop ended "
            f"(total frames processed: {frame_count})"
        )

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
