# orchestrator/ingestor_manager.py
import logging
from typing import Dict
from .stream_ingestor import StreamIngestor

class IngestorManager:
    def __init__(self, preprocessor, seq_buffer, engine, dispatcher):
        self.ingestors: Dict[int, StreamIngestor] = {}
        self.preprocessor = preprocessor
        self.seq_buffer = seq_buffer
        self.engine = engine
        self.dispatcher = dispatcher
        self.logger = logging.getLogger("IngestorManager")

    def sync_from_snapshot(self, cameras_list: list):
        """SNAPSHOT mesajı geldiğinde tüm kameraları senkronize eder"""
        active_ids = {cam['cameraId'] for cam in cameras_list if cam['detectionEnabled']}
        
        # Artık listede olmayan veya devre dışı bırakılanları durdur
        for cam_id in list(self.ingestors.keys()):
            if cam_id not in active_ids:
                self.stop_ingestor(cam_id)

        # Yeni veya güncellenenleri başlat
        for cam in cameras_list:
            if cam['detectionEnabled']:
                self.update_or_start_ingestor(cam)

    def update_or_start_ingestor(self, cam_config: dict):
        """Tekil kamera güncelleme (UPSERT)"""
        cam_id = cam_config['cameraId']
        url = cam_config['rtspUrl']
        
        if cam_id in self.ingestors:
            # URL değiştiyse yeniden başlat
            if self.ingestors[cam_id].url != url:
                self.stop_ingestor(cam_id)
                self._start_new(cam_config)
        else:
            self._start_new(cam_config)

    def _start_new(self, cfg):
        ingestor = StreamIngestor(
            url=cfg['rtspUrl'],
            preprocessor=self.preprocessor,
            seq_buffer=self.seq_buffer,
            engine=self.engine,
            dispatcher=self.dispatcher
        )
        ingestor.cameraId = cfg['cameraId'] # Metadata ekle
        ingestor.start_capture()
        self.ingestors[cfg['cameraId']] = ingestor
        self.logger.info(f"Kamera {cfg['cameraId']} başlatıldı.")

    def stop_ingestor(self, cam_id: int):
        if cam_id in self.ingestors:
            self.ingestors[cam_id].stop_capture()
            del self.ingestors[cam_id]
            self.logger.info(f"Kamera {cam_id} durduruldu.")