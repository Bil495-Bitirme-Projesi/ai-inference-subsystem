"""
AIS (AI Inference Subsystem) — Ana başlatma dosyası.

Tüm bileşenleri doğru sırayla oluşturur, CMS ile bağlantı kurar
ve kamera konfigürasyonunu WebSocket üzerinden dinlemeye başlar.

Başlatma akışı:
    1. AI Engine yüklenir (VideoMAE) ve InferenceService ile sarmalanır
    2. CMS'e login olunur (JWT token alınır)
    3. WebSocket bağlantısı kurulur
    4. CMS'den SNAPSHOT ile kamera listesi alınır
    5. Her kamera için StreamIngestor başlatılır
    6. Anomali tespitinde → klip kayıt → CMS ingest → MinIO upload

Kullanım:
    uv run main.py
"""

import sys
import signal
import logging
import threading

from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

from source.comm.api_client import CMSApiClient
from source.comm.ws_client import CMSWebSocketClient
from source.orchestrator.ingestor_manager import IngestorManager
from source.dispatch.result_dispatcher import ResultDispatcher
from source.proc.preproc import Preprocessor
from source.engine.inference_factory import InferenceFactory
from source.engine.inference_service import InferenceService
from source.recording import ClipUploader

# VideoMAE engine'i factory'ye kaydetmek için import et
# (@register_inference_engine decorator'ı import sırasında çalışır)
import source.engine.videomae_engine  # noqa: F401


def setup_logging():
    """Logging konfigürasyonu."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    setup_logging()
    logger = logging.getLogger("AIS")
    logger.info("=" * 50)
    logger.info("AI Inference Subsystem starting...")
    logger.info("=" * 50)

    # AI Engine — model bir kez yüklenir, tüm kameralar paylaşır
    logger.info("Loading AI inference engine...")
    raw_engine = InferenceFactory.create("VideoMAE", "config/videomae_cfg.json")
    engine = InferenceService(raw_engine)
    logger.info(f"Engine ready on device: {engine.device}")

    preprocessor = Preprocessor()

    #CMS API Client — login
    api_client = CMSApiClient()
    if not api_client.login():
        logger.critical("CMS login failed! Check CMS_REST_URL, SUBSYSTEM_ID, SUBSYSTEM_SECRET.")
        sys.exit(1)

    # CMS ingest + MinIO clip upload
    clip_uploader = ClipUploader()
    dispatcher = ResultDispatcher(api_client=api_client, clip_uploader=clip_uploader)

    # kamera yaşam döngüsü yönetimi
    manager = IngestorManager(
        preprocessor=preprocessor,
        engine=engine,
        dispatcher=dispatcher,
        sequence_length=16,
        stride=1,
    )

    # WebSocket Client — CMS ile konfigürasyon senkronizasyonu
    ws_client = CMSWebSocketClient(
        api_client=api_client,
        on_snapshot=manager.sync_from_snapshot,
        on_camera_delta=manager.handle_camera_delta,
    )

    # WS client'ı manager'a bağla (kamera ONLINE/OFFLINE + heartbeat için)
    manager.ws_client = ws_client

    # WebSocket bağlantısını başlat (arka plan thread'i)
    ws_client.connect()

    logger.info("AIS is running. Waiting for camera configuration from CMS...")

    # Graceful shutdown
    shutdown_event = threading.Event()

    def signal_handler(signum, frame):
        logger.info(f"Shutdown signal received (signal={signum}). Stopping...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Ana thread'i canlı tut — shutdown sinyali bekle
    try:
        shutdown_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down AIS...")
        manager.stop_all()
        ws_client.stop()
        logger.info("AIS stopped. Goodbye.")


if __name__ == "__main__":
    main()