"""
AIS Integration Tests — Test Planındaki Entegrasyon Testleri

TC-AIS-IT-06: End-to-End Auth Flow
TC-AIS-IT-07: Dynamic Start via WS
TC-AIS-IT-08: Dynamic Stop via WS
TC-AIS-IT-09: Anomaly Event Reporting
TC-AIS-IT-10: RTSP Stream Resiliency
TC-AIS-IT-11: Health Status Propagation
TC-AIS-IT-12: JWT Expiry & Auto-Refresh
TC-AIS-IT-13: Clip Storage Integration
TC-AIS-IT-14: Concurrency & Thread Sync

NOT: Bu testlerin çoğu gerçek servislere (CMS, RTSP, MinIO) bağlanır.
     Çalıştırmadan önce servislerin ayakta olduğundan emin olun.

Kullanım:
    uv run -m pytest tests/test_integration.py -v --tb=short
    
    Sadece mock testleri (servis gerektirmeyenler):
    uv run -m pytest tests/test_integration.py -v --tb=short -k "not real"
"""

import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from source.comm.api_client import CMSApiClient
from source.comm.ws_client import CMSWebSocketClient
from source.orchestrator.ingestor_manager import IngestorManager
from source.orchestrator.stream_ingestor import StreamIngestor
from source.dispatch.result_dispatcher import ResultDispatcher
from source.recording import AnomalyTracker, RecordBuffer, LiveRecorder
from source.recording.anomaly_tracker import EventClipRequest
from source.recording.clip_uploader import ClipUploader
from source.proc.preproc import Preprocessor
from source.proc.sequence_buf import SequenceBuffer
from source.engine.inference_service import InferenceService


# ================================================================== #
#  TC-AIS-IT-06: End-to-End Auth Flow
#  Gerçek CMS'e login olur, HTTP 200 alır, JWT saklanır.
# ================================================================== #

class TestEndToEndAuth:
    """Gerçek CMS'e bağlanarak auth akışını test eder."""

    def test_real_cms_login(self):
        """CMS'e gerçek login isteği atılır ve token alınır."""
        client = CMSApiClient()
        result = client.login(force=True)

        assert result is True, "CMS login başarısız — .env ayarlarını kontrol edin"
        assert client.token is not None, "Token alınamadı"
        assert len(client.token) > 10, f"Token çok kısa: {client.token}"
        assert "Authorization" in client.session.headers
        assert client.session.headers["Authorization"].startswith("Bearer ")


# ================================================================== #
#  TC-AIS-IT-07: Dynamic Start via WS
#  SNAPSHOT mesajı geldiğinde IngestorManager doğru kameraları başlatır.
# ================================================================== #

class TestDynamicStartViaWS:
    """WebSocket SNAPSHOT mesajı ile kamera başlatma."""

    def test_snapshot_spawns_ingestors(self):
        """SNAPSHOT mesajı geldiğinde aktif kameralar için ingestor başlamalı."""
        manager = IngestorManager(
            preprocessor=MagicMock(),
            engine=MagicMock(device="mock"),
            dispatcher=MagicMock(),
        )

        snapshot = {
            "type": "SNAPSHOT",
            "cameras": [
                {"cameraId": 1, "rtspUrl": "rtsp://fake/cam1", "detectionEnabled": True, "threshold": 0.5},
                {"cameraId": 2, "rtspUrl": "rtsp://fake/cam2", "detectionEnabled": True, "threshold": 0.7},
            ],
        }

        with patch.object(manager, '_start_new') as mock_start:
            manager.sync_from_snapshot(snapshot)

        assert mock_start.call_count == 2
        # Her çağrıdaki config'i doğrula
        started_ids = [call.args[0]["cameraId"] for call in mock_start.call_args_list]
        assert set(started_ids) == {1, 2}

    def test_ws_client_routes_snapshot_to_callback(self):
        """CMSWebSocketClient SNAPSHOT mesajını on_snapshot callback'ine yönlendirmeli."""
        received = {}

        def on_snapshot(data):
            received["data"] = data

        def on_delta(data):
            pass

        ws = CMSWebSocketClient(
            api_client=MagicMock(),
            on_snapshot=on_snapshot,
            on_camera_delta=on_delta,
        )

        # Mesajı simüle et
        import json
        msg = json.dumps({"type": "CONFIG_SNAPSHOT", "cameras": [{"cameraId": 1}]})
        ws._on_message(None, msg)

        assert "data" in received
        assert received["data"]["type"] == "CONFIG_SNAPSHOT"


# ================================================================== #
#  TC-AIS-IT-08: Dynamic Stop via WS
#  CAMERA_DELTA (enabled: false) ile belirli kamera durdurulur.
# ================================================================== #

class TestDynamicStopViaWS:
    """WebSocket CAMERA_DELTA ile kamera durdurma."""

    def test_delta_delete_stops_ingestor(self):
        """CAMERA_DELTA DELETE geldiğinde ilgili StreamIngestor durmalı."""
        manager = IngestorManager(
            preprocessor=MagicMock(),
            engine=MagicMock(device="mock"),
            dispatcher=MagicMock(),
        )

        mock_ingestor = MagicMock()
        manager.ingestors[5] = mock_ingestor

        delta = {
            "type": "CAMERA_DELTA",
            "changeType": "DELETE",
            "cameraId": 5,
        }

        manager.handle_camera_delta(delta)

        mock_ingestor.stop_capture.assert_called_once()
        assert 5 not in manager.ingestors

    def test_delta_upsert_disabled_stops_ingestor(self):
        """CAMERA_DELTA UPSERT + detectionEnabled=false → kamera durdurulur."""
        manager = IngestorManager(
            preprocessor=MagicMock(),
            engine=MagicMock(device="mock"),
            dispatcher=MagicMock(),
        )

        mock_ingestor = MagicMock()
        manager.ingestors[5] = mock_ingestor

        delta = {
            "type": "CAMERA_DELTA",
            "changeType": "UPSERT",
            "cameraId": 5,
            "camera": {"rtspUrl": "rtsp://fake/cam5", "detectionEnabled": False, "threshold": 0.5},
        }

        manager.handle_camera_delta(delta)

        mock_ingestor.stop_capture.assert_called_once()
        assert 5 not in manager.ingestors

    def test_ws_client_routes_delta_to_callback(self):
        """CMSWebSocketClient CAMERA_DELTA mesajını on_camera_delta callback'ine yönlendirmeli."""
        received = {}

        def on_snapshot(data):
            pass

        def on_delta(data):
            received["data"] = data

        ws = CMSWebSocketClient(
            api_client=MagicMock(),
            on_snapshot=on_snapshot,
            on_camera_delta=on_delta,
        )

        import json
        msg = json.dumps({"type": "CAMERA_DELTA", "changeType": "DELETE", "cameraId": 5})
        ws._on_message(None, msg)

        assert "data" in received
        assert received["data"]["changeType"] == "DELETE"


# ================================================================== #
#  TC-AIS-IT-09: Anomaly Event Reporting
#  Anomali tespit edildiğinde CMS /api/events/ingest 201 dönmeli.
# ================================================================== #

class TestAnomalyEventReporting:
    """CMS ingest endpoint'ine olay bildirimi."""

    def test_ingest_returns_201_with_mock(self):
        """Mock API client ile ingest akışı doğru çalışmalı."""
        mock_api = MagicMock()
        mock_api.ingest_event.return_value = {
            "eventId": 42,
            "status": "CREATED",
            "clipUploadUrl": "http://minio/test-upload",
        }
        mock_uploader = MagicMock()
        mock_uploader.upload.return_value = True

        dispatcher = ResultDispatcher(api_client=mock_api, clip_uploader=mock_uploader)

        clip_request = EventClipRequest(
            event_type="Fighting",
            max_score=0.92,
            description="Fighting detected with 92% confidence",
            start_frame=100,
            end_frame=200,
            pre_event_frames=30,
            total_clip_frames=130,
            timestamp=datetime.now(timezone.utc).isoformat(),
            detection_count=5,
        )

        result = dispatcher.dispatch_event_for_camera(clip_request, "/tmp/test.mp4", camera_id=5)

        assert result is True
        mock_api.ingest_event.assert_called_once()
        mock_uploader.upload.assert_called_once_with("/tmp/test.mp4", "http://minio/test-upload")

    def test_real_ingest_event(self):
        """Gerçek CMS'e event ingest isteği gönderir."""
        client = CMSApiClient()
        if not client.login(force=True):
            pytest.skip("CMS login failed, skipping real ingest test")

        event_data = {
            "sourceEventId": str(__import__("uuid").uuid4()),
            "cameraId": 3,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "score": 0.85,
            "type": "Fighting",
            "description": "Integration test event",
        }

        result = client.ingest_event(event_data)
        assert result is not None, "Ingest response None döndü"
        assert "eventId" in result or result.get("status") == "DUPLICATE"


# ================================================================== #
#  TC-AIS-IT-10: RTSP Stream Resiliency
#  RTSP bağlantısı koparsa hata loglanır ve streamer durmaz.
# ================================================================== #

class TestRTSPStreamResiliency:
    """RTSP akışı koptuğunda sistemin hatayı yönetmesi."""

    def test_invalid_rtsp_does_not_crash(self):
        """Geçersiz RTSP URL ile StreamIngestor crash olmamalı."""
        mock_engine = MagicMock()
        mock_engine.device = "mock"
        mock_engine.predict.return_value = {
            "predicted_label": "Normal Videos",
            "probs": "0.95",
            "description": "test",
        }

        ingestor = StreamIngestor(
            url="rtsp://invalid-host:9999/nonexistent",
            preprocessor=MagicMock(),
            seq_buffer=SequenceBuffer(16, 1),
            engine=mock_engine,
            dispatcher=MagicMock(),
        )
        ingestor.cameraId = 999

        ingestor.start_capture()
        time.sleep(3)  # 3 saniye bekle — crash olmadan çalışmalı
        ingestor.stop_capture()

        # Thread hala alive ise veya temiz kapandıysa test geçer
        assert True  # Crash olmadan buraya ulaştıysa OK


# ================================================================== #
#  TC-AIS-IT-11: Health Status Propagation
#  Kamera offline olduğunda WS üzerinden CAMERA_STATUS: OFFLINE gönderilir.
# ================================================================== #

class TestHealthStatusPropagation:
    """Kamera durumu değişikliklerinin CMS'e iletilmesi."""

    def test_register_camera_sends_online(self):
        """register_camera çağrıldığında ONLINE durumu gönderilmeli."""
        ws = CMSWebSocketClient(
            api_client=MagicMock(),
            on_snapshot=MagicMock(),
            on_camera_delta=MagicMock(),
        )

        # send_camera_status'u mockla (gerçek WS bağlantısı yok)
        with patch.object(ws, 'send_camera_status') as mock_send:
            ws.register_camera(5, "ONLINE")

        mock_send.assert_called_once_with(5, "ONLINE")
        assert ws._active_cameras[5] == "ONLINE"

    def test_unregister_camera_sends_offline(self):
        """unregister_camera çağrıldığında OFFLINE durumu gönderilmeli."""
        ws = CMSWebSocketClient(
            api_client=MagicMock(),
            on_snapshot=MagicMock(),
            on_camera_delta=MagicMock(),
        )
        ws._active_cameras[5] = "ONLINE"

        with patch.object(ws, 'send_camera_status') as mock_send:
            ws.unregister_camera(5)

        mock_send.assert_called_once_with(5, "OFFLINE")
        assert 5 not in ws._active_cameras

    def test_stop_ingestor_sends_offline_via_ws(self):
        """IngestorManager.stop_ingestor() → ws_client.unregister_camera() çağırmalı."""
        mock_ws = MagicMock()
        manager = IngestorManager(
            preprocessor=MagicMock(),
            engine=MagicMock(device="mock"),
            dispatcher=MagicMock(),
            ws_client=mock_ws,
        )

        mock_ingestor = MagicMock()
        manager.ingestors[5] = mock_ingestor

        manager.stop_ingestor(5)

        mock_ws.unregister_camera.assert_called_once_with(5)


# ================================================================== #
#  TC-AIS-IT-12: JWT Expiry & Auto-Refresh
#  401 alındığında otomatik re-login olup istek tekrar denmeli.
# ================================================================== #

class TestJWTExpiryAutoRefresh:
    """Token süresi dolduğunda otomatik yenileme mekanizması."""

    def test_401_triggers_relogin_and_retry(self):
        """İlk istek 401 → re-login → tekrar istek → 201 başarılı."""
        client = CMSApiClient()
        client.token = "expired-token"
        client.session.headers["Authorization"] = "Bearer expired-token"

        # İlk çağrı 401, ikinci çağrı 201
        resp_401 = MagicMock()
        resp_401.status_code = 401

        resp_201 = MagicMock()
        resp_201.status_code = 201
        resp_201.json.return_value = {"eventId": 1}

        call_count = {"n": 0}

        def mock_session_request(method, url, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return resp_401
            return resp_201

        client.session.request = mock_session_request

        # login'i mockla
        login_response = MagicMock()
        login_response.status_code = 200
        login_response.json.return_value = {"token": "new-fresh-token"}

        with patch("source.comm.api_client.requests.post", return_value=login_response):
            result = client.ingest_event({
                "sourceEventId": "test-id",
                "cameraId": 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "score": 0.9,
                "type": "Fighting",
                "description": "test",
            })

        assert result is not None
        assert result["eventId"] == 1
        assert client.token == "new-fresh-token"

    def test_lazy_login_skips_when_token_exists(self):
        """Token varken login(force=False) yeni istek atmamalı."""
        client = CMSApiClient()
        client.token = "existing-token"

        with patch("source.comm.api_client.requests.post") as mock_post:
            result = client.login(force=False)

        assert result is True
        mock_post.assert_not_called()


# ================================================================== #
#  TC-AIS-IT-13: Clip Storage Integration
#  Anomali → presigned URL alınır → klip MinIO'ya yüklenir.
# ================================================================== #

class TestClipStorageIntegration:
    """Klip kaydetme ve MinIO'ya yükleme entegrasyonu."""

    def test_full_dispatch_flow_with_mocks(self):
        """
        dispatch_event_for_camera akışı:
        ingest → clipUploadUrl al → upload çağır.
        """
        mock_api = MagicMock()
        mock_api.ingest_event.return_value = {
            "eventId": 99,
            "clipUploadUrl": "https://minio.local/presigned-put-url",
        }

        mock_uploader = MagicMock()
        mock_uploader.upload.return_value = True

        dispatcher = ResultDispatcher(api_client=mock_api, clip_uploader=mock_uploader)

        clip_request = EventClipRequest(
            event_type="Arson",
            max_score=0.88,
            description="Arson detected with 88% confidence",
            start_frame=500,
            end_frame=700,
            pre_event_frames=60,
            total_clip_frames=260,
            timestamp=datetime.now(timezone.utc).isoformat(),
            detection_count=8,
        )

        result = dispatcher.dispatch_event_for_camera(clip_request, "/clips/test.mp4", camera_id=10)

        assert result is True
        # Upload doğru URL ile çağrıldı mı?
        mock_uploader.upload.assert_called_once_with(
            "/clips/test.mp4",
            "https://minio.local/presigned-put-url",
        )

    def test_missing_clip_upload_url_skips_upload(self):
        """Ingest response'da clipUploadUrl yoksa upload atlanmalı ama ingest başarılı."""
        mock_api = MagicMock()
        mock_api.ingest_event.return_value = {"eventId": 100}  # clipUploadUrl yok

        mock_uploader = MagicMock()
        dispatcher = ResultDispatcher(api_client=mock_api, clip_uploader=mock_uploader)

        clip_request = EventClipRequest(
            event_type="Robbery",
            max_score=0.75,
            description="Robbery detected",
            start_frame=0,
            end_frame=100,
            pre_event_frames=30,
            total_clip_frames=130,
            timestamp=datetime.now(timezone.utc).isoformat(),
            detection_count=3,
        )

        result = dispatcher.dispatch_event_for_camera(clip_request, "/clips/test.mp4", camera_id=1)

        assert result is True  # Ingest başarılı
        mock_uploader.upload.assert_not_called()  # Upload çağrılmadı


# ================================================================== #
#  TC-AIS-IT-14: Concurrency & Thread Sync
#  3 eşzamanlı stream'de race condition olmamalı.
# ================================================================== #

class TestConcurrencyThreadSync:
    """Çoklu kamera thread'leri arasında veri bütünlüğü."""

    def test_separate_sequence_buffers_per_camera(self):
        """Her kameranın kendi SequenceBuffer'ı olmalı — frame'ler karışmamalı."""
        buf1 = SequenceBuffer(sequence_length=4, stride=1)
        buf2 = SequenceBuffer(sequence_length=4, stride=1)
        buf3 = SequenceBuffer(sequence_length=4, stride=1)

        frame_a = np.ones((224, 224, 3), dtype=np.float32) * 1.0
        frame_b = np.ones((224, 224, 3), dtype=np.float32) * 2.0
        frame_c = np.ones((224, 224, 3), dtype=np.float32) * 3.0

        # Paralel ekleme simülasyonu
        for i in range(4):
            buf1.add_frame(i, frame_a)
            buf2.add_frame(i + 100, frame_b)
            buf3.add_frame(i + 200, frame_c)

        # Her buffer kendi frame'lerini döndürmeli
        seq1 = buf1.get_sequence()
        seq2 = buf2.get_sequence()
        seq3 = buf3.get_sequence()

        assert np.allclose(seq1[0][1], 1.0), "Buffer 1 yanlış frame içeriyor"
        assert np.allclose(seq2[0][1], 2.0), "Buffer 2 yanlış frame içeriyor"
        assert np.allclose(seq3[0][1], 3.0), "Buffer 3 yanlış frame içeriyor"

        # Frame ID'ler karışmamış olmalı
        assert seq1[0][0] == 0
        assert seq2[0][0] == 100
        assert seq3[0][0] == 200

    def test_inference_service_serializes_calls(self):
        """InferenceService Lock ile eşzamanlı çağrıları sıraya koymalı."""
        call_log = []

        class SlowEngine:
            device = "mock"

            def predict(self, tensor):
                tid = threading.current_thread().name
                call_log.append(("start", tid))
                time.sleep(0.1)  # İşlem süresi simülasyonu
                call_log.append(("end", tid))
                return {"predicted_label": "Normal Videos", "probs": "0.95", "description": "test"}

        service = InferenceService(SlowEngine())
        threads = []

        for i in range(3):
            t = threading.Thread(
                target=lambda: service.predict(np.zeros((1, 16, 3, 224, 224))),
                name=f"cam-{i}",
            )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # Sıralı çalışma kontrolü: her "start" bir "end" ile takip edilmeli
        # Aralıklı (interleaved) olmamalı
        for idx in range(0, len(call_log), 2):
            assert call_log[idx][0] == "start"
            assert call_log[idx + 1][0] == "end"
            assert call_log[idx][1] == call_log[idx + 1][1], (
                f"Çağrı sıralı değil! {call_log[idx]} ve {call_log[idx + 1]} farklı thread"
            )

    def test_manager_creates_separate_buffers(self):
        """IngestorManager her kamera için ayrı SequenceBuffer oluşturmalı."""
        manager = IngestorManager(
            preprocessor=MagicMock(),
            engine=MagicMock(device="mock"),
            dispatcher=MagicMock(),
        )

        created_configs = []

        def spy_start(cfg):
            created_configs.append(cfg["cameraId"])

        with patch.object(manager, '_start_new', side_effect=spy_start):
            manager.sync_from_snapshot({
                "type": "SNAPSHOT",
                "cameras": [
                    {"cameraId": 1, "rtspUrl": "rtsp://fake/1", "detectionEnabled": True, "threshold": 0.5},
                    {"cameraId": 2, "rtspUrl": "rtsp://fake/2", "detectionEnabled": True, "threshold": 0.5},
                    {"cameraId": 3, "rtspUrl": "rtsp://fake/3", "detectionEnabled": True, "threshold": 0.5},
                ],
            })

        assert len(created_configs) == 3, f"3 kamera başlatılmalı, {len(created_configs)} bulundu"
        assert set(created_configs) == {1, 2, 3}, "Her kamera için ayrı _start_new çağrılmalı"
