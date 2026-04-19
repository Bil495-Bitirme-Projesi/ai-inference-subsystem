"""
AIS Unit Tests — Test Planındaki Birim Testleri

TC-AIS-UT-06: Buffer Size Enforcement
TC-AIS-UT-07: Payload Mapping
TC-AIS-UT-08: Snapshot Message Parsing
TC-AIS-UT-09: Inference Result Scoring
TC-AIS-UT-10: Auth Header Validation

Kullanım:
    uv run -m pytest tests/test_unit.py -v --tb=short
"""

import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from source.proc.sequence_buf import SequenceBuffer
from source.recording.anomaly_tracker import AnomalyTracker, EventClipRequest
from source.dispatch.result_dispatcher import ResultDispatcher
from source.comm.api_client import CMSApiClient
from source.orchestrator.ingestor_manager import IngestorManager


# ================================================================== #
#  TC-AIS-UT-06: Buffer Size Enforcement
# ================================================================== #

class TestBufferSizeEnforcement:
    """
    SequenceBuffer(max_size=16) + 20 frames →
    Buffer length remains 16; oldest 4 frames are evicted.
    """

    def test_buffer_never_exceeds_sequence_length(self):
        """Buffer boyutu sequence_length'i aşmamalı."""
        buf = SequenceBuffer(sequence_length=16, stride=1)
        dummy_frame = np.zeros((224, 224, 3), dtype=np.float32)

        for i in range(20):
            buf.add_frame(i, dummy_frame)

        assert len(buf.buffer) == 16, (
            f"Buffer {len(buf.buffer)} eleman içeriyor, 16 olmalı"
        )

    def test_oldest_frames_are_evicted(self):
        """20 frame eklendiğinde ilk 4 frame (0-3) elenmeli, 4-19 kalmalı."""
        buf = SequenceBuffer(sequence_length=16, stride=1)
        dummy_frame = np.zeros((224, 224, 3), dtype=np.float32)

        for i in range(20):
            buf.add_frame(i, dummy_frame)

        frame_ids = [item[0] for item in buf.buffer]
        assert frame_ids[0] == 4, f"İlk kare ID=4 olmalı, {frame_ids[0]} bulundu"
        assert frame_ids[-1] == 19, f"Son kare ID=19 olmalı, {frame_ids[-1]} bulundu"

    def test_buffer_is_ready_after_filling(self):
        """16 frame dolduğunda is_ready() True dönmeli."""
        buf = SequenceBuffer(sequence_length=16, stride=1)
        dummy_frame = np.zeros((224, 224, 3), dtype=np.float32)

        for i in range(15):
            buf.add_frame(i, dummy_frame)
        assert not buf.is_ready(), "15 frame ile henüz hazır olmamalı"

        buf.add_frame(15, dummy_frame)
        assert buf.is_ready(), "16 frame ile hazır olmalı"


# ================================================================== #
#  TC-AIS-UT-07: Payload Mapping
# ================================================================== #

class TestPayloadMapping:
    """
    ResultDispatcher.dispatch_event_for_camera() ile oluşturulan JSON'ın
    AIS-CMS kontratına uygun yapıda olması (UUID, ISO-8601, vb.).
    """

    def test_event_payload_has_correct_structure(self):
        """Oluşturulan payload gerekli tüm alanları içermeli."""
        mock_api = MagicMock()
        mock_api.ingest_event.return_value = {
            "eventId": 1,
            "clipUploadUrl": "http://minio/test",
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

        dispatcher.dispatch_event_for_camera(clip_request, "/tmp/test.mp4", camera_id=5)

        # ingest_event'e gönderilen payload'ı yakala
        call_args = mock_api.ingest_event.call_args[0][0]

        # UUID formatı kontrolü
        assert "sourceEventId" in call_args
        parsed_uuid = uuid.UUID(call_args["sourceEventId"])  # Geçersizse ValueError fırlatır
        assert parsed_uuid.version == 4

        # ISO-8601 timestamp kontrolü
        assert "timestamp" in call_args
        datetime.fromisoformat(call_args["timestamp"])  # Geçersizse ValueError fırlatır

        # Diğer gerekli alanlar
        assert call_args["cameraId"] == 5
        assert call_args["type"] == "Fighting"
        assert call_args["score"] == 0.92
        assert call_args["description"] == "Fighting detected with 92% confidence"

    def test_dispatch_logs_without_api_client(self):
        """API client yokken dispatch() sadece print ile loglama yapar."""
        dispatcher = ResultDispatcher(api_client=None)
        # Hata fırlatmamalı
        dispatcher.dispatch(
            {"predicted_label": "Normal Videos", "probs": "0.95"},
            {"start_sec": 0.0, "end_sec": 0.5},
            np.zeros((224, 224, 3)),
        )


# ================================================================== #
#  TC-AIS-UT-08: Snapshot Message Parsing
# ================================================================== #

class TestSnapshotMessageParsing:
    """
    Mock JSON ile 2 enabled, 1 disabled kamera →
    IngestorManager sadece 2 kamera başlatmalı.
    """

    def test_only_enabled_cameras_are_started(self):
        """detectionEnabled=true olan kameralar başlatılmalı."""
        mock_preprocessor = MagicMock()
        mock_engine = MagicMock()
        mock_engine.device = "mock"
        mock_dispatcher = MagicMock()

        manager = IngestorManager(
            preprocessor=mock_preprocessor,
            engine=mock_engine,
            dispatcher=mock_dispatcher,
        )

        snapshot_data = {
            "type": "SNAPSHOT",
            "cameras": [
                {"cameraId": 1, "rtspUrl": "rtsp://fake/cam1", "detectionEnabled": True, "threshold": 0.5},
                {"cameraId": 2, "rtspUrl": "rtsp://fake/cam2", "detectionEnabled": True, "threshold": 0.7},
                {"cameraId": 3, "rtspUrl": "rtsp://fake/cam3", "detectionEnabled": False, "threshold": 0.3},
            ],
        }

        # _start_new'ü mocklayarak gerçek thread başlatılmasını engelle
        with patch.object(manager, '_start_new') as mock_start:
            manager.sync_from_snapshot(snapshot_data)
            assert mock_start.call_count == 2, (
                f"2 kamera başlatılmalı, {mock_start.call_count} çağrıldı"
            )

    def test_disabled_camera_in_snapshot_is_stopped(self):
        """Önceden aktif olan ama yeni snapshot'ta disabled olan kamera durdurulmalı."""
        mock_preprocessor = MagicMock()
        mock_engine = MagicMock()
        mock_engine.device = "mock"
        mock_dispatcher = MagicMock()

        manager = IngestorManager(
            preprocessor=mock_preprocessor,
            engine=mock_engine,
            dispatcher=mock_dispatcher,
        )

        # Sahte bir aktif ingestor ekle
        mock_ingestor = MagicMock()
        manager.ingestors[3] = mock_ingestor

        # Yeni snapshot: cameraId=3 artık enabled=false
        snapshot_data = {
            "type": "SNAPSHOT",
            "cameras": [
                {"cameraId": 1, "rtspUrl": "rtsp://fake/cam1", "detectionEnabled": True, "threshold": 0.5},
            ],
        }

        with patch.object(manager, '_start_new'):
            manager.sync_from_snapshot(snapshot_data)
            # cameraId=3 artık ingestors dict'inde olmamalı
            assert 3 not in manager.ingestors, "Disabled kamera durdurulmalıydı"
            mock_ingestor.stop_capture.assert_called_once()


# ================================================================== #
#  TC-AIS-UT-09: Inference Result Scoring
# ================================================================== #

class TestInferenceResultScoring:
    """
    predicted_label="Fighting", score=0.9 →
    AnomalyTracker anomali olarak işaretler (IDLE → ACTIVE geçişi).
    """

    def test_anomaly_detection_transitions_to_active(self):
        """Eşiğin üzerinde anomali skoru geldiğinde IDLE → ACTIVE geçişi olmalı."""
        tracker = AnomalyTracker(threshold=0.5, smoothing_alpha=1.0, normal_label="Normal Videos")
        assert tracker.state == "IDLE"

        prediction = {
            "predicted_label": "Fighting",
            "probs": "0.90",
            "description": "Fighting detected with 90% confidence",
        }

        tracker.update(prediction, frame_id=100)
        assert tracker.state == "ACTIVE", (
            f"State ACTIVE olmalı, {tracker.state} bulundu"
        )

    def test_below_threshold_stays_idle(self):
        """Eşiğin altında anomali skoru geldiğinde IDLE'da kalmalı."""
        tracker = AnomalyTracker(threshold=0.95, smoothing_alpha=1.0, normal_label="Normal Videos")

        prediction = {
            "predicted_label": "Fighting",
            "probs": "0.90",
            "description": "Fighting detected with 90% confidence",
        }

        tracker.update(prediction, frame_id=100)
        assert tracker.state == "IDLE", (
            f"Score < threshold iken IDLE kalmalı, {tracker.state} bulundu"
        )

    def test_normal_label_stays_idle(self):
        """Normal label geldiğinde her zaman IDLE kalmalı."""
        tracker = AnomalyTracker(threshold=0.5, smoothing_alpha=1.0, normal_label="Normal Videos")

        prediction = {
            "predicted_label": "Normal Videos",
            "probs": "0.95",
            "description": "Normal Videos detected with 95% confidence",
        }

        tracker.update(prediction, frame_id=100)
        assert tracker.state == "IDLE"

    def test_full_event_lifecycle(self):
        """IDLE → ACTIVE → POST_WAIT → IDLE döngüsünde EventClipRequest üretilmeli."""
        tracker = AnomalyTracker(
            threshold=0.5,
            pre_event_seconds=1,
            post_event_seconds=0.1,  # 3 frame (çok kısa)
            fps=30.0,
            smoothing_alpha=1.0,
            normal_label="Normal Videos",
        )

        anomaly = {"predicted_label": "Fighting", "probs": "0.90", "description": "test"}
        normal = {"predicted_label": "Normal Videos", "probs": "0.95", "description": "test"}

        # IDLE → ACTIVE
        tracker.update(anomaly, frame_id=100)
        assert tracker.state == "ACTIVE"

        # ACTIVE → POST_WAIT (normal geldi)
        tracker.update(normal, frame_id=116)
        assert tracker.state == "POST_WAIT"

        # POST_WAIT süresi dolsun (frame_id farkı >= post_event_frames)
        clip_request = tracker.update(normal, frame_id=200)
        assert tracker.state == "IDLE"
        assert clip_request is not None
        assert isinstance(clip_request, EventClipRequest)
        assert clip_request.event_type == "Fighting"
        assert clip_request.max_score == 0.90

    def test_max_duration_forces_finalization(self):
        """max_event_seconds dolduğunda olay finalize edilmeli ve anomali devam
        ediyorsa yeni olay chain edilmeli (state ACTIVE kalmalı)."""
        tracker = AnomalyTracker(
            threshold=0.5,
            max_event_seconds=1.0,   # 30 frame @ 30fps
            pre_event_seconds=0,
            post_event_seconds=0,
            fps=30.0,
            smoothing_alpha=1.0,
            normal_label="Normal Videos",
        )

        anomaly = {"predicted_label": "Fighting", "probs": "0.90", "description": "test"}

        # IDLE → ACTIVE
        tracker.update(anomaly, frame_id=0)
        assert tracker.state == "ACTIVE"

        # Feed continuous anomaly up to just before max limit (29 frames)
        clip_request = None
        for fid in range(1, 30):
            clip_request = tracker.update(anomaly, frame_id=fid)
        assert clip_request is None, "Should not finalize before max_event_frames"

        # Frame 30: exactly at max limit → should finalize and chain
        clip_request = tracker.update(anomaly, frame_id=30)
        assert clip_request is not None, "Should finalize at max_event_frames"
        assert isinstance(clip_request, EventClipRequest)
        assert clip_request.event_type == "Fighting"
        # Anomaly still active → state should be ACTIVE (chained)
        assert tracker.state == "ACTIVE", (
            f"Continuous anomaly should chain a new event, got {tracker.state}"
        )


# ================================================================== #
#  TC-AIS-UT-10: Auth Header Validation
# ================================================================== #

class TestAuthHeaderValidation:
    """
    CMSApiClient ile token='xyz123' atandıktan sonra
    session header'ında 'Authorization: Bearer xyz123' olmalı.
    """

    def test_auth_header_is_set_after_login(self):
        """Başarılı login sonrası session header'ında doğru token olmalı."""
        client = CMSApiClient()

        # login() isteğini mockla
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"token": "xyz123"}

        with patch("source.comm.api_client.requests.post", return_value=mock_response):
            result = client.login(force=True)

        assert result is True
        assert client.token == "xyz123"
        assert client.session.headers["Authorization"] == "Bearer xyz123"

    def test_auth_header_format(self):
        """Token formatı 'Bearer <token>' şeklinde olmalı."""
        client = CMSApiClient()
        test_token = "eyJhbGciOiJIUzI1NiJ9.test"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"token": test_token}

        with patch("source.comm.api_client.requests.post", return_value=mock_response):
            client.login(force=True)

        header = client.session.headers["Authorization"]
        assert header.startswith("Bearer "), f"Bearer prefix eksik: {header}"
        assert header == f"Bearer {test_token}"

    def test_failed_login_does_not_set_token(self):
        """Başarısız login sonrası token None kalmalı."""
        client = CMSApiClient()

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("source.comm.api_client.requests.post", return_value=mock_response):
            result = client.login(force=True)

        assert result is False
        assert client.token is None
