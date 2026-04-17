import websocket
import json
import ssl
import threading
import time
import os
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

# .env değişkenlerini yükle
load_dotenv()

class CMSWebSocketClient:
    def __init__(self, api_client, on_snapshot: callable, on_camera_delta: callable):
        """
        Args:
            api_client:       Kimlik doğrulama yönetimi için CMSApiClient instance'ı.
            on_snapshot:      SNAPSHOT mesajını işleyecek callback — f(data: dict)
            on_camera_delta:  CAMERA_DELTA mesajını işleyecek callback — f(data: dict)
        """
        self.api_client = api_client
        self.on_snapshot = on_snapshot
        self.on_camera_delta = on_camera_delta
        self.ws_base_url = os.getenv("CMS_WS_URL")
        self.ws = None
        self.logger = logging.getLogger("CMSWebSocketClient")
        self.should_reconnect = True

        # SSL context (self-signed cert desteği)
        cert_path = os.getenv("SSL_CERT_PATH")
        self._sslopt = {}
        if cert_path:
            self._sslopt = {
                "cert_reqs": ssl.CERT_NONE,
                "ca_certs": cert_path,
            }

        # Exponential backoff parametreleri
        self._reconnect_delay = 1       # Başlangıç bekleme süresi (saniye)
        self._max_reconnect_delay = 60  # Maksimum bekleme süresi (saniye)

        # Heartbeat — aktif kameraların durumunu periyodik bildirir
        self._heartbeat_interval = 60   # saniye
        self._heartbeat_timer = None
        self._active_cameras = {}       # {cameraId: "ONLINE"/"OFFLINE"}

    # ------------------------------------------------------------------ #
    #  Bağlantı yönetimi
    # ------------------------------------------------------------------ #

    def _ensure_connection_url(self) -> str:
        """
        API client'ın geçerli bir tokenı olduğundan emin olur ve WS URL'ini oluşturur.
        """
        self.logger.info("Attempting to login before connecting to WebSocket...")
        login_success = self.api_client.login()

        if not login_success:
            raise ConnectionError("Authentication failed: Could not acquire JWT token.")

        # Güncel token ile tam URL oluşturulur
        return f"{self.ws_base_url}?token={self.api_client.token}"

    def connect(self):
        """WebSocket bağlantısını arka planda bir thread içinde başlatır."""
        threading.Thread(target=self._run_forever_loop, daemon=True).start()

    def _run_forever_loop(self):
        """Bağlantı koptuğunda exponential backoff ile yeniden bağlanır."""
        while self.should_reconnect:
            try:
                # Bağlanmadan önce token'dan emin ol ve URL'i al
                url = self._ensure_connection_url()

                self.logger.info(f"Establishing WebSocket connection to: {self.ws_base_url}")

                self.ws = websocket.WebSocketApp(
                    url,
                    on_message=self._on_message,
                    on_open=self._on_open,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )

                # run_forever bağlantı kesilene kadar thread'i bloklar
                self.ws.run_forever(sslopt=self._sslopt)

            except Exception as e:
                self.logger.error(f"WebSocket connection loop error: {e}")

            # Exponential backoff ile yeniden bağlan
            if self.should_reconnect:
                self.logger.info(f"Reconnecting in {self._reconnect_delay}s...")
                time.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self._max_reconnect_delay,
                )

    # ------------------------------------------------------------------ #
    #  WebSocket event handler'ları
    # ------------------------------------------------------------------ #

    def _on_open(self, ws):
        """Bağlantı başarılı bir şekilde açıldığında tetiklenir."""
        self.logger.info("WebSocket connection opened successfully.")

        # Başarılı bağlantıda backoff süresini sıfırla
        self._reconnect_delay = 1

        # Bağlantı açılınca SNAPSHOT isteği gönder
        try:
            request_msg = json.dumps({"type": "SNAPSHOT"})
            self.ws.send(request_msg)
            self.logger.info("SNAPSHOT request sent to CMS.")
        except Exception as e:
            self.logger.error(f"Failed to send initial snapshot request: {e}")

        # Heartbeat döngüsünü başlat
        self._start_heartbeat()

    def _on_message(self, ws, message):
        """CMS'ten gelen mesajları ayrıştırır ve ilgili callback'e yönlendirir."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            self.logger.debug(f"Received message type: {msg_type}")

            if msg_type == "CONFIG_SNAPSHOT":
                self.logger.info("Config snapshot received from CMS.")
                self.on_snapshot(data)
            elif msg_type == "CAMERA_DELTA":
                self.logger.info(f"Camera delta received: {data.get('changeType')}")
                self.on_camera_delta(data)
            else:
                self.logger.warning(f"Unrecognized message received: {data}")

        except json.JSONDecodeError:
            self.logger.error("Received an invalid JSON message from WebSocket.")
        except Exception as e:
            self.logger.error(f"Error while processing WebSocket message: {e}")

    def _on_error(self, ws, error):
        """WebSocket kütüphanesinden gelen hataları loglar."""
        self.logger.error(f"WebSocket execution error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """Bağlantı kapandığında tetiklenir."""
        self._stop_heartbeat()
        self.logger.warning(
            f"WebSocket connection closed. Code: {close_status_code}, Message: {close_msg}"
        )

    # ------------------------------------------------------------------ #
    #  Kamera durumu bildirimi (AIS → CMS)
    # ------------------------------------------------------------------ #

    def send_camera_status(self, camera_id, status: str):
        """
        Kameranın ONLINE/OFFLINE durumunu CMS'e iletir.
        reportedAt alanı ISO-8601 UTC formatında eklenir.
        """
        if self.ws and self.ws.sock and self.ws.sock.connected:
            try:
                msg = {
                    "type": "CAMERA_STATUS",
                    "cameraId": camera_id,
                    "status": status.upper(),
                    "reportedAt": datetime.now(timezone.utc).isoformat(),
                }
                self.ws.send(json.dumps(msg))
                self.logger.info(f"Status update sent for camera {camera_id}: {status.upper()}")
            except Exception as e:
                self.logger.error(f"Failed to send status update for camera {camera_id}: {e}")
        else:
            self.logger.warning(
                f"Cannot send status update. WebSocket is not connected for camera: {camera_id}"
            )

    def register_camera(self, camera_id, status: str = "ONLINE"):
        """Kamerayı aktif listesine kaydeder ve durumunu bildirir."""
        self._active_cameras[camera_id] = status
        self.send_camera_status(camera_id, status)

    def unregister_camera(self, camera_id):
        """Kamerayı listeden çıkarır ve OFFLINE bildirir."""
        self.send_camera_status(camera_id, "OFFLINE")
        self._active_cameras.pop(camera_id, None)

    # ------------------------------------------------------------------ #
    #  Heartbeat — periyodik kamera durum bildirimi
    # ------------------------------------------------------------------ #

    def _start_heartbeat(self):
        """Periyodik heartbeat döngüsünü başlatır."""
        self._stop_heartbeat()
        self._heartbeat_timer = threading.Timer(
            self._heartbeat_interval, self._send_heartbeat
        )
        self._heartbeat_timer.daemon = True
        self._heartbeat_timer.start()
        self.logger.debug(f"Heartbeat started ({self._heartbeat_interval}s interval).")

    def _send_heartbeat(self):
        """Tüm aktif kameraların durumunu periyodik olarak bildirir."""
        for cam_id, status in self._active_cameras.items():
            self.send_camera_status(cam_id, status)

        # Tekrar zamanla
        if self.should_reconnect:
            self._heartbeat_timer = threading.Timer(
                self._heartbeat_interval, self._send_heartbeat
            )
            self._heartbeat_timer.daemon = True
            self._heartbeat_timer.start()

    def _stop_heartbeat(self):
        """Heartbeat döngüsünü durdurur."""
        if self._heartbeat_timer:
            self._heartbeat_timer.cancel()
            self._heartbeat_timer = None

    # ------------------------------------------------------------------ #
    #  Kapatma
    # ------------------------------------------------------------------ #

    def stop(self):
        """Bağlantı döngüsünü durdurur ve mevcut socket'i kapatır."""
        self.logger.info("Shutting down WebSocket client...")
        self.should_reconnect = False
        self._stop_heartbeat()
        if self.ws:
            self.ws.close()