import websocket
import json
import threading
import time
import os
import logging
from dotenv import load_dotenv

# .env değişkenlerini yükle
load_dotenv()

class CMSWebSocketClient:
    def __init__(self, api_client, on_config_sync: callable):
        """
        api_client: Kimlik doğrulama yönetimi için CMSApiClient instance'ı.
        on_config_sync: SNAPSHOT ve CAMERA_DELTA mesajlarını işleyecek callback fonksiyonu.
        """
        self.api_client = api_client
        self.on_config_sync = on_config_sync
        self.ws_base_url = os.getenv("CMS_WS_URL")
        self.ws = None
        self.logger = logging.getLogger("CMSWebSocketClient")
        self.should_reconnect = True
        self.reconnect_delay = 5  # saniye cinsinden yeniden bağlanma süresi

    def _ensure_connection_url(self) -> str:
        """
        API client'ın geçerli bir tokenı olduğundan emin olur ve WS URL'ini oluşturur.
        """

        self.logger.info("Attempting to login before connecting to WebSocket...")
        login_success = self.api_client.login()
            
        if not login_success:
            # Login başarısızsa hata fırlatılır, reconnect döngüsü bunu yakalar
            raise ConnectionError("Authentication failed: Could not acquire JWT token.")
        
        # Güncel token ile tam URL oluşturulur
        return f"{self.ws_base_url}?token={self.api_client.token}"

    def connect(self):
        """WebSocket bağlantısını arka planda bir thread içinde başlatır."""
        threading.Thread(target=self._run_forever_loop, daemon=True).start()

    def _run_forever_loop(self):
        """Bağlantı koptuğunda otomatik yeniden bağlanmayı sağlayan ana döngü."""
        while self.should_reconnect:
            try:
                # Bağlanmadan önce token'dan emin ol ve URL'i al
                url = self._ensure_connection_url()
                
                self.logger.info(f"Establishing WebSocket connection to: {self.ws_base_url}")
                
                self.ws = websocket.WebSocketApp(
                    url,
                    on_message=self.on_message,
                    on_open=self.on_open,
                    on_error=self.on_error,
                    on_close=self.on_close
                )
                
                # run_forever bağlantı kesilene kadar thread'i bloklar
                self.ws.run_forever()
                
            except Exception as e:
                self.logger.error(f"WebSocket connection loop error: {e}")
            
            # Bağlantı koptuysa bekle ve tekrar dene
            if self.should_reconnect:
                self.logger.info(f"Attempting to reconnect in {self.reconnect_delay} seconds...")
                time.sleep(self.reconnect_delay)

    def on_open(self, ws):
        """Bağlantı başarılı bir şekilde açıldığında tetiklenir."""
        self.logger.info("WebSocket connection opened successfully.")
        
        # Kontrat gereği bağlantı açılınca SNAPSHOT_REQUEST gönderilir
        try:
            request_msg = json.dumps({"type": "SNAPSHOT_REQUEST"})
            self.ws.send(request_msg)
            self.logger.info("SNAPSHOT_REQUEST sent to CMS.")
        except Exception as e:
            self.logger.error(f"Failed to send initial snapshot request: {e}")

    def on_message(self, ws, message):
        """CMS'ten gelen mesajları ayrıştırır ve ilgili callback'e gönderir."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            self.logger.debug(f"Received message type: {msg_type}")

            # Sadece konfigürasyonla ilgili mesajlar Orchestrator'a iletilir
            if msg_type in ["SNAPSHOT", "CAMERA_DELTA"]:
                self.logger.info(f"Config update received via {msg_type}")
                self.on_config_sync(data)
            else:
                self.logger.warning(f"Unrecognized message type received: {msg_type}")
                
        except json.JSONDecodeError:
            self.logger.error("Received an invalid JSON message from WebSocket.")
        except Exception as e:
            self.logger.error(f"Error while processing WebSocket message: {e}")

    def on_error(self, ws, error):
        """WebSocket kütüphanesinden gelen hataları loglar."""
        self.logger.error(f"WebSocket execution error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Bağlantı kapandığında tetiklenir."""
        self.logger.warning(f"WebSocket connection closed. Code: {close_status_code}, Message: {close_msg}")

    def send_camera_status(self, camera_id: str, status: str):
        """
        Kameranın ONLINE/OFFLINE durumunu CMS'e iletir.
        """
        # Socket'in o an bağlı olup olmadığını kontrol eder
        if self.ws and self.ws.sock and self.ws.sock.connected:
            try:
                msg = {
                    "type": "CAMERA_STATUS", 
                    "cameraId": camera_id, 
                    "status": status.upper()
                }
                self.ws.send(json.dumps(msg))
                self.logger.info(f"Status update sent for camera {camera_id}: {status.upper()}")
            except Exception as e:
                self.logger.error(f"Failed to send status update for camera {camera_id}: {e}")
        else:
            self.logger.warning(f"Cannot send status update. WebSocket is not connected for camera: {camera_id}")

    def stop(self):
        """Bağlantı döngüsünü durdurur ve mevcut socket'i kapatır."""
        self.logger.info("Shutting down WebSocket client...")
        self.should_reconnect = False
        if self.ws:
            self.ws.close()