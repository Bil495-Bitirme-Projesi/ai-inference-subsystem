import websocket
import json
import threading

class CMSWebSocketClient:
    def __init__(self, ws_url: str, token: str, on_config_sync: callable):
        self.ws_url = f"{ws_url}?token={token}"
        self.on_config_sync = on_config_sync # Orchestrator'a bağlı callback
        self.ws = None

    def connect(self):
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self.on_message,
            on_open=self.on_open
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def on_open(self, ws):
        # Bağlantı açıldığında tam konfigürasyon (SNAPSHOT) istenir
        self.ws.send(json.dumps({"type": "SNAPSHOT_REQUEST"}))

    def on_message(self, ws, message):
        data = json.loads(message)
        if data["type"] in ["SNAPSHOT", "CAMERA_DELTA"]:
            self.on_config_sync(data) # Kamera listesini güncelle

    def send_camera_status(self, camera_id: str, status: str):
        """ONLINE/OFFLINE bilgisini gönderir"""
        msg = {"type": "CAMERA_STATUS", "cameraId": camera_id, "status": status}
        self.ws.send(json.dumps(msg))