import requests
import os
import logging
from dotenv import load_dotenv
from typing import Dict, Optional

# .env dosyasını yükle
load_dotenv()

class CMSApiClient:
    def __init__(self):
        self.base_url = os.getenv("CMS_BASE_URL", "http://localhost:8050").rstrip("/")
        self.subsystem_id = os.getenv("SUBSYSTEM_ID")
        self.subsystem_secret = os.getenv("SUBSYSTEM_SECRET")
        self.token = None
        self.logger = logging.getLogger("CMSApiClient")
        self.session = requests.Session()

    def login(self) -> bool:
        """CMS'ten JWT token alır"""
        url = f"{self.base_url}/api/auth/subsystem-login"
        payload = {"subsystemId": self.subsystem_id, "subsystemSecret": self.subsystem_secret}
        try:
            response = self.session.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                self.token = response.json().get("token")
                # Session'a tokenı gömüyoruz, sonraki tüm isteklerde otomatik gidecek
                self.session.headers.update({"Authorization": f"Bearer {self.token}"})
                self.logger.info("CMS login successful. Token acquired.")
                return True
            else:
                self.logger.error(f"Login failed! Status: {response.status_code}, Body: {response.text}")
                return False
        except requests.RequestException as e:
            self.logger.error(f"Network error during login: {e}")
            return False

    def _request_with_retry(self, method: str, url: str, **kwargs):
        """
        Token süresi dolduğunda (401) otomatik login olup isteği tekrar dener.
        """
        try:
            response = self.session.request(method, url, **kwargs)
            # Eğer token eskidiyse (401), login olup tekrar dene
            if response.status_code == 401:
                self.logger.warning("Token expired (401). Attempting to re-login...")
                if self.login():
                    return self.session.request(method, url, **kwargs)
                
            return response
        except requests.RequestException as e:
            self.logger.error(f"Request error for {url}: {e}")
            return None

    def get_upload_url(self, camera_id: str, event_id: str) -> Optional[str]:
        """
        MinIO için presigned PUT URL alır.
        """
        url = f"{self.base_url}/api/clips/upload-url"
        params = {"cameraId": camera_id, "sourceEventId": event_id}
        response = self._request_with_retry("GET", url, params=params, timeout=10)
        if response and response.status_code == 200:
            return response.json().get("uploadUrl")
        
        self.logger.error(f"Could not get upload URL for camera {camera_id}")
        return None

    def ingest_event(self, event_data: Dict) -> bool:
        """
        Anomali olayını (metadata) CMS'e bildirir.
        """
        url = f"{self.base_url}/api/events/ingest"
        response = self._request_with_retry("POST", url, json=event_data, timeout=10)
        if response and response.status_code == 201:
            self.logger.info(f"Event successfully ingested: {event_data.get('sourceEventId')}")
            return True
        self.logger.error(f"Event ingestion failed! Status: {response.status_code if response else 'No Response'}")
        return False