import requests
import os
import logging
from dotenv import load_dotenv
from typing import Dict, Optional

# .env dosyasını yükle
load_dotenv()

class CMSApiClient:
    def __init__(self):
        self.base_url = os.getenv("CMS_REST_URL", "http://localhost:8050").rstrip("/")
        self.subsystem_id = os.getenv("SUBSYSTEM_ID")
        self.subsystem_secret = os.getenv("SUBSYSTEM_SECRET")
        self.token = None
        self.logger = logging.getLogger("CMSApiClient")
        self.session = requests.Session()

    def login(self, force: bool = False) -> bool:
        """
        CMS'ten JWT token alır.
        force=False ise ve zaten token varsa tekrar istek atmaz.
        """
        if self.token and not force:
            return True

        url = f"{self.base_url}/api/auth/subsystem-login"
        payload = {"subsystemId": self.subsystem_id, "subsystemSecret": self.subsystem_secret}
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                self.token = response.json().get("token")
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
                self.token = None  # Token'ı temizle ki login(force=True) yeni istek atsın
                if self.login(force=True):
                    return self.session.request(method, url, **kwargs)
                
            return response
        except requests.RequestException as e:
            self.logger.error(f"Request error for {url}: {e}")
            return None

    def ingest_event(self, event_data: Dict) -> Optional[Dict]:
        """
        Anomali olayını (metadata) CMS'e bildirir.
        
          - 201 Created: Olay başarıyla kaydedildi → response body döner
          - 409 Conflict: Aynı sourceEventId ile tekrar gönderildi → başarılı kabul edilir

        Response body:
          {
            "eventId": 42,
            "status": "CREATED",
            "clipObjectKey": "cameras/5/events/42.mp4",
            "clipUploadUrl": "http://...",
            "clipUploadExpiresInSeconds": 300
          }

        Returns:
            Başarılı ise response body (dict), başarısız ise None.
        """
        url = f"{self.base_url}/api/events/ingest"
        response = self._request_with_retry("POST", url, json=event_data, timeout=10)

        if response is None:
            self.logger.error("Event ingestion failed: no response from CMS.")
            return None

        # 201 Created — başarılı kayıt
        if response.status_code == 201:
            data = response.json()
            self.logger.info(
                f"Event ingested successfully: "
                f"eventId={data.get('eventId')}, "
                f"sourceEventId={event_data.get('sourceEventId')}"
            )
            return data

        # 409 Conflict — idempotent tekrar, başarılı kabul et
        if response.status_code == 409:
            self.logger.info(
                f"Event already ingested (409 idempotent): "
                f"sourceEventId={event_data.get('sourceEventId')}"
            )
            # 409 response'da da body dönebilir
            try:
                return response.json()
            except Exception:
                return {"status": "DUPLICATE"}

        self.logger.error(
            f"Event ingestion failed! "
            f"Status: {response.status_code}, Body: {response.text[:200]}"
        )
        return None