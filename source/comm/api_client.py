import requests
import os
from typing import Dict, Optional

class CMSApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.token = None
        self.subsystem_id = os.getenv("SUBSYSTEM_ID")
        self.subsystem_secret = os.getenv("SUBSYSTEM_SECRET")

    def login(self) -> bool:
        """CMS'ten JWT token alır"""
        url = f"{self.base_url}/api/auth/subsystem-login"
        payload = {"subsystemId": self.subsystem_id, "subsystemSecret": self.subsystem_secret}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            self.token = response.json().get("token")
            return True
        return False

    def get_upload_url(self, camera_id: str, event_id: str) -> Optional[str]:
        """MinIO için presigned PUT URL alır"""
        headers = {"Authorization": f"Bearer {self.token}"}
        url = f"{self.base_url}/api/clips/upload-url"
        params = {"cameraId": camera_id, "sourceEventId": event_id}
        resp = requests.get(url, headers=headers, params=params)
        return resp.json().get("uploadUrl") if resp.status_code == 200 else None

    def ingest_event(self, event_data: Dict):
        """Anomali olayını CMS'e bildirir"""
        headers = {"Authorization": f"Bearer {self.token}"}
        url = f"{self.base_url}/api/events/ingest"
        return requests.post(url, json=event_data, headers=headers)