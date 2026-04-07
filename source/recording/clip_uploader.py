"""
ClipUploader — Kaydedilmiş video klibini presigned PUT URL ile MinIO'ya yükler.

AIS ↔ CMS kontratına göre:
  1. POST /api/events/ingest  →  clipUploadUrl döner
  2. PUT  <clipUploadUrl>      →  video/mp4 body ile yüklenir
  3. Auth header gerekmez (imza URL içinde gömülüdür)

Kullanım:
    uploader = ClipUploader()
    success = uploader.upload("clips/cam5_evt42.mp4", clip_upload_url)
"""

import os
import logging
from typing import Optional

import requests


class ClipUploader:
    """
    Presigned PUT URL üzerinden MinIO'ya video klip yükler.

    Deploy stack'te HTTPS + self-signed sertifika kullanıyorsa
    cert_path parametresi ile sertifika dosyası verilmelidir.
    """

    def __init__(
        self,
        timeout: int = 120,
        cert_path: Optional[str] = None,
    ):
        """
        Args:
            timeout:   Upload isteği zaman aşımı (saniye).
            cert_path: Self-signed sertifika dosya yolu (deploy stack için).
                       Örn: "nginx/certs/server.crt"
                       None ise varsayılan SSL doğrulama kullanılır.
        """
        self.timeout = timeout
        self.cert_path = cert_path
        self.logger = logging.getLogger("ClipUploader")

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def upload(
        self,
        clip_path: str,
        upload_url: str,
        cleanup: bool = True,
    ) -> bool:
        """
        Klip dosyasını presigned URL'e PUT eder.

        Args:
            clip_path:  Yüklenecek .mp4 dosyanın yolu.
            upload_url: CMS'den alınan presigned PUT URL.
            cleanup:    Başarılı yükleme sonrası dosyayı silmek için True.

        Returns:
            Başarılı ise True, değilse False.
        """
        if not os.path.exists(clip_path):
            self.logger.error(f"Clip file not found: {clip_path}")
            return False

        file_size = os.path.getsize(clip_path)
        self.logger.info(
            f"Uploading clip: {clip_path} ({file_size} bytes) -> {upload_url[:80]}..."
        )

        verify = self.cert_path if self.cert_path else True

        try:
            with open(clip_path, "rb") as f:
                response = requests.put(
                    upload_url,
                    data=f,
                    headers={"Content-Type": "video/mp4"},
                    verify=verify,
                    timeout=self.timeout,
                )

            if response.status_code == 200:
                self.logger.info(f"Clip uploaded successfully: {clip_path}")
                if cleanup:
                    self._remove_file(clip_path)
                return True

            self.logger.error(
                f"Clip upload failed — "
                f"HTTP {response.status_code}: {response.text[:200]}"
            )
            return False

        except requests.Timeout:
            self.logger.error(
                f"Clip upload timed out ({self.timeout}s): {clip_path}"
            )
            return False
        except requests.ConnectionError as e:
            self.logger.error(f"MinIO connection error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected upload error: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  Yardımcı
    # ------------------------------------------------------------------ #

    def _remove_file(self, path: str) -> None:
        """Dosyayı sessizce siler."""
        try:
            os.remove(path)
            self.logger.debug(f"Temporary clip file removed: {path}")
        except OSError as e:
            self.logger.warning(f"Could not remove file: {path} — {e}")
