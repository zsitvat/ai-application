import base64
import io
import os
import uuid
from datetime import datetime
from pathlib import Path

import aiofiles
import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as OAuthCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

from src.schemas.file_upload_schema import FileUploadResponse, StorageBackendConfig
from src.services.logger.logger_service import LoggerService


class FileUploadService:
    """Service for handling file upload operations."""

    def __init__(self, config: StorageBackendConfig | None = None) -> None:
        """Initialize the file upload service.

        Args:
            config: Storage backend configuration
        """
        self.config = config or StorageBackendConfig()
        self.logger = LoggerService().setup_logger()

    def validate_file_size(self, file_size: int) -> None:
        """Validate file size against configured limits.

        Args:
            file_size: Size of the file in bytes

        Raises:
            HTTPException: If file is too large
        """
        max_size = self.config.max_file_size_mb * 1024 * 1024
        if file_size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {self.config.max_file_size_mb}MB",
            )

    async def save_to_local_storage(
        self, file_data: bytes, filename: str, mime_type: str
    ) -> FileUploadResponse:
        """Save file to local filesystem.

        Args:
            file_data: File content as bytes
            filename: Original filename
            mime_type: MIME type of the file

        Returns:
            FileUploadResponse with file details
        """
        uploads_dir = Path(self.config.local_upload_dir)
        uploads_dir.mkdir(parents=True, exist_ok=True)

        file_id = str(uuid.uuid4())
        file_extension = Path(filename).suffix
        safe_filename = f"{file_id}{file_extension}"
        file_path = uploads_dir / safe_filename
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_data)

        self.logger.info(f"File saved to local storage: {file_path}")

        return FileUploadResponse(
            file_id=file_id,
            filename=filename,
            file_path=str(file_path),
            file_size=len(file_data),
            mime_type=mime_type,
            storage_backend="local",
        )

    def save_to_s3(
        self, file_data: bytes, filename: str, mime_type: str
    ) -> FileUploadResponse:
        """Save file to AWS S3.

        Args:
            file_data: File content as bytes
            filename: Original filename
            mime_type: MIME type of the file

        Returns:
            FileUploadResponse with file details
        """

        bucket_name = self.config.s3_bucket or os.getenv("AWS_S3_BUCKET")
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = self.config.s3_region

        if not all([bucket_name, aws_access_key, aws_secret_key]):
            raise HTTPException(
                status_code=500,
                detail="AWS S3 configuration incomplete. Check environment variables.",
            )

        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region,
        )

        file_id = str(uuid.uuid4())
        file_extension = Path(filename).suffix
        s3_key = f"uploads/{file_id}{file_extension}"

        try:
            s3_client.put_object(
                Bucket=bucket_name, Key=s3_key, Body=file_data, ContentType=mime_type
            )

            s3_url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{s3_key}"
            self.logger.info(f"File uploaded to S3: {s3_url}")

            return FileUploadResponse(
                file_id=file_id,
                filename=filename,
                file_path=s3_url,
                file_size=len(file_data),
                mime_type=mime_type,
                storage_backend="s3",
            )

        except ClientError as e:
            self.logger.error(f"S3 upload error: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to upload to S3: {str(e)}"
            )

    def save_to_google_drive(
        self, file_data: bytes, filename: str, mime_type: str, thread_id: str = None
    ) -> FileUploadResponse:
        """Save file to Google Drive with filename as thread id and date."""
        oauth_credentials_file = self.config.service_account_file or os.getenv(
            "GOOGLE_OAUTH_CREDENTIALS_FILE"
        )
        drive_folder_id = self.config.drive_folder_id or os.getenv(
            "GOOGLE_DRIVE_FOLDER_ID"
        )

        SCOPES = ["https://www.googleapis.com/auth/drive.file"]

        creds = None
        token_file = "token.json"

        if os.path.exists(token_file):
            creds = OAuthCredentials.from_authorized_user_file(token_file, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    oauth_credentials_file, SCOPES
                )
                creds = flow.run_local_server(port=0)

            with open(token_file, "w") as token:
                token.write(creds.to_json())

        service = build("drive", "v3", credentials=creds)

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        thread_part = thread_id if thread_id else "no_thread"
        file_extension = Path(filename).suffix
        drive_filename = f"{thread_part}_{now_str}{file_extension}"
        file_metadata = {
            "name": drive_filename,
        }

        if drive_folder_id:
            file_metadata["parents"] = [drive_folder_id]

        media = MediaIoBaseUpload(
            io.BytesIO(file_data), mimetype=mime_type, resumable=True
        )

        file_result = (
            service.files()
            .create(
                body=file_metadata,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            )
            .execute()
        )

        drive_file_id = file_result.get("id")
        drive_url = f"https://drive.google.com/file/d/{drive_file_id}/view"

        self.logger.info(f"File uploaded to Google Drive: {drive_url}")

        return FileUploadResponse(
            file_id=drive_file_id,
            filename=drive_filename,
            file_path=drive_url,
            file_size=len(file_data),
            mime_type=mime_type,
            storage_backend="drive",
        )

    def _get_configured_storage_backend(self) -> str:
        """Determine which storage backend to use based on environment configuration.

        Returns:
            Storage backend name (local, s3, drive)
        """
        if (
            os.getenv("AWS_S3_BUCKET")
            and os.getenv("AWS_ACCESS_KEY_ID")
            and os.getenv("AWS_SECRET_ACCESS_KEY")
        ):
            return "s3"

        if os.getenv("GOOGLE_OAUTH_CREDENTIALS_FILE"):
            return "drive"
        return "local"

    async def upload_file(
        self, file_data: bytes, filename: str, mime_type: str, thread_id: str = None
    ) -> FileUploadResponse:
        """Upload file using configured storage backend.

        Args:
            file_data: File content as bytes
            filename: Original filename
            mime_type: MIME type of the file

        Returns:
            FileUploadResponse with file details

        Raises:
            HTTPException: If upload fails
        """
        self.validate_file_size(len(file_data))

        storage_backend = self._get_configured_storage_backend()
        if storage_backend == "local":
            return await self.save_to_local_storage(file_data, filename, mime_type)
        elif storage_backend == "s3":
            return self.save_to_s3(file_data, filename, mime_type)
        elif storage_backend == "drive":
            return self.save_to_google_drive(file_data, filename, mime_type, thread_id)
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Unsupported storage backend: {storage_backend}",
            )

    async def upload_base64_file(
        self, filename: str, base64_data: str, thread_id: str = None
    ) -> FileUploadResponse:
        """Upload base64 encoded file using configured storage backend.

        Args:
            filename: Original filename
            base64_data: Base64 encoded file data

        Returns:
            FileUploadResponse with file details

        Raises:
            HTTPException: If base64 data is invalid or upload fails
        """
        try:
            file_content = base64.b64decode(base64_data)
            mime_type = "application/octet-stream"
            if "," in base64_data:
                header = base64_data.split(",")[0]
                if "data:" in header and ";" in header:
                    mime_type = header.split("data:")[1].split(";")[0]
                    file_content = base64.b64decode(base64_data.split(",")[1])

            return await self.upload_file(file_content, filename, mime_type, thread_id)

        except base64.binascii.Error:
            raise HTTPException(status_code=400, detail="Invalid base64 data")
