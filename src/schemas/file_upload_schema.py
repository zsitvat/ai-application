from pydantic import BaseModel, Field


class FileUploadResponse(BaseModel):
    """Response model for file upload operations."""

    file_id: str = Field(description="Unique identifier for the uploaded file")
    filename: str = Field(description="Original filename")
    file_path: str = Field(description="Path or URL where the file is stored")
    file_size: int = Field(ge=0, description="File size in bytes")
    mime_type: str = Field(description="MIME type of the file")
    storage_backend: str = Field(description="Storage backend used (local, s3, drive)")


class Base64FileUpload(BaseModel):
    """Model for base64 encoded file upload requests."""

    filename: str = Field(min_length=1, description="Original filename")
    mime_type: str = Field(min_length=1, description="MIME type of the file")
    file_data: str = Field(min_length=1, description="Base64 encoded file data")
    thread_id: str | None = Field(default=None, description="Thread ID for filename")


class StorageBackendConfig(BaseModel):
    """Configuration for different storage backends."""

    # Local storage config
    local_upload_dir: str = Field(
        default="files/uploads", description="Local upload directory"
    )

    # S3 config
    s3_bucket: str | None = Field(default=None, description="AWS S3 bucket name")
    s3_region: str = Field(default="us-east-1", description="AWS S3 region")

    # Google Drive config
    drive_folder_id: str | None = Field(
        default=None, description="Google Drive folder ID"
    )
    service_account_file: str | None = Field(
        default=None, description="Google Service Account file path"
    )

    # File size limitss
    max_file_size_mb: int = Field(
        default=15, ge=1, le=20, description="Maximum file size in MB"
    )
