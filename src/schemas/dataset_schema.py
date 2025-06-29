from uuid import UUID

from pydantic import BaseModel, Field


class DatasetRequestSchema(BaseModel):
    dataset_name: str
    description: str | None = None
    test_cases: list[dict] | None = Field(
        default=None,
        examples=[
            [
                {
                    "inputs": {"question": ""},
                    "outputs": {"answer": ""},
                }
            ]
        ],
    )


class ApplicationIdentifierSchema(BaseModel):
    tenantIdentifier: int
    applicationIdentifier: int


class DatasetRunConfigSchema(BaseModel):
    endpoint: str | None = None
    uuid: UUID | str | None = None
    applicationIdentifier: ApplicationIdentifierSchema | None = None
    platform: str | None = None
    context: dict | None = None
    parameters: dict | None = None


class DatasetRunRequestSchema(BaseModel):
    config: DatasetRunConfigSchema | None = None


class DatasetResponseSchema(BaseModel):
    id: str | None = None
    name: str
    description: str | None = None
    test_cases: list[dict]
    created_at: str
    updated_at: str


class DatasetNotFoundError(Exception):
    pass


class DatasetCreationError(Exception):
    pass


class DatasetUpdateError(Exception):
    pass


class DatasetRunError(Exception):
    pass
