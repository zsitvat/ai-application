from uuid import UUID

from pydantic import BaseModel, Field

from .graph_schema import ApplicationIdentifierSchema, PlatformType


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


class DatasetRunConfigSchema(BaseModel):
    endpoint: str | None = None
    uuid: UUID | str | None = None
    applicationIdentifier: ApplicationIdentifierSchema | None = None
    platform: PlatformType | None = None
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
