from pydantic import BaseModel


class DatasetCreateRequestSchema(BaseModel):
    dataset_name: str
    description: str | None = None
    test_cases: list[dict]


class DatasetUpdateRequestSchema(BaseModel):
    dataset_name: str
    description: str | None = None
    test_cases: list[dict] | None = None


class DatasetRunRequestSchema(BaseModel):
    dataset_name: str


class DatasetResponseSchema(BaseModel):
    name: str
    description: str | None = None
    test_cases: list[dict]
    created_at: str
    updated_at: str
