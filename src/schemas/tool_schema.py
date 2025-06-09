from pydantic import BaseModel


class SearchKwargs(BaseModel):
    k: int
    threshold: float = 0.5
    search_type: str = "similarity"


class Tool(BaseModel):
    name: str


class WebSearchTool(Tool):
    engine: str = "google"
