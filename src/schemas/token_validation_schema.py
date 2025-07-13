from pydantic import BaseModel, Field


class TokenCountResult(BaseModel):
    """Result of token counting operation."""

    text: str
    token_count: int
    character_count: int
    encoding_used: str
    chars_per_token: float


class TruncationResult(BaseModel):
    """Result of text truncation operation."""

    original_text: str
    truncated_text: str
    original_tokens: int
    final_tokens: int
    was_truncated: bool
    encoding_used: str


class TokenEstimationResult(BaseModel):
    """Result of token estimation operation."""

    estimated_characters: int
    target_tokens: int
    encoding_used: str
    chars_per_token_ratio: float
