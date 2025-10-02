from src.schemas.web_scraping_schema import (
    OutputType,
    WebScrapingRequestSchema,
    WebScrapingResponseSchema,
)


def test_web_scraping_request_schema():
    """Test WebScrapingRequestSchema instantiation and field values."""
    from schemas.model_schema import Model, ModelProviderType, ModelType

    model = Model(
        provider=ModelProviderType.OPENAI,
        name="text-embedding-3-large",
        type=ModelType.EMBEDDING,
    )
    obj = WebScrapingRequestSchema(
        urls=["https://example.com"],
        max_depth=2,
        output_type=OutputType.JSON,
        output_path="/tmp/output.json",
        vector_db_index="index1",
        allowed_domains=["example.com"],
        content_selectors=[".main-content"],
        excluded_selectors=[".ads"],
        embedding_model=model,
    )
    assert obj.urls == ["https://example.com"]
    assert obj.max_depth == 2
    assert obj.output_type == OutputType.JSON
    assert obj.output_path == "/tmp/output.json"
    assert obj.vector_db_index == "index1"
    assert obj.allowed_domains == ["example.com"]
    assert obj.content_selectors == [".main-content"]
    assert obj.excluded_selectors == [".ads"]
    # The default value for embedding_model is a Model object, not None
    assert isinstance(obj.embedding_model, Model)
    assert obj.embedding_model.provider == ModelProviderType.OPENAI
    assert obj.embedding_model.name == "text-embedding-3-large"
    assert obj.embedding_model.type == ModelType.EMBEDDING


def test_web_scraping_response_schema():
    """Test WebScrapingResponseSchema instantiation and field values."""
    obj = WebScrapingResponseSchema(
        success=True,
        message="Scraping completed",
        scraped_urls=["https://example.com"],
        failed_urls=[],
        content="<html>...</html>",
    )
    assert obj.success is True
    assert obj.message == "Scraping completed"
    assert obj.scraped_urls == ["https://example.com"]
    assert obj.failed_urls == []
    assert obj.content == "<html>...</html>"
