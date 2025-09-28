from bs4 import BeautifulSoup

from src.services.graph.tools.document_link_analyzer_tool import (
    DocumentLinkAnalyzerTool,
)


def test_extract_main_content_basic():
    html = """
    <html><body><main>Important content here.</main></body></html>
    """
    soup = BeautifulSoup(html, "html.parser")
    tool = DocumentLinkAnalyzerTool()
    result = tool._extract_main_content(soup)
    assert "Important content here." in result


def test_extract_main_content_no_main():
    html = """
    <html><body><div>Fallback body content.</div></body></html>
    """
    soup = BeautifulSoup(html, "html.parser")
    tool = DocumentLinkAnalyzerTool()
    result = tool._extract_main_content(soup)
    assert "Fallback body content." in result


def test_format_url_result():
    tool = DocumentLinkAnalyzerTool()
    content_data = {
        "title": "Test Title",
        "description": "Test Description",
        "content": "Test Content" * 50,
        "url": "http://test.com",
        "status": "success",
    }
    result = tool._format_url_result(1, "http://test.com", content_data, True)
    assert "Test Title" in result
    assert "Test Description" in result
    assert "Summary:" in result


def test_validate_url():
    tool = DocumentLinkAnalyzerTool()
    assert tool._validate_url("https://example.com") is True
    assert tool._validate_url("not_a_url") is False
