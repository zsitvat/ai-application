from unittest.mock import patch

from src.services.graph.tools.document_link_analyzer_tool import (
    DocumentLinkAnalyzerTool,
)


def test_extract_urls():
    tool = DocumentLinkAnalyzerTool()
    text = "Check https://example.com and http://test.com for info."
    urls = tool._extract_urls(text)
    assert "https://example.com" in urls
    assert "http://test.com" in urls


def test_run_no_urls():
    tool = DocumentLinkAnalyzerTool()
    result = tool._run("No links here.")
    assert "No URLs found" in result


def test_run_limit_links():
    tool = DocumentLinkAnalyzerTool()
    text = " ".join([f"https://site{i}.com" for i in range(20)])
    with patch.object(
        tool,
        "_fetch_url_content",
        return_value={"title": "Title", "content": "Content"},
    ):
        result = tool._run(text, max_links=5)
        assert "Found 5 URL(s)" in result


def test_run_fetch_error():
    tool = DocumentLinkAnalyzerTool()
    text = "https://error.com"
    with patch.object(tool, "_fetch_url_content", side_effect=Exception("fail")):
        result = tool._run(text)
        assert "Error: fail" in result


def test_run_success():
    tool = DocumentLinkAnalyzerTool()
    text = "https://success.com"
    with patch.object(
        tool,
        "_fetch_url_content",
        return_value={"title": "Title", "content": "Content"},
    ):
        result = tool._run(text)
        assert "Found 1 URL(s)" in result
        assert "Error: 'description'" in result
