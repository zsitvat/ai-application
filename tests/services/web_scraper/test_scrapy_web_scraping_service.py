import os
import tempfile

import pytest

from src.services.web_scraper.scrapy_web_scraping_service import ScrapySpider


class DummyResponse:
    def __init__(self, url, meta=None, css_map=None):
        self.url = url
        self.meta = meta or {}
        self._css_map = css_map or {}

    def css(self, selector):
        return DummySelector(self._css_map.get(selector, []))


class DummySelector:
    def __init__(self, values):
        self._values = values

    def get(self):
        return self._values[0] if self._values else None

    def getall(self):
        return self._values


@pytest.fixture
def spider():
    return ScrapySpider(
        start_urls=["http://example.com"],
        max_depth=1,
        allowed_domains=["example.com"],
        content_selectors=["body"],
        excluded_selectors=["footer"],
    )


def test_extract_content_basic(spider):
    css_map = {
        "title::text": ["Test Title"],
        "body ::text": ["Hello World", "Footer text"],
        "footer ::text": ["Footer text"],
    }
    response = DummyResponse("http://example.com", css_map=css_map)
    spider._extract_content(response)
    assert "http://example.com" in spider.scraped_data
    assert "Test Title" in spider.scraped_data["http://example.com"]
    assert "Hello World" in spider.scraped_data["http://example.com"]
    assert "Footer text" not in spider.scraped_data["http://example.com"]


def test_filter_excluded_text(spider):
    css_map = {"footer ::text": ["Remove me"]}
    response = DummyResponse("http://example.com", css_map=css_map)
    result = spider._filter_excluded_text(response, ["Keep me", "Remove me"])
    assert "Keep me" in result
    assert "Remove me" not in result


def test_handle_error_adds_failed_url(spider):
    class DummyFailure:
        def __init__(self, url):
            self.request = type("req", (), {"url": url})
            self.value = Exception("fail")

    failure = DummyFailure("http://fail.com")
    spider.handle_error(failure)
    assert "http://fail.com" in spider.failed_urls


def test_save_scraped_content_to_file_txt_html_json_string(spider):
    spider.scraped_data = {"http://example.com": "Some content"}
    with tempfile.TemporaryDirectory() as tmpdir:
        # TXT
        txt_file = spider._save_scraped_content_to_file("txt", tmpdir)
        assert txt_file.endswith(".txt")
        assert os.path.exists(txt_file)
        # HTML
        html_file = spider._save_scraped_content_to_file("html", tmpdir)
        assert html_file.endswith(".html")
        assert os.path.exists(html_file)
        # JSON
        json_file = spider._save_scraped_content_to_file("json", tmpdir)
        assert json_file.endswith(".json")
        assert os.path.exists(json_file)
        # STRING
        string_content = spider._save_scraped_content_to_file("string", tmpdir)
        assert "Some content" in string_content


def test_save_scraped_content_to_file_docx_pdf(spider):
    spider.scraped_data = {"http://example.com": "Some content"}
    with tempfile.TemporaryDirectory() as tmpdir:
        # DOCX
        docx_file = spider._save_scraped_content_to_file("docx", tmpdir)
        assert docx_file.endswith(".docx")
        assert os.path.exists(docx_file)
        # PDF
        pdf_file = spider._save_scraped_content_to_file("pdf", tmpdir)
        assert pdf_file.endswith(".pdf")
        assert os.path.exists(pdf_file)


def test_save_scraped_content_to_file_empty(spider):
    spider.scraped_data = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        result = spider._save_scraped_content_to_file("txt", tmpdir)
        assert result == ""


def test_save_scraped_content_to_file_unsupported(spider):
    spider.scraped_data = {"http://example.com": "Some content"}
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            spider._save_scraped_content_to_file("unsupported", tmpdir)
        except ValueError as e:
            assert "Unsupported output type" in str(e)


def test_get_all_content(spider):
    spider.scraped_data = {
        f"http://example{i}.com": f"Example {i} tartalom" for i in range(1, 24)
    }
    result = spider._get_all_content()
    for i in range(1, 24):
        assert f"example{i}.com" in result


def test_extract_content_error_branch(spider, caplog):
    class ErrorResponse:
        url = "http://error.com"

        def css(self, selector):
            raise ValueError("Value error")

    with caplog.at_level("ERROR"):
        spider._extract_content(ErrorResponse())
    assert "Error extracting content" in caplog.text
    assert "http://error.com" in spider.failed_urls


def test_extract_from_content_selectors_empty(spider):
    class DummyResponse:
        def css(self, selector):
            return None

    content_parts = []
    spider._extract_from_content_selectors(DummyResponse(), content_parts)
    assert content_parts == []


def test_filter_excluded_text_empty_list(spider):
    class DummyResponse:
        def css(self, selector):
            return DummySelector([])

    result = spider._filter_excluded_text(DummyResponse(), [])
    assert result == []


def test_prepare_scraping_config_variants(spider):
    config = spider._prepare_scraping_config(["main"], ["footer"])
    assert config == {"content_selectors": ["main"], "excluded_selectors": ["footer"]}
    config = spider._prepare_scraping_config(None, ["footer"])
    assert config == {"excluded_selectors": ["footer"]}
    config = spider._prepare_scraping_config(["main"], None)
    assert config == {"content_selectors": ["main"]}
    config = spider._prepare_scraping_config(None, None)
    assert config is None


def test_get_scraped_content_as_string_and_get_content_as_string(spider):

    spider.scraped_data = {
        "http://example-1.com": "Example 1 content",
        "http://example-2.com": "Example 2 content",
    }
    s1 = spider.get_scraped_content_as_string()
    s2 = spider.get_content_as_string(spider.scraped_data)
    assert "example-1.com" in s1 and "example-2.com" in s2
    assert "Example 1 content" in s1 and "Example 2 content" in s2


def test_save_to_file_empty_and_error(monkeypatch):
    spider = ScrapySpider()
    # Empty scraped_data
    result = spider.save_to_file({}, "txt", None)
    assert result == ""
    # Error branch: unsupported type
    spider.scraped_data = {"https://github.com": "GitHub tartalom"}

    def raise_unsupported_output_type(*args, **kwargs):
        raise ValueError("Unsupported output type")

    monkeypatch.setattr(
        spider,
        "_save_scraped_content_to_file",
        raise_unsupported_output_type,
    )
    try:
        spider.save_to_file(spider.scraped_data, "unsupported", None)
    except ValueError as e:
        assert "Unsupported output type" in str(e)


def test_extract_json_from_output_valid_and_invalid():
    spider = ScrapySpider()
    # Valid JSON line
    output = '{"a": 1}\nSome log\n'
    result = spider._extract_json_from_output(output)
    assert result == {"a": 1}
    # Valid JSON array
    output = "[1,2,3]\nLog\n"
    result = spider._extract_json_from_output(output)
    assert result == [1, 2, 3]
    # Invalid JSON
    try:
        spider._extract_json_from_output("no json here")
    except ValueError as e:
        assert "No valid JSON" in str(e)
