from src.utils.extract_message_content import extract_message_content


def test_extract_message_content_basic():

    result = extract_message_content("hello world")
    assert result == "hello world"

    result = extract_message_content({"content": "test"})
    assert result == "test"


def test_extract_message_content_empty():
    result = extract_message_content("")
    assert result == ""


def test_extract_message_content_none():
    result = extract_message_content(None)
    assert result is None or result == "None"


def test_extract_message_content_dict_no_content():
    result = extract_message_content({"other": "value"})
    assert result is None or result == "None" or result == str({"other": "value"})
