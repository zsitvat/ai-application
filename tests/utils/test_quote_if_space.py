from src.utils.quote_if_space import quote_if_space


def test_quote_if_space():
    assert quote_if_space("hello world") == '"hello world"'
    assert quote_if_space("hello") == "hello"


def test_quote_if_space_empty_string():
    assert quote_if_space("") == ""


def test_quote_if_space_spaces_only():
    assert quote_if_space("   ") == '"   "'


def test_quote_if_space_numeric():
    assert quote_if_space(str(123)) == "123"
