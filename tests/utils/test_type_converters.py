import pytest

from src.utils.type_converters import str2bool


def test_str2bool():
    assert str2bool("True") is True
    assert str2bool("false") is False
    assert str2bool("1") is True
    assert str2bool("0") is False
    with pytest.raises(ValueError):
        str2bool("notabool")
