def str2bool(value: str) -> bool:
    """Convert string to boolean.

    Args:
        v: String value to convert to boolean

    Returns:
        bool: Converted boolean value

    Raises:
        ValueError: If the string cannot be converted to a boolean
    """
    if not isinstance(value, str):
        raise ValueError(f"Expected string input, got {type(value)}")

    v_lower = value.lower().strip()

    if v_lower in ("true", "1", "yes", "on"):
        return True
    elif v_lower in ("false", "0", "no", "off", ""):
        return False
    else:
        raise ValueError(f"Cannot convert string '{value}' to boolean")
