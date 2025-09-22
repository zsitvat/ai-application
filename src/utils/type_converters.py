def str2bool(v: str) -> bool:
    """Convert string to boolean.

    Args:
        v: String value to convert to boolean

    Returns:
        bool: Converted boolean value

    Raises:
        ValueError: If the string cannot be converted to a boolean
    """
    if not isinstance(v, str):
        raise ValueError(f"Expected string input, got {type(v)}")

    v_lower = v.lower().strip()

    if v_lower in ("true", "1", "yes", "on"):
        return True
    elif v_lower in ("false", "0", "no", "off", ""):
        return False
    else:
        raise ValueError(f"Cannot convert string '{v}' to boolean")
