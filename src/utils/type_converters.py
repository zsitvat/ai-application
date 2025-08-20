def str2bool(v) -> bool:
    """Convert string to boolean

    Args:
        v (str): String value to convert
    Returns:
        bool: Converted boolean value
    Raises:
        ValueError: If the string cannot be converted to a boolean
    """

    if v in ("True", "true", "TRUE", True, "1", 1):
        return True
    elif v in ("False", "false", "FALSE", False, None, "0", 0):
        return False
    else:
        raise ValueError(
            "Boolean value expected on converting with str2bool(). Value {}".format(
                str(v)
            )
        )
