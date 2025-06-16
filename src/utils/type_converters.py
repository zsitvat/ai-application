def str2bool(v) -> bool:
    """Convert string to boolean

    Args:
        v (str): String value to convert
    Returns:
        bool: Converted boolean value
    Raises:
        ValueError: If the string cannot be converted to a boolean
    """

    if v == "True" or v == "true" or v == "TRUE" or v is True:
        return True
    elif v == "False" or v == "false" or v == "FALSE" or v is False or v is None:
        return False
    else:
        raise ValueError(
            "Boolean value expected on converting with str2bool(). Value {}".format(
                str(v)
            )
        )
