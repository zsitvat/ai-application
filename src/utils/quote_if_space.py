def quote_if_space(val: str) -> str:
    """Put quotes around the value if it contains spaces."""
    return f'"{val}"' if " " in val else val
