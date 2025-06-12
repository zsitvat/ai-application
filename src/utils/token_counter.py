import tiktoken
import asyncio


async def token_counter(string: str, encoding_name: str) -> int:
    """Count the number of tokens in a string using the specified encoding.
    Args:
        string (str): The input string to count tokens in.
        encoding_name (str): The name of the encoding to use (e.g., "gpt-3.5-turbo").
    Returns:
        int: The number of tokens in the input string.
    Raises:
        ValueError: If the encoding name is not recognized.
    """
    encoding = await asyncio.to_thread(tiktoken.get_encoding, encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
