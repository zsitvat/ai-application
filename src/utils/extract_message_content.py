def extract_message_content(msg):
    """Safely extract string content from a message object or dict, flattening lists if needed."""
    if isinstance(msg, dict) and "content" in msg:
        content = msg["content"]
    elif hasattr(msg, "content"):
        content = msg.content
    else:
        return str(msg)
    if isinstance(content, list):
        return " ".join(str(x) for x in content)
    return str(content)
