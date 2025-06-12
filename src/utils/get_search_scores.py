async def create_score_string(doc, index):
    """Create a formatted string for a document and its score.
    Args:
        doc (tuple): A tuple containing the document and its score.
        index (int): The index of the document in the list.
    Returns:
        str: Formatted string with document index and score.
    """
    if not doc or len(doc) < 2:
        return "Invalid document data"

    return "Doc" + str(index) + " " + str(doc[1]) + "\n"


async def get_scores_as_string(docs):
    """Convert a list of documents with scores into a formatted string.
    Args:
        docs (list): List of tuples where each tuple contains a document and its score.
    Returns:
        str: Formatted string with document scores.
    """

    if not docs:
        return "No documents found."

    scores = ""
    for i, doc in enumerate(docs):
        scores += await create_score_string(doc, i + 1)
    return scores
