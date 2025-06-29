DEFAULT_INDEX_SCHEMA = [
    {"name": "content", "type": "text"},
    {"name": "source", "type": "text"},
    {"name": "document_index", "type": "numeric"},
    {
        "name": "vector",
        "type": "vector",
        "attrs": {
            "dims": 1536,
            "distance_metric": "cosine",
            "algorithm": "flat",
            "datatype": "float32",
        },
    },
]
