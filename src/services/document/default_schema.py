DEFAULT_INDEX_SCHEMA = [
    {"name": "content", "type": "text", "attrs": {}},
    {"name": "source", "type": "text", "attrs": {}},
    {"name": "document_index", "type": "numeric", "attrs": {}},
    {
        "name": "vector",
        "type": "vector",
        "attrs": {
            "dims": 1536,
            "algorithm": "FLAT",
            "datatype": "FLOAT32",
            "distance_metric": "COSINE",
            "initial_cap": 20000,
            "block_size": 1000,
        },
    },
]
