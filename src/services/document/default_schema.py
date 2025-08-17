DEFAULT_INDEX_SCHEMA = {
    "text": [
        {"name": "text"},
        {"name": "source"},
    ],
    "vector": [
        {
            "name": "embedding",
            "algorithm": "FLAT",
            "block_size": 1000,
            "datatype": "FLOAT32",
            "dims": 1536,
            "distance_metric": "COSINE",
            "initial_cap": 20000,
        }
    ],
}
