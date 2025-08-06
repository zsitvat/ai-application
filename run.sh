#!/bin/sh
uv run uvicorn src.app:app --host 0.0.0.0 --log-level debug --port 5000 --reload
