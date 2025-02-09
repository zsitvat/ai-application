#!/bin/sh
cd ./src
uvicorn app:app --host 0.0.0.0 --log-level debug --port 5000 --reload
