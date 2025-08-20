
#!/bin/sh

# Usage:
#   ./run.sh                # Only FastAPI
#   ./run.sh pytest         # FastAPI + pytest in parallel
#   ./run.sh fastapi-langgraph # FastAPI + LangGraph Dev in parallel

if [ "$1" = "pytest" ]; then
	make test
	TEST_EXIT_CODE=$?
	if [ $TEST_EXIT_CODE -eq 0 ]; then
		uv run uvicorn src.app:app --host 0.0.0.0 --log-level debug --port 5000 --reload
	else
		echo "Tests failed, FastAPI server will not start."
		exit $TEST_EXIT_CODE
	fi
elif [ "$1" = "fastapi-langgraph" ]; then
    make test
	uv run uvicorn src.app:app --host 0.0.0.0 --log-level debug --port 5000 --reload &
	FASTAPI_PID=$!
	.venv/bin/langgraph dev &
	LANGGRAPH_PID=$!
	wait $FASTAPI_PID
	wait $LANGGRAPH_PID
else
	uv run uvicorn src.app:app --host 0.0.0.0 --log-level debug --port 5000 --reload
fi
