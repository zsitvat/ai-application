import subprocess
import sys
import os

# Run tests first
print("Running tests...")
test_result = subprocess.run(["make", "test"])
if test_result.returncode != 0:
    print("Tests failed, exiting.")
    sys.exit(test_result.returncode)

print("Starting FastAPI and LangGraph Dev...")
fastapi_cmd = [
    "uvicorn",
    "src.app:app",
    "--host",
    "0.0.0.0",
    "--log-level",
    "debug",
    "--port",
    "5000",
    "--reload",
]
langgraph_cmd = [os.path.join(".venv", "bin", "langgraph"), "dev"]

fastapi_proc = subprocess.Popen(fastapi_cmd)
langgraph_proc = subprocess.Popen(langgraph_cmd)

try:
    fastapi_proc.wait()
    langgraph_proc.wait()
except KeyboardInterrupt:
    fastapi_proc.terminate()
    langgraph_proc.terminate()
    fastapi_proc.wait()
    langgraph_proc.wait()
