#!/usr/bin/env python3
import subprocess
import sys
import os
import threading
import time
import socket
from pathlib import Path


def check_port_available(port):
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return True
        except OSError:
            return False


def run_fastapi():
    """Run FastAPI server with uvicorn"""
    print("Starting FastAPI server on port 5000...")
    cmd = [
        "uv",
        "run",
        "uvicorn",
        "src.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        "5000",
        "--reload",
        "--log-level",
        "info",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent / "src")

    try:
        subprocess.run(cmd, cwd=Path(__file__).parent, env=env)
    except KeyboardInterrupt:
        print("FastAPI server stopped")


def run_langgraph():
    """Run LangGraph dev server"""
    print("Starting LangGraph Dev server on port 2024...")
    # Small delay to let FastAPI start first
    time.sleep(8)

    cmd = ["uv", "run", "langgraph", "dev"]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent / "src")

    try:
        subprocess.run(cmd, cwd=Path(__file__).parent, env=env)
    except KeyboardInterrupt:
        print("LangGraph Dev server stopped")


def run_agent_chat_ui():
    """Run Agent Chat UI (Next.js)"""
    try:
        print("Starting Agent Chat UI on port 3000...")
        # Small delay to let other servers start first
        time.sleep(15)

        # Check if the directory exists
        agent_ui_path = Path(__file__).parent / "agent-chat-ui"
        print(f"Checking Agent Chat UI path: {agent_ui_path}")
        if not agent_ui_path.exists():
            print(f"Error: Agent Chat UI directory not found at {agent_ui_path}")
            return

        # Install dependencies first
        print("Installing Agent Chat UI dependencies...")
        install_cmd = ["pnpm", "install"]
        try:
            result = subprocess.run(
                install_cmd, cwd=agent_ui_path, capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"Failed to install dependencies: {result.stderr}")
                return
            else:
                print("Dependencies installed successfully")
        except Exception as e:
            print(f"Error installing dependencies: {e}")
            return

        # Check if port 3000 is available
        if not check_port_available(3000):
            print("Error: Port 3000 is already in use!")
            return

        print("Starting Agent Chat UI server...")
        cmd = ["pnpm", "dev", "--port", "3000"]

        # Set environment variable to ensure port 3000
        env = os.environ.copy()
        env["PORT"] = "3000"

        try:
            print(f"Running command: {' '.join(cmd)} in {agent_ui_path}")
            process = subprocess.Popen(
                cmd,
                cwd=agent_ui_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Print output in real-time
            for line in process.stdout:
                print(f"[Agent Chat UI] {line.strip()}")

            process.wait()
        except KeyboardInterrupt:
            print("Agent Chat UI stopped")
            if process:
                process.terminate()
        except Exception as e:
            print(f"Error starting Agent Chat UI: {e}")

    except Exception as e:
        print(f"Fatal error in Agent Chat UI thread: {e}")
        import traceback

        traceback.print_exc()
        # Keep the thread alive even if there's an error
        while True:
            time.sleep(1)


def run_tests():
    """Run tests before starting servers"""
    print("Running tests...")
    cmd = ["uv", "run", "pytest", "tests/"]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent / "src")

    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, env=env)
        if result.returncode != 0:
            print("Tests failed! Stopping...")
            sys.exit(result.returncode)
        else:
            print("All tests passed!")
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)


def main():
    """Main function to run tests first, then start all three servers in parallel"""
    # Run tests first
    run_tests()

    print("Starting development servers...")
    print("FastAPI server will be available at: http://localhost:5000")
    print("API Docs will be available at: http://localhost:5000/docs")
    print("LangGraph server will be available at: http://localhost:2024")
    print(
        "Studio UI will be available at: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"
    )
    print("Agent Chat UI will be available at: http://localhost:3000")
    print("=" * 80)

    # Start all three servers in separate threads
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=False)
    langgraph_thread = threading.Thread(target=run_langgraph, daemon=False)
    agent_ui_thread = threading.Thread(target=run_agent_chat_ui, daemon=False)

    fastapi_thread.start()
    langgraph_thread.start()
    agent_ui_thread.start()

    try:
        # Keep main thread alive
        while (
            fastapi_thread.is_alive()
            or langgraph_thread.is_alive()
            or agent_ui_thread.is_alive()
        ):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        sys.exit(0)


if __name__ == "__main__":
    main()
