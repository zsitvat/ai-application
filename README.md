vagy pip-pel:
# Recruiter AI App

This application is a multi-agent artificial intelligence system capable of processing, searching, and filtering documents, web data, and other sources. The system supports vector database, personal data masking, web search, and graph-based workflows.

## Getting Started

1. **Install uv (if not already installed):**
   ```bash
   # macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

   # With pip
   pip install uv
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

   Or to install only production dependencies:
   ```bash
   uv sync --no-dev
   ```

3. **Start Redis Stack with Docker:**
   ```bash
   docker run -d --name redis-stack-server -p 6380:6379 -p 8001:8001 redis/redis-stack:latest
   ```
   - Redis server available at: `localhost:6380`
   - RedisInsight UI: [http://localhost:8001](http://localhost:8001)

4. **Set environment variables:**
   Copy the contents of `sample.env` to `.env` and fill in the required keys.


5. **Start the application:**
   ```bash
   uv run python src/app.py
   ```
   or
   ```bash
   ./run.sh
   ```

   Or you can start the FastAPI app directly with uvicorn (from the `src` directory):
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 5000
   ```


## Main API Endpoints (Summary)

- `/api/document/...` – Document management
- `/api/personal_data_filter/...` – Personal data masking
- `/api/topic_validation/...` – Topic validation
- `/api/web_scraping/...` – Web data collection
- `/api/graph` – Multi-agent AI response generation (document, web, filtering)
- `/api/vector_db/create` – Create vector database from documents
- `/api/health-check` – Health check

For more details and examples, see the `/docs` endpoint (FastAPI automatic documentation).

---

Before starting, make sure all required environment variables are set and dependencies are installed (`uv sync`).


## Using LangGraph Studio (for local development)

**What is LangGraph Studio?**

LangGraph Studio is a developer tool for visually testing, developing, and debugging LangGraph-based workflows. It helps you easily understand data flow between agents, try out graph configurations, and quickly get feedback on system behavior. It's especially useful for developing or debugging complex multi-agent processes.

### 1. Install LangGraph CLI (if not already installed):

```bash
uv add --dev "langgraph-cli[inmem]"
```

or with pip:

```bash
pip install -U "langgraph-cli[inmem]"
```

### 2. Start the LangGraph development server:

```bash
uv run langgraph dev
```

If you use Safari browser or have issues accessing localhost, start the server as follows:

```bash
uv run langgraph dev --tunnel
```

### 3. Open LangGraph Studio in your browser:

Visit:

https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

If the server runs on a different port or host, adjust the `baseUrl` parameter accordingly.

More info: [LangGraph Studio Quickstart](https://langchain-ai.github.io/langgraph/cloud/how-tos/studio/quick_start/#local-development-server)

#### (Optional) Attach Debugger

If you want to debug the application step-by-step:

```bash
uv add --dev debugpy
uv run langgraph dev --debug-port 5678
```

Then connect to port 5678 in VS Code or another IDE.

## uv vs Poetry Comparison

The project switched from Poetry to uv for the following advantages:

- **Faster installation**: uv is written in Rust, much faster dependency resolution and installation
- **Simpler usage**: fewer commands, simpler workflow
- **Modern Python tooling**: up-to-date Python project management
- **Better performance**: virtual environment handling and package caching
- **Compatibility**: full PEP 517/518 support

> **Migration guide**: If you previously used Poetry, see the `MIGRATION_TO_UV.md` file for detailed migration steps.

### Common uv commands:

```bash
# Project initialization (creates new pyproject.toml)
uv init

# Install dependencies (downloads and installs all required packages)
uv sync

# Add dependency (adds a new package to project dependencies)
uv add package-name

# Add dev dependency (adds a development package)
uv add --dev package-name

# Run command in virtual environment (e.g. tests, scripts)
uv run python script.py

# Run Python script (start main application)
uv run python src/app.py

# Activate shell in virtual environment (interactive shell with all uv-installed packages available)
uv shell
```
