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

## File Upload Storage Configuration

The application supports multiple file storage backends with automatic selection based on your configuration. The system prioritizes cloud storage over local storage.

### Storage Backend Priority

1. AWS S3 (highest priority)
2. Google Drive (medium priority)  
3. Local Storage (fallback - always available)

### Configuration Options

#### AWS S3 Storage

For production environments with AWS infrastructure:

```bash
# Required environment variables in .env
AWS_S3_BUCKET=your-bucket-name
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
```

Setup:
1. Create an S3 bucket in AWS Console
2. Create IAM user with S3 permissions
3. Add credentials to `.env` file
4. Files will be automatically uploaded to S3

#### Google Drive Storage

For Google Workspace integration:

```bash
# Required environment variables in .env
GOOGLE_OAUTH_CREDENTIALS_FILE=credentials.json
GOOGLE_DRIVE_FOLDER_ID=
```

Setup:
1. Create Google Cloud Project and enable Drive API
2. Create OAuth 2.0 credentials and download `credentials.json`
3. Create a Google Drive folder and copy its ID from the URL
4. Run `uv run python generate_google_token.py` to authenticate
5. Files will be named as: `{thread_id}_{timestamp}_{filename}`

#### Local Storage (Default)

No configuration needed - works out of the box:

- Files stored in: `./files/uploads/`
- Automatic directory creation
- UUID-based filenames
- Suitable for development and testing

### Supported File Types

- Images: JPEG, PNG, GIF, WEBP
- Documents: PDF
- Max file size: 15MB (configurable)

### Testing File Upload:
```bash
# Test file upload via API
curl -X POST "http://localhost:5000/api/file_upload/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-file.pdf"
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

## Testing UI with LangChain Agent Chat UI

For testing and interacting with the multi-agent system, you can use the **LangChain Agent Chat UI** - a modern, responsive chat interface designed specifically for LangGraph applications.

### Features:
- Real-time chat interface for testing agents
- Support for streaming responses
- File upload capabilities
- Message history and conversation management
- Built-in support for LangGraph workflows

### Quick Setup:

1. **Clone the Agent Chat UI repository:**
   ```bash
   git clone https://github.com/langchain-ai/agent-chat-ui.git
   cd agent-chat-ui
   ```

2. **Install dependencies:**
   ```bash
   npm install
   # or
   pnpm install
   ```

3. **Configure the backend URL:**
   Update the configuration to point to your Recruiter AI App running on `http://localhost:5000`

4. **Start the UI:**
   ```bash
   npm run dev
   # or
   pnpm dev
   ```

5. **Access the UI:**
   Open [http://localhost:3000](http://localhost:3000) in your browser

This provides a user-friendly interface for testing your multi-agent workflows, trying different inputs, and observing how the personal data filter, topic validation, and other components work together.

**Repository:** https://github.com/langchain-ai/agent-chat-ui

## Examples and Research Materials

This repository contains comprehensive examples and research notebooks related to the thesis work:

### Examples Directory (`/examples/`)

- Configuration Examples: Sample graph configurations for different AI providers (OpenAI, Azure OpenAI)
- Document Examples: Sample position descriptions and company documents for testing
- API Examples: Postman collection with ready-to-use API requests

### Research Notebooks (`/notebooks/`)

The notebooks directory contains detailed research and experimentation work for the thesis:

- `graphs.ipynb` - Multi-agent graph workflow development and testing
- `fine_tunning.ipynb` - Model fine-tuning experiments and analysis
- `guardrails.ipynb` - AI safety and content filtering research
- `local_llms.ipynb` - Local language model integration and benchmarking
- `retriver_test.ipynb` - Vector database and retrieval system optimization
- `redis_filter.ipynb` - Redis-based filtering and caching mechanisms
- `prompt_copy_script.ipynb` - Prompt engineering and optimization techniques

#### Intent Recognition Research (`/notebooks/intent_recognition/`)

Specialized notebooks for natural language understanding:

- `BERT.ipynb` - BERT model implementation for intent classification
- `HuBERT.ipynb` - Hungarian BERT model adaptation
- `HuSpaCy.ipynb` - Hungarian language processing with spaCy
- `tensorflow_IR.ipynb` - TensorFlow-based intent recognition
- `whisper_openai.ipynb` - Speech-to-text integration with OpenAI Whisper
- `rasa.ipynb` - Conversational AI development with Rasa framework

### Documentation (`/docs/`)

Comprehensive API and service documentation:

- Service architecture overview
- API endpoint specifications
- Implementation guides for each component

These materials provide complete insight into the research process, experimental results, and technical implementation details of the multi-agent AI system developed for the thesis.
