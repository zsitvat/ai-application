help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies using Poetry
	poetry install

clean:  ## Clean cache and temporary files
	poetry cache clear --all pypi -n
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

test:  ## Run tests with pytest
	poetry run pytest tests/ -v --cov=src --cov-report=html --cov-report=term

run:  ## Run the application
	poetry run uvicorn src.app:app --host 0.0.0.0 --port 8000

dev:  ## Run the application in development mode with auto-reload
	poetry run uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

build:  ## Build the package
	poetry build

lint:  ## Run linting tools
	poetry run flake8 src/ tests/
	poetry run mypy src/

format:  ## Format code using black and isort
	poetry run black src/ tests/
	poetry run isort src/ tests/

setup:  ## Initial setup - install dependencies and pre-commit hooks
	poetry install
	poetry run pre-commit install

shell:  ## Activate the virtual environment (use 'source' or 'eval')
	@echo "To activate the virtual environment, run one of these commands:"
	@echo "  source $$(poetry env info --path)/bin/activate"
	@echo "  eval $$(poetry env activate)"

update:  ## Update dependencies
	poetry update

lock:  ## Update the poetry.lock file
	poetry lock

docker-build:  ## Build Docker image
	docker build -t ai-app .

docker-run:  ## Run Docker container
	docker run -p 8000:8000 ai-app