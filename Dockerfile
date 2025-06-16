# syntax=docker/dockerfile:1

FROM python:3.11-slim-buster

WORKDIR /app

RUN apt-get update -qq && \
  apt-get install -y --no-install-recommends build-essential curl && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --upgrade pip && \
  pip install poetry

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
  POETRY_VENV_IN_PROJECT=1 \
  POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-dev && rm -rf "$POETRY_CACHE_DIR"

# Copy application code
COPY . .

EXPOSE 5000

WORKDIR /app/src

CMD ["poetry", "run", "python", "app.py"]