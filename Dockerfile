# syntax=docker/dockerfile:1

FROM python:3.11-slim-buster

WORKDIR /app

# Install system dependencies
RUN apt-get update -qq && \
  apt-get install -y --no-install-recommends \
  build-essential \
  curl \
  git \
  libffi-dev \
  libssl-dev && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --upgrade pip && \
  pip install poetry

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
  POETRY_VENV_IN_PROJECT=1 \
  POETRY_CACHE_DIR=/tmp/poetry_cache \
  POETRY_HOME="/opt/poetry"

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --only=main --no-root && rm -rf "$POETRY_CACHE_DIR"

# Copy application code
COPY . .

EXPOSE 5000

WORKDIR /app/src

CMD ["poetry", "run", "python", "app.py"]