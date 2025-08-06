# syntax=docker/dockerfile:1

FROM python:3.11-slim

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

# Install uv
RUN pip install --upgrade pip && \
  pip install uv

# Copy project files
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv sync --no-dev --frozen

# Copy application code
COPY . .

EXPOSE 5000

CMD ["uv", "run", "python", "src/app.py"]