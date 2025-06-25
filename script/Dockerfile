# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.7.1
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set workdir
WORKDIR /app

# Copy pyproject.toml and poetry.lock
COPY ../server/pyproject.toml ./server/pyproject.toml
COPY ../server/poetry.lock ./server/poetry.lock

# Install dependencies
RUN cd server && poetry install --no-root --only main

# Copy app code
COPY ../server ./server

# Expose port
EXPOSE 8000

# Entrypoint
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 