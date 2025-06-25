# Kishara LLM API Server

A FastAPI-based server for the Kishara Local LLM project.

## Setup

### Poetry (Recommended)

1. Install Poetry: https://python-poetry.org/docs/#installation
2. Install dependencies:

```bash
poetry install
```

3. Run the server:

```bash
poetry run python start.py
```

Or, for direct uvicorn:

```bash
poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Running with Docker

1. Build the Docker image:

```bash
docker build -f script/Dockerfile -t kishara-llm-api .
```

2. Run the container:

```bash
docker run -p 8000:8000 kishara-llm-api
```

## Running with Docker Compose

1. Start the service:

```bash
docker compose up --build
```

2. Stop the service:

```bash
docker compose down
```

## Environment Variables

You can configure the server using these environment variables:

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `RELOAD`: Enable auto-reload (default: true)
- `LOG_LEVEL`: Logging level (default: info)

Example:

```bash
HOST=127.0.0.1 PORT=8080 poetry run python start.py
```

## API Endpoints

- `GET /`: Root endpoint
- `GET /health`: Health check
- `GET /api/v1/status`: API status
- `GET /docs`: Interactive API documentation (Swagger UI)
- `GET /redoc`: Alternative API documentation (ReDoc)

## Development

The server includes:

- FastAPI framework
- CORS middleware
- Auto-reload for development
- Interactive API documentation
- Health check endpoints

## Accessing the API

Once running, you can access:

- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc
