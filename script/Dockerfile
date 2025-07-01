FROM registry.dev.facere.ai/python-ml-base:latest

WORKDIR /app
#RUN apt-get install python3.10-venv -y
# Copy relevant files
COPY pyproject.toml poetry.lock ./
# Install requirements, we share mount the poetry cache directory to speed up installation.
RUN --mount=type=cache,target=/root/.cache/pypoetry,sharing=locked \
    poetry install --no-interaction --no-ansi --no-root

#Install tesseract
#RUN apt-get update && apt-get install -y tesseract-ocr
#RUN rm -rf /var/lib/apt/lists/*

COPY ./server /app/server
#COPY ./tests /app/tests
#COPY ./training /app/training

ENV PYTHONPATH=/app
ENV PYTHONBUFFERED=1

EXPOSE 8000

## TODO: uvicorn vs gunicorn or other?
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws-max-size", "1048576000"]