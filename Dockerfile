FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md openenv.yaml __init__.py client.py models.py tasks.py graders.py ./
COPY server ./server

RUN pip install --upgrade pip \
    && pip install .

EXPOSE 8000

ENV ENABLE_WEB_INTERFACE=true
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
