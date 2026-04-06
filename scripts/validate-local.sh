#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:${PORT}"
SKIP_DOCKER="false"

if [[ "${1:-}" == "--skip-docker" ]]; then
  SKIP_DOCKER="true"
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${DOCKER_CONTAINER_ID:-}" ]]; then
    docker rm -f "${DOCKER_CONTAINER_ID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

cd "${ROOT_DIR}"

echo "[1/6] pytest"
pytest -q

echo "[2/6] openenv validate ."
openenv validate .

echo "[3/6] start local server"
python -m server.app --port "${PORT}" >/tmp/support_ops_env_server.log 2>&1 &
SERVER_PID=$!
sleep 3

echo "[4/6] runtime validation"
openenv validate --url "${BASE_URL}"
curl -fsS "${BASE_URL}/health" >/dev/null
curl -fsS -X POST "${BASE_URL}/reset" -H "content-type: application/json" \
  -d '{"task_id":"easy_vip_sso"}' >/dev/null

if [[ "${SKIP_DOCKER}" == "true" ]]; then
  echo "[5/6] docker skipped"
  echo "Validation complete without Docker checks."
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required unless --skip-docker is used." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not available." >&2
  exit 1
fi

echo "[5/6] docker build"
docker build -t support-ops-env:latest .

echo "[6/6] docker smoke test"
DOCKER_CONTAINER_ID="$(docker run -d -p 18000:8000 support-ops-env:latest)"
sleep 5
curl -fsS "http://127.0.0.1:18000/health" >/dev/null
curl -fsS -X POST "http://127.0.0.1:18000/reset" -H "content-type: application/json" \
  -d '{"task_id":"easy_vip_sso"}' >/dev/null

echo "Validation complete."
