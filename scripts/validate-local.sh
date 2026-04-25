#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:${PORT}"
TASK_ID="${TASK_ID:-ds_prompt_injection_access}"
SKIP_DOCKER="false"
VENV_BIN=""

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  VENV_BIN="${ROOT_DIR}/.venv/bin/"
fi

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
"${VENV_BIN}pytest" -q

echo "[2/6] openenv validate ."
"${VENV_BIN}openenv" validate .

echo "[3/6] start local server"
SERVER_LOG="${TMPDIR:-/tmp}/support_ops_env_server.log"
"${VENV_BIN}python" -m server.app --port "${PORT}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 15); do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
  echo "Local server failed to become healthy. Server log:" >&2
  cat "${SERVER_LOG}" >&2 || true
  exit 1
fi

echo "[4/6] runtime validation"
"${VENV_BIN}openenv" validate --url "${BASE_URL}"
curl -fsS -X POST "${BASE_URL}/reset" -H "content-type: application/json" \
  -d "{\"task_id\":\"${TASK_ID}\"}" >/dev/null

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
  -d "{\"task_id\":\"${TASK_ID}\"}" >/dev/null

echo "Validation complete."
