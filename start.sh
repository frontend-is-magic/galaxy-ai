#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PID=""
FRONTEND_PID=""

info() {
  printf '[galaxy-ai] %s\n' "$1"
}

fail() {
  printf '[galaxy-ai] ERROR: %s\n' "$1" >&2
  exit 1
}

load_env() {
  local env_file="$ROOT_DIR/.env"

  if [ ! -f "$env_file" ]; then
    fail "Root .env is missing. Create it from .env.example, then run ./start.sh again."
  fi

  set -a
  # shellcheck disable=SC1090
  . "$env_file"
  set +a
}

require_command() {
  local command_name="$1"
  local install_hint="$2"

  if ! command -v "$command_name" >/dev/null 2>&1; then
    fail "$command_name is required but was not found. Suggested install: $install_hint"
  fi
}

require_path() {
  local path="$1"
  local message="$2"

  if [ ! -e "$path" ]; then
    fail "$message"
  fi
}

cleanup() {
  local status=$?

  trap - EXIT INT TERM

  if [ -n "$FRONTEND_PID" ]; then
    stop_process_tree "$FRONTEND_PID"
  fi

  if [ -n "$BACKEND_PID" ]; then
    stop_process_tree "$BACKEND_PID"
  fi

  if [ -n "$FRONTEND_PID" ]; then
    wait "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi

  if [ -n "$BACKEND_PID" ]; then
    wait "$BACKEND_PID" >/dev/null 2>&1 || true
  fi

  exit "$status"
}

stop_process_tree() {
  local target_pid="$1"
  local child_pid

  if command -v pgrep >/dev/null 2>&1; then
    for child_pid in $(pgrep -P "$target_pid" 2>/dev/null || true); do
      stop_process_tree "$child_pid"
    done
  fi

  kill "$target_pid" >/dev/null 2>&1 || true
}

is_job_running() {
  local target_pid="$1"
  local running_pid

  for running_pid in $(jobs -pr); do
    if [ "$running_pid" = "$target_pid" ]; then
      return 0
    fi
  done

  return 1
}

wait_for_services() {
  local exited_service
  local exited_pid
  local exited_status

  while true; do
    if ! is_job_running "$BACKEND_PID"; then
      exited_service="Backend"
      exited_pid="$BACKEND_PID"
      break
    fi

    if ! is_job_running "$FRONTEND_PID"; then
      exited_service="Frontend"
      exited_pid="$FRONTEND_PID"
      break
    fi

    sleep 1
  done

  set +e
  wait "$exited_pid"
  exited_status=$?
  set -e

  info "$exited_service stopped with exit code $exited_status."
  return "$exited_status"
}

resolve_uv() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return
  fi

  if [ -x "$HOME/.local/bin/uv" ]; then
    printf '%s\n' "$HOME/.local/bin/uv"
    return
  fi

  fail "uv is required but was not found. Suggested install: curl -LsSf https://astral.sh/uv/install.sh | sh"
}

load_env
BACKEND_HOST="${GALAXY_AI_HOST:-127.0.0.1}"
BACKEND_PORT="${GALAXY_AI_PORT:-8000}"
FRONTEND_HOST="127.0.0.1"
FRONTEND_PORT="5173"

info "Running startup checks."
require_command "python3" "Install Python 3.11+ from https://www.python.org/downloads/"
require_command "node" "Install Node.js LTS from https://nodejs.org/"
require_command "npm" "Install npm with Node.js LTS from https://nodejs.org/"
UV_BIN="$(resolve_uv)"

require_path "$ROOT_DIR/frontend" "frontend/ does not exist."
require_path "$ROOT_DIR/frontend/node_modules" "frontend dependencies are missing. Run ./init.sh first."
require_path "$ROOT_DIR/backend" "backend/ does not exist."
require_path "$ROOT_DIR/backend/.venv" "backend virtual environment is missing. Run ./init.sh first."

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

info "Starting backend on http://${BACKEND_HOST}:${BACKEND_PORT}"
(cd "$ROOT_DIR/backend" && "$UV_BIN" run --no-sync uvicorn app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT") &
BACKEND_PID=$!

info "Starting frontend on http://${FRONTEND_HOST}:${FRONTEND_PORT}"
(cd "$ROOT_DIR/frontend" && npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT") &
FRONTEND_PID=$!

info "Galaxy AI is running. Press Ctrl+C to stop both services."
wait_for_services
