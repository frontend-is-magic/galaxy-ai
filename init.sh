#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

info() {
  printf '[galaxy-ai] %s\n' "$1"
}

fail() {
  printf '[galaxy-ai] ERROR: %s\n' "$1" >&2
  exit 1
}

require_command() {
  local command_name="$1"
  local install_hint="$2"

  if ! command -v "$command_name" >/dev/null 2>&1; then
    fail "$command_name is required but was not found. Suggested install: $install_hint"
  fi
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

info "Running initialization checks."
require_command "python3" "Install Python 3.11+ from https://www.python.org/downloads/"
UV_BIN="$(resolve_uv)"

if [ -d "$ROOT_DIR/frontend" ]; then
  require_command "node" "Install Node.js LTS from https://nodejs.org/"
  require_command "npm" "Install npm with Node.js LTS from https://nodejs.org/"

  if [ -f "$ROOT_DIR/frontend/package-lock.json" ]; then
    info "Installing frontend dependencies with npm ci."
    (cd "$ROOT_DIR/frontend" && npm ci)
  elif [ -f "$ROOT_DIR/frontend/package.json" ]; then
    info "Installing frontend dependencies with npm install."
    (cd "$ROOT_DIR/frontend" && npm install)
  fi
else
  info "frontend/ does not exist yet; skipping Node.js and npm setup."
fi

if [ ! -d "$ROOT_DIR/backend" ]; then
  fail "backend/ does not exist."
fi

info "Synchronizing backend environment with uv."
(cd "$ROOT_DIR/backend" && "$UV_BIN" sync)

info "Initialization complete. Start Galaxy AI with ./start.sh."
