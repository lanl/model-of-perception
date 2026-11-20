#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Render sets --preload by default. Background threads must start in the worker.
unset GUNICORN_CMD_ARGS
exec uv run --locked --no-dev gunicorn --config gunicorn.conf.py 'task_app.server:create_app()'
