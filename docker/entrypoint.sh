#!/usr/bin/env sh
set -eu

APP_PORT="${PORT:-5000}"

exec gunicorn \
  --bind "0.0.0.0:${APP_PORT}" \
  --workers 1 \
  --threads 8 \
  --timeout 120 \
  paper_web:app
