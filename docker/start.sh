#!/bin/sh
set -e

echo "Running database migrations..."
alembic upgrade head

echo "Starting Face Authentication API..."
# proxy-headers: honor X-Forwarded-Proto from Nginx (HTTPS /docs).
# forwarded-allow-ips=*: API is not published to the host, only Nginx can reach it.
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*'
