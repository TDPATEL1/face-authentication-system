#!/bin/sh
set -e

# Self-signed certificate for local HTTPS only. Do not use in production.
# Do not commit server.key.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CERT_DIR="$SCRIPT_DIR/../nginx/certs"

mkdir -p "$CERT_DIR"

openssl req -x509 -nodes -newkey rsa:2048 \
  -keyout "$CERT_DIR/server.key" \
  -out "$CERT_DIR/server.crt" \
  -days 365 \
  -subj "/CN=localhost"

chmod 600 "$CERT_DIR/server.key"
echo "Wrote $CERT_DIR/server.crt and server.key"
