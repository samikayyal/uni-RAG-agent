#!/bin/sh
set -eu

# Force the large overlayfs copy during boot, before readiness can succeed.
cp -f /app/seed-data/uni_rag.sqlite /data/uni_rag.sqlite

exec uvicorn uni_rag_agent.app.api:create_app \
  --factory \
  --host 0.0.0.0 \
  --port "${PORT:-8080}" \
  --proxy-headers \
  --forwarded-allow-ips='*'
