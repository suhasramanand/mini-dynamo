#!/usr/bin/env bash
# Smoke test: verify the router is up, backends are registered, and a request
# streams tokens end-to-end through the disaggregated pipeline.
set -euo pipefail

ROUTER="${1:-http://localhost:8000}"

echo "== 1. router health =="
curl -fsS "$ROUTER/health" | python3 -m json.tool

echo
echo "== 2. registered backends =="
curl -fsS "$ROUTER/backends" | python3 -m json.tool

echo
echo "== 3. streamed generation (session s-smoke) =="
curl -fsS -N -X POST "$ROUTER/v1/generate" \
  -H 'content-type: application/json' \
  -d '{"session_id":"s-smoke","prompt":"the quick brown fox jumps","max_tokens":16}'

echo
echo "== 4. same session again (should be a cache hit / sticky) =="
curl -fsS -N -X POST "$ROUTER/v1/generate" \
  -H 'content-type: application/json' \
  -d '{"session_id":"s-smoke","prompt":"and again please","max_tokens":8}'

echo
echo "smoke test OK"
