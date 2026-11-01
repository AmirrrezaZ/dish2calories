#!/usr/bin/env bash
# Run the FastAPI backend and the Streamlit UI in a single container.
# Used by hosts that run exactly one container (Hugging Face Spaces, Cloud Run, Render, ...).
#
#   PORT      port the public UI listens on (default 8501; Cloud Run/Render set this)
#   API_PORT  internal API port (default 8000)
set -euo pipefail

export API_HOST="${API_HOST:-127.0.0.1}"
export API_PORT="${API_PORT:-8000}"
export API_URL="${API_URL:-http://127.0.0.1:${API_PORT}}"

python inference.py &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT

exec streamlit run app.py \
  --server.port="${PORT:-8501}" \
  --server.address=0.0.0.0 \
  --server.headless=true
