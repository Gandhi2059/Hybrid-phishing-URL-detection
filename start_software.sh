#!/usr/bin/env bash
# start_software.sh — Launch PhishDetect full-stack application
# ---------------------------------------------------------------
# Usage:
#   ./start_software.sh           # starts backend + frontend
#   ./start_software.sh --api-only
#   ./start_software.sh --ui-only (streamlit)
# ---------------------------------------------------------------

set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PORT=8000
FRONTEND_PORT=5173

echo -e "${BLUE}"
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║   🛡️  PhishDetect — Full-Stack Launcher      ║"
echo "  ╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# Activate venv if present
if [ -d "${ROOT_DIR}/venv" ]; then
    source "${ROOT_DIR}/venv/bin/activate"
    echo -e "${GREEN}✔ Virtual environment activated${NC}"
fi

# Free ports
fuser -k ${API_PORT}/tcp 2>/dev/null && echo -e "${YELLOW}  Freed port ${API_PORT}${NC}" || true
fuser -k ${FRONTEND_PORT}/tcp 2>/dev/null && echo -e "${YELLOW}  Freed port ${FRONTEND_PORT}${NC}" || true

MODE="${1:---full}"

# ── FastAPI backend ──────────────────────────────────────────────────────── #
if [[ "$MODE" != "--ui-only" ]]; then
    echo -e "\n${GREEN}→ Starting FastAPI backend on port ${API_PORT}…${NC}"
    uvicorn backend.api:app \
        --host 127.0.0.1 \
        --port ${API_PORT} \
        --app-dir "${ROOT_DIR}" &
    BACKEND_PID=$!
    sleep 2
    echo -e "${GREEN}  ✔ Backend PID: ${BACKEND_PID}${NC}"
fi

# ── React / Vite frontend ────────────────────────────────────────────────── #
if [[ "$MODE" != "--api-only" ]]; then
    echo -e "\n${GREEN}→ Starting React frontend on port ${FRONTEND_PORT}…${NC}"
    cd "${ROOT_DIR}/frontend"
    npm run dev -- --host 127.0.0.1 --port ${FRONTEND_PORT} &
    FRONTEND_PID=$!
    cd "${ROOT_DIR}"
    echo -e "${GREEN}  ✔ Frontend PID: ${FRONTEND_PID}${NC}"
fi

echo -e "\n${BLUE}══════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅  PhishDetect is running!${NC}"
echo -e "${BLUE}══════════════════════════════════════════════${NC}"
[[ "$MODE" != "--ui-only" ]] && \
    echo -e "  Backend  → http://127.0.0.1:${API_PORT}/docs"
[[ "$MODE" != "--api-only" ]] && \
    echo -e "  Frontend → http://127.0.0.1:${FRONTEND_PORT}"
echo -e "\n  Press ${YELLOW}[Ctrl+C]${NC} to stop all servers.\n"

# Graceful shutdown
cleanup() {
    echo -e "\n${YELLOW}Stopping servers…${NC}"
    [[ -n "${BACKEND_PID:-}"  ]] && kill "${BACKEND_PID}"  2>/dev/null || true
    [[ -n "${FRONTEND_PID:-}" ]] && kill "${FRONTEND_PID}" 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

wait
