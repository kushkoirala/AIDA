#!/bin/bash
# =============================================================================
# AIDA Flight Simulator - Startup Script
# =============================================================================
# Starts the complete flight simulation system:
#   1. HTTP server for the 3D viewer (port 8000)
#   2. Flight dynamics server with WebSocket telemetry (port 8765)
#
# Usage:
#   ./start_flight_sim.sh          # Normal start (2x simulation speed)
#   ./start_flight_sim.sh --speed 4  # 4x simulation speed
#
# Then open http://localhost:8000 in your browser
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
SIM_SPEED="2.0"
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--speed)
            SIM_SPEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--speed MULTIPLIER]"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "  AIDA Flight Simulator"
echo "=============================================="
echo ""

# Kill any existing servers
echo "[1/4] Stopping any existing servers..."
pkill -f "python.*run_dynamic_xc" 2>/dev/null || true
pkill -f "python.*http.server.*8000" 2>/dev/null || true
sleep 1

# Activate virtual environment
echo "[2/4] Activating Python environment..."
if [ -f ".venv-linux/bin/activate" ]; then
    source .venv-linux/bin/activate
else
    echo "ERROR: Virtual environment not found at .venv-linux/"
    echo "Please create it first: python3 -m venv .venv-linux && pip install -r requirements.txt"
    exit 1
fi

# Start HTTP server for viewer
echo "[3/4] Starting HTTP viewer server on port 8000..."
cd viewer/public
nohup python3 -m http.server 8000 > /tmp/aida_http_server.log 2>&1 &
HTTP_PID=$!
cd "$SCRIPT_DIR"
echo "       HTTP server PID: $HTTP_PID"

# Wait for HTTP server to start
sleep 1
if ! kill -0 $HTTP_PID 2>/dev/null; then
    echo "ERROR: HTTP server failed to start. Check /tmp/aida_http_server.log"
    exit 1
fi

# Start flight dynamics server
echo "[4/4] Starting flight dynamics server (${SIM_SPEED}x speed)..."
nohup python scripts/run_dynamic_xc.py --speed "$SIM_SPEED" > /tmp/aida_flight_server.log 2>&1 &
FLIGHT_PID=$!
echo "       Flight server PID: $FLIGHT_PID"

# Wait for flight server to initialize
sleep 3
if ! kill -0 $FLIGHT_PID 2>/dev/null; then
    echo "ERROR: Flight server failed to start. Check /tmp/aida_flight_server.log"
    kill $HTTP_PID 2>/dev/null
    exit 1
fi

echo ""
echo "=============================================="
echo "  Flight Simulator Ready!"
echo "=============================================="
echo ""
echo "  Open in browser: http://localhost:8000"
echo ""
echo "  Logs:"
echo "    HTTP server:   /tmp/aida_http_server.log"
echo "    Flight server: /tmp/aida_flight_server.log"
echo ""
echo "  To stop: ./stop_flight_sim.sh"
echo "           or: pkill -f 'python.*run_dynamic_xc'"
echo "=============================================="
echo ""

# Optionally follow the flight server log
if [ -t 1 ]; then
    echo "Press Ctrl+C to stop following logs (servers will continue running)"
    echo ""
    tail -f /tmp/aida_flight_server.log
fi
