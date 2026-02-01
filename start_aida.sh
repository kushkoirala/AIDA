#!/bin/bash
# =============================================================================
# AIDA Flight Simulator - Complete Startup Script (Version A)
# =============================================================================
# Starts the complete flight simulation system with LLM:
#   1. HTTP server for the 3D viewer (port 8000)
#   2. Flight dynamics server with WebSocket telemetry (port 8765)
#   3. LLM command server for natural language commands (port 8766)
#
# Usage:
#   ./start_aida.sh              # Normal start (2x simulation speed)
#   ./start_aida.sh --speed 4    # 4x simulation speed
#   ./start_aida.sh --no-llm     # Start without LLM server
#
# Then open http://localhost:8000 in your browser
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
SIM_SPEED="2.0"
START_LLM=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--speed)
            SIM_SPEED="$2"
            shift 2
            ;;
        --no-llm)
            START_LLM=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--speed MULTIPLIER] [--no-llm]"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "  AIDA Flight Simulator - Version A"
echo "=============================================="
echo ""

# Kill any existing servers
echo "[1/5] Stopping any existing servers..."
pkill -f "python.*run_dynamic_xc" 2>/dev/null || true
pkill -f "python.*http.server.*8000" 2>/dev/null || true
pkill -f "python.*llm_command_server" 2>/dev/null || true
sleep 1

# Activate virtual environment
echo "[2/5] Activating Python environment..."
if [ -f ".venv-linux/bin/activate" ]; then
    source .venv-linux/bin/activate
else
    echo "ERROR: Virtual environment not found at .venv-linux/"
    echo "Please create it first:"
    echo "  python3 -m venv .venv-linux"
    echo "  source .venv-linux/bin/activate"
    echo "  pip install -r requirements.txt"
    echo "  pip install llama-cpp-python"
    exit 1
fi

# Start HTTP server for viewer
echo "[3/5] Starting HTTP viewer server on port 8000..."
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
echo "[4/5] Starting flight dynamics server (${SIM_SPEED}x speed)..."
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

# Start LLM command server (optional)
if [ "$START_LLM" = true ]; then
    echo "[5/5] Starting LLM command server on port 8766..."

    # Check if model exists
    LLM_MODEL="llm/models/Llama-xLAM-2-8B-fc-r-Q4_K_M.gguf"
    if [ ! -f "$LLM_MODEL" ]; then
        echo "       WARNING: LLM model not found at $LLM_MODEL"
        echo "       Download it with:"
        echo "         mkdir -p llm/models && cd llm/models"
        echo "         wget https://huggingface.co/bartowski/Llama-xLAM-2-8B-fc-r-GGUF/resolve/main/Llama-xLAM-2-8B-fc-r-Q4_K_M.gguf"
        echo "       Starting without LLM (rule-based parsing only)..."
    fi

    nohup python llm/llm_command_server.py > /tmp/aida_llm_server.log 2>&1 &
    LLM_PID=$!
    echo "       LLM server PID: $LLM_PID"

    # Wait for LLM server to initialize
    sleep 2
    if ! kill -0 $LLM_PID 2>/dev/null; then
        echo "       WARNING: LLM server failed to start. Check /tmp/aida_llm_server.log"
        echo "       Continuing without LLM..."
    fi
else
    echo "[5/5] Skipping LLM server (--no-llm specified)"
fi

echo ""
echo "=============================================="
echo "  AIDA Flight Simulator Ready!"
echo "=============================================="
echo ""
echo "  Open in browser: http://localhost:8000"
echo ""
echo "  Servers running:"
echo "    - HTTP Viewer:     http://localhost:8000"
echo "    - Flight Dynamics: ws://localhost:8765"
if [ "$START_LLM" = true ]; then
echo "    - LLM Commands:    ws://localhost:8766"
fi
echo ""
echo "  Logs:"
echo "    - HTTP server:   /tmp/aida_http_server.log"
echo "    - Flight server: /tmp/aida_flight_server.log"
if [ "$START_LLM" = true ]; then
echo "    - LLM server:    /tmp/aida_llm_server.log"
fi
echo ""
echo "  To stop: ./stop_aida.sh"
echo "=============================================="
echo ""

# Optionally follow the flight server log
if [ -t 1 ]; then
    echo "Press Ctrl+C to stop following logs (servers will continue running)"
    echo ""
    tail -f /tmp/aida_flight_server.log
fi
