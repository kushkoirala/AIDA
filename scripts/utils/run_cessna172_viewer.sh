#!/bin/bash
# Cessna 172 Telemetry Viewer Launcher
# Starts HTTP server and telemetry simulation for 3D viewer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# PID files
HTTP_PID_FILE="/tmp/cessna172_http_server.pid"
TELEMETRY_PID_FILE="/tmp/cessna172_telemetry.pid"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down servers...${NC}"

    if [ -f "$HTTP_PID_FILE" ]; then
        HTTP_PID=$(cat "$HTTP_PID_FILE")
        if kill -0 "$HTTP_PID" 2>/dev/null; then
            echo "Stopping HTTP server (PID: $HTTP_PID)"
            kill "$HTTP_PID" 2>/dev/null || true
        fi
        rm -f "$HTTP_PID_FILE"
    fi

    if [ -f "$TELEMETRY_PID_FILE" ]; then
        TELEMETRY_PID=$(cat "$TELEMETRY_PID_FILE")
        if kill -0 "$TELEMETRY_PID" 2>/dev/null; then
            echo "Stopping telemetry server (PID: $TELEMETRY_PID)"
            kill "$TELEMETRY_PID" 2>/dev/null || true
        fi
        rm -f "$TELEMETRY_PID_FILE"
    fi

    echo -e "${GREEN}Servers stopped${NC}"
}

trap cleanup EXIT INT TERM

# Check if virtual environment exists
if [ ! -d ".venv-linux" ]; then
    echo -e "${RED}Error: Virtual environment not found at .venv-linux${NC}"
    exit 1
fi

# Activate virtual environment
source .venv-linux/bin/activate

# Set environment variables
export CUPY_CACHE_DIR=/tmp/cupy_cache
export MPLCONFIGDIR=/tmp/matplotlib

# Check if model exists
MODEL_PATH="checkpoints/cessna172_curriculum/phase1_ground_roll/best_model.zip"
if [ ! -f "$MODEL_PATH" ]; then
    MODEL_PATH="checkpoints/cessna172_curriculum/phase1_ground_roll/phase1_ground_roll_ppo_final.zip"
    if [ ! -f "$MODEL_PATH" ]; then
        echo -e "${RED}Error: No trained model found!${NC}"
        echo "Expected: checkpoints/cessna172_curriculum/phase1_ground_roll/best_model.zip"
        echo "Or: checkpoints/cessna172_curriculum/phase1_ground_roll/phase1_ground_roll_ppo_final.zip"
        exit 1
    fi
fi

echo -e "${GREEN}===========================================================${NC}"
echo -e "${GREEN}  CESSNA 172 TELEMETRY VIEWER${NC}"
echo -e "${GREEN}===========================================================${NC}"
echo ""
echo -e "Model: ${YELLOW}$MODEL_PATH${NC}"
echo -e "Task: ${YELLOW}ground_roll${NC}"
echo ""
echo -e "${GREEN}===========================================================${NC}"
echo ""

# Start HTTP server in background
echo -e "${YELLOW}Starting HTTP server on port 8000...${NC}"
cd viewer/public
python3 -m http.server 8000 > /tmp/cessna172_http_server.log 2>&1 &
HTTP_PID=$!
echo $HTTP_PID > "$HTTP_PID_FILE"
cd "$SCRIPT_DIR"
sleep 1

if kill -0 "$HTTP_PID" 2>/dev/null; then
    echo -e "${GREEN}✓ HTTP server started (PID: $HTTP_PID)${NC}"
else
    echo -e "${RED}✗ HTTP server failed to start${NC}"
    cat /tmp/cessna172_http_server.log
    exit 1
fi

# Start telemetry server
echo -e "${YELLOW}Starting telemetry server on port 8765...${NC}"
python scripts/run_cessna172_with_telemetry.py \
    --model "$MODEL_PATH" \
    --task ground_roll \
    --dt 0.05 \
    > /tmp/cessna172_telemetry.log 2>&1 &
TELEMETRY_PID=$!
echo $TELEMETRY_PID > "$TELEMETRY_PID_FILE"
sleep 2

if kill -0 "$TELEMETRY_PID" 2>/dev/null; then
    echo -e "${GREEN}✓ Telemetry server started (PID: $TELEMETRY_PID)${NC}"
else
    echo -e "${RED}✗ Telemetry server failed to start${NC}"
    cat /tmp/cessna172_telemetry.log
    exit 1
fi

echo ""
echo -e "${GREEN}===========================================================${NC}"
echo -e "${GREEN}  SERVERS RUNNING${NC}"
echo -e "${GREEN}===========================================================${NC}"
echo ""
echo -e "  Viewer URL:  ${YELLOW}http://localhost:8000${NC}"
echo -e "  WebSocket:   ${YELLOW}ws://localhost:8765${NC}"
echo ""
echo -e "  HTTP Log:    /tmp/cessna172_http_server.log"
echo -e "  Telemetry:   /tmp/cessna172_telemetry.log"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
echo -e "${GREEN}===========================================================${NC}"
echo ""

# Wait for telemetry server (HTTP server will be cleaned up on exit)
wait $TELEMETRY_PID
