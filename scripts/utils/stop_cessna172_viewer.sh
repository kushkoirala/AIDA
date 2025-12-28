#!/bin/bash
# Stop Cessna 172 Telemetry Viewer

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# PID files
HTTP_PID_FILE="/tmp/cessna172_http_server.pid"
TELEMETRY_PID_FILE="/tmp/cessna172_telemetry.pid"

echo -e "${YELLOW}Stopping Cessna 172 Telemetry Viewer...${NC}"
echo ""

STOPPED=0

# Stop HTTP server
if [ -f "$HTTP_PID_FILE" ]; then
    HTTP_PID=$(cat "$HTTP_PID_FILE")
    if kill -0 "$HTTP_PID" 2>/dev/null; then
        echo -e "Stopping HTTP server (PID: ${YELLOW}$HTTP_PID${NC})"
        kill "$HTTP_PID" 2>/dev/null || true
        sleep 1
        if ! kill -0 "$HTTP_PID" 2>/dev/null; then
            echo -e "${GREEN}✓ HTTP server stopped${NC}"
            STOPPED=1
        else
            echo -e "${YELLOW}  Forcing stop...${NC}"
            kill -9 "$HTTP_PID" 2>/dev/null || true
            echo -e "${GREEN}✓ HTTP server stopped (forced)${NC}"
            STOPPED=1
        fi
    else
        echo -e "${YELLOW}HTTP server not running${NC}"
    fi
    rm -f "$HTTP_PID_FILE"
else
    echo -e "${YELLOW}HTTP server PID file not found${NC}"
fi

# Stop telemetry server
if [ -f "$TELEMETRY_PID_FILE" ]; then
    TELEMETRY_PID=$(cat "$TELEMETRY_PID_FILE")
    if kill -0 "$TELEMETRY_PID" 2>/dev/null; then
        echo -e "Stopping telemetry server (PID: ${YELLOW}$TELEMETRY_PID${NC})"
        kill "$TELEMETRY_PID" 2>/dev/null || true
        sleep 1
        if ! kill -0 "$TELEMETRY_PID" 2>/dev/null; then
            echo -e "${GREEN}✓ Telemetry server stopped${NC}"
            STOPPED=1
        else
            echo -e "${YELLOW}  Forcing stop...${NC}"
            kill -9 "$TELEMETRY_PID" 2>/dev/null || true
            echo -e "${GREEN}✓ Telemetry server stopped (forced)${NC}"
            STOPPED=1
        fi
    else
        echo -e "${YELLOW}Telemetry server not running${NC}"
    fi
    rm -f "$TELEMETRY_PID_FILE"
else
    echo -e "${YELLOW}Telemetry server PID file not found${NC}"
fi

# Also kill any orphaned processes on ports 8000 and 8765
echo ""
echo -e "${YELLOW}Checking for orphaned processes...${NC}"

# Port 8000 (HTTP)
HTTP_PROC=$(lsof -ti:8000 2>/dev/null || true)
if [ -n "$HTTP_PROC" ]; then
    echo -e "Found process on port 8000 (PID: ${YELLOW}$HTTP_PROC${NC})"
    kill $HTTP_PROC 2>/dev/null || true
    sleep 1
    if ! kill -0 $HTTP_PROC 2>/dev/null; then
        echo -e "${GREEN}✓ Port 8000 freed${NC}"
        STOPPED=1
    fi
fi

# Port 8765 (WebSocket)
WS_PROC=$(lsof -ti:8765 2>/dev/null || true)
if [ -n "$WS_PROC" ]; then
    echo -e "Found process on port 8765 (PID: ${YELLOW}$WS_PROC${NC})"
    kill $WS_PROC 2>/dev/null || true
    sleep 1
    if ! kill -0 $WS_PROC 2>/dev/null; then
        echo -e "${GREEN}✓ Port 8765 freed${NC}"
        STOPPED=1
    fi
fi

echo ""
if [ $STOPPED -eq 1 ]; then
    echo -e "${GREEN}All servers stopped${NC}"
else
    echo -e "${YELLOW}No servers were running${NC}"
fi
