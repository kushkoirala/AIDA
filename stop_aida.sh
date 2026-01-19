#!/bin/bash
# =============================================================================
# AIDA Flight Simulator - Stop Script
# =============================================================================
# Stops all AIDA servers (HTTP, Flight Dynamics, LLM)
# =============================================================================

echo "=============================================="
echo "  Stopping AIDA Flight Simulator"
echo "=============================================="

# Stop flight dynamics server
echo "Stopping flight dynamics server..."
pkill -f "python.*run_dynamic_xc" 2>/dev/null && echo "  Stopped." || echo "  Not running."

# Stop HTTP server
echo "Stopping HTTP viewer server..."
pkill -f "python.*http.server.*8000" 2>/dev/null && echo "  Stopped." || echo "  Not running."

# Stop LLM server
echo "Stopping LLM command server..."
pkill -f "python.*llm_command_server" 2>/dev/null && echo "  Stopped." || echo "  Not running."

echo ""
echo "All AIDA servers stopped."
echo "=============================================="
