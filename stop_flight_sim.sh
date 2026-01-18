#!/bin/bash
# =============================================================================
# AIDA Flight Simulator - Stop Script
# =============================================================================
# Stops all flight simulation servers
# =============================================================================

echo "Stopping AIDA Flight Simulator..."

# Kill flight server
pkill -f "python.*run_dynamic_xc" 2>/dev/null && echo "  Flight server stopped" || echo "  Flight server not running"

# Kill HTTP server
pkill -f "python.*http.server.*8000" 2>/dev/null && echo "  HTTP server stopped" || echo "  HTTP server not running"

echo "Done."
