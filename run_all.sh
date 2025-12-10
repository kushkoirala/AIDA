#!/bin/bash
# Start both the telemetry server and the viewer HTTP server for robust local testing.
# Usage: ./run_all.sh

# Kill any previous servers on the required ports
kill $(lsof -t -i:8080) 2>/dev/null || true
kill $(lsof -t -i:8765) 2>/dev/null || true

# Start HTTP server for viewer (background)
python3 -m http.server 8080 --directory viewer/public &
HTTP_PID=$!
echo "Viewer HTTP server started with PID $HTTP_PID on port 8080."

# Start telemetry server (background)
export PYTHONPATH=$(pwd)
.venv/bin/python scripts/run_sim_with_telemetry.py \
  --checkpoint checkpoints/ppo_flight_final.pt \
  --task takeoff \
  --wait-for-client \
  --reset-on-connect &
TELEMETRY_PID=$!
echo "Telemetry server started with PID $TELEMETRY_PID on port 8765 (wait-for-client)."

# Wait for both to exit
wait $HTTP_PID $TELEMETRY_PID
