#!/bin/bash
# Start both the telemetry server and the viewer HTTP server for robust local testing.
# Usage: ./run_all.sh

# Kill any previous servers on the required ports
kill $(lsof -t -i:8080) 2>/dev/null || true
kill $(lsof -t -i:8765) 2>/dev/null || true

# Start HTTP server for viewer (background). Sandbox restricts 0.0.0.0 binds,
# so bind explicitly to localhost.
python3 -m http.server 8080 --bind 127.0.0.1 --directory viewer/public &
HTTP_PID=$!
echo "Viewer HTTP server started with PID $HTTP_PID on port 8080."

# Start telemetry server (background) using cruise hold checkpoint (seed2, curriculum level 0)
export PYTHONPATH=$(pwd)
.venv/bin/python scripts/run_sim_with_telemetry.py \
  --checkpoint checkpoints/runs/ppo_sb3_vec_seed2_t1M_cruisehold_bc.zip \
  --task takeoff_and_cruise \
  --curriculum-level 0 \
  --host 127.0.0.1 \
  --port 8765 \
  --wait-for-client \
  --reset-on-connect &
TELEMETRY_PID=$!
echo "Telemetry server started with PID $TELEMETRY_PID on port 8765 (wait-for-client) using cruise seed2 checkpoint."

# Wait for both to exit
wait $HTTP_PID $TELEMETRY_PID
