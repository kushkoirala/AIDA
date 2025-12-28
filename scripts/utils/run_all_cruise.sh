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

# Start telemetry server (background)
export PYTHONPATH=$(pwd)

# Allow overriding the checkpoint path; if missing, run without a policy (idle throttle).
DEFAULT_CKPT="checkpoints/ppo_flight_final.pt"
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CKPT}"
TELEMETRY_ARGS="--task cruise --host 127.0.0.1 --wait-for-client --reset-on-connect --twin-warmup"
if [ -n "$CHECKPOINT" ] && [ -f "$CHECKPOINT" ]; then
  TELEMETRY_ARGS="--checkpoint $CHECKPOINT $TELEMETRY_ARGS"
else
  echo "Warning: checkpoint not found at '$CHECKPOINT'; starting telemetry without a policy." >&2
fi
if [ -n "$LOG_DIR" ]; then
  TELEMETRY_ARGS="$TELEMETRY_ARGS --log-dir $LOG_DIR"
  echo "Logging telemetry to $LOG_DIR"
fi

.venv-linux/bin/python scripts/run_sim_with_telemetry.py $TELEMETRY_ARGS &
TELEMETRY_PID=$!
echo "Telemetry server started with PID $TELEMETRY_PID on port 8765 (wait-for-client)."

# Wait for both to exit
wait $HTTP_PID $TELEMETRY_PID
