#!/usr/bin/env python3
"""Check telemetry data to debug viewer issues."""

import websocket
import json

try:
    ws = websocket.create_connection('ws://localhost:8765', timeout=2)
    data = json.loads(ws.recv())
    print('Current telemetry data:')
    print(f"  Position (m, NED): {data['position']}")
    print(f"  Quaternion: {data['quaternion']}")
    print(f"  Throttle: {data.get('throttle', 'N/A')}")

    # Convert to feet for display
    pos_ft = [p * 3.28084 for p in data['position']]
    print(f"  Position (ft): [{pos_ft[0]:.1f}, {pos_ft[1]:.1f}, {pos_ft[2]:.1f}]")

    # Check if position makes sense
    print("\nDiagnostics:")
    x, y, z = data['position']
    print(f"  X (lateral): {x:.2f} m ({x*3.28:.1f} ft)")
    print(f"  Y (along runway): {y:.2f} m ({y*3.28:.1f} ft)")
    print(f"  Z (altitude, NED down): {z:.2f} m ({z*3.28:.1f} ft)")

    if abs(x) > 1000 or abs(y) > 1000 or abs(z) > 100:
        print("\n  WARNING: Position values seem very large!")

    ws.close()
    print("\nTelemetry connection OK")
except Exception as e:
    print(f'Error connecting to telemetry: {e}')
