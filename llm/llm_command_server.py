#!/usr/bin/env python3
"""
AIDA LLM Command Server
Processes natural language flight commands using Phi-3 Mini

WebSocket server on port 8766 that:
1. Receives natural language commands from the viewer chatbot
2. Parses intent using Phi-3 LLM
3. Sends structured commands to the flight controller
4. Returns pilot-style acknowledgements
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import websockets
from dataclasses import dataclass

# Add AIDA to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("[LLM Server] llama-cpp-python not available, using rule-based parsing")


@dataclass
class FlightCommand:
    """Structured flight command"""
    action: str  # "heading", "altitude", "land", "status", "unknown"
    value: Optional[float] = None
    target: Optional[str] = None  # e.g., "KHUT" for landing
    raw_text: str = ""


class FlightCommandParser:
    """Parses natural language into flight commands"""

    def __init__(self, model_path: Optional[str] = None):
        self.llm = None
        self.use_llm = False

        if LLAMA_AVAILABLE and model_path and Path(model_path).exists():
            print(f"[LLM Server] Loading Phi-3 model: {model_path}")
            try:
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=4,
                    n_gpu_layers=0,  # CPU only for now
                    verbose=False
                )
                self.use_llm = True
                print("[LLM Server] Phi-3 model loaded successfully")
            except Exception as e:
                print(f"[LLM Server] Failed to load LLM: {e}")
                self.use_llm = False
        else:
            print("[LLM Server] Using rule-based command parsing")

    def parse(self, text: str) -> FlightCommand:
        """Parse natural language command into structured command"""
        text_lower = text.lower().strip()

        # Try LLM first if available
        if self.use_llm:
            return self._parse_with_llm(text)

        # Fall back to rule-based parsing
        return self._parse_rules(text_lower, text)

    def _parse_rules(self, text_lower: str, original: str) -> FlightCommand:
        """Rule-based command parsing"""

        # Status request
        if any(word in text_lower for word in ["status", "report", "position", "where"]):
            return FlightCommand(action="status", raw_text=original)

        # Heading commands
        heading_patterns = [
            r"(?:turn|fly|head|go)?\s*(?:to)?\s*heading\s*(\d{1,3})",
            r"hdg\s*(\d{1,3})",
            r"heading\s*(\d{1,3})",
            r"turn\s*(?:to)?\s*(\d{1,3})\s*(?:degrees?)?",
            r"fly\s*(\d{1,3})\s*(?:degrees?)?",
        ]
        for pattern in heading_patterns:
            match = re.search(pattern, text_lower)
            if match:
                heading = int(match.group(1))
                if 0 <= heading <= 360:
                    return FlightCommand(action="heading", value=heading, raw_text=original)

        # Altitude commands
        altitude_patterns = [
            r"(?:climb|descend|go)?\s*(?:to)?\s*(?:altitude)?\s*(\d{3,5})\s*(?:feet|ft)?",
            r"alt\s*(\d{3,5})",
            r"altitude\s*(\d{3,5})",
            r"climb\s*(?:to)?\s*(\d{3,5})",
            r"descend\s*(?:to)?\s*(\d{3,5})",
            r"(\d{4,5})\s*(?:feet|ft)",
        ]
        for pattern in altitude_patterns:
            match = re.search(pattern, text_lower)
            if match:
                altitude = int(match.group(1))
                # Sanity check: altitude should be between 500 and 15000 ft
                if 500 <= altitude <= 15000:
                    return FlightCommand(action="altitude", value=altitude, raw_text=original)

        # Landing commands
        landing_patterns = [
            r"land\s*(?:at)?\s*(\w+)",
            r"proceed\s*(?:to)?\s*(\w+)\s*(?:for)?\s*landing",
            r"divert\s*(?:to)?\s*(\w+)",
            r"go\s*(?:to)?\s*(\w+)\s*(?:and)?\s*land",
            r"resume\s*(?:mission|landing)",
        ]
        for pattern in landing_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if "resume" in text_lower:
                    return FlightCommand(action="land", target="KHUT", raw_text=original)
                target = match.group(1).upper()
                return FlightCommand(action="land", target=target, raw_text=original)

        # Unknown command
        return FlightCommand(action="unknown", raw_text=original)

    def _parse_with_llm(self, text: str) -> FlightCommand:
        """Parse using Phi-3 LLM"""
        prompt = f"""<|system|>
You are a flight command parser. Extract the intent from the pilot's command.
Return ONLY a JSON object with these fields:
- action: one of "heading", "altitude", "land", "status", "unknown"
- value: number (for heading 0-360, for altitude in feet)
- target: string (airport code for landing, e.g., "KHUT")

Examples:
"Turn to heading 270" -> {{"action": "heading", "value": 270}}
"Climb to 7000 feet" -> {{"action": "altitude", "value": 7000}}
"Land at KHUT" -> {{"action": "land", "target": "KHUT"}}
"What's my status?" -> {{"action": "status"}}
<|end|>
<|user|>
{text}
<|end|>
<|assistant|>
"""
        try:
            response = self.llm(
                prompt,
                max_tokens=100,
                temperature=0.1,
                stop=["<|end|>", "\n\n"]
            )
            result_text = response["choices"][0]["text"].strip()

            # Parse JSON from response
            json_match = re.search(r'\{[^}]+\}', result_text)
            if json_match:
                data = json.loads(json_match.group())
                return FlightCommand(
                    action=data.get("action", "unknown"),
                    value=data.get("value"),
                    target=data.get("target"),
                    raw_text=text
                )
        except Exception as e:
            print(f"[LLM Server] LLM parse error: {e}")

        # Fall back to rules
        return self._parse_rules(text.lower(), text)


class PilotResponder:
    """Generates pilot-style responses"""

    def __init__(self, callsign: str = "AIDA-1"):
        self.callsign = callsign

    def acknowledge_heading(self, heading: float, current_heading: Optional[float] = None) -> str:
        direction = ""
        if current_heading is not None:
            diff = (heading - current_heading + 180) % 360 - 180
            if diff > 0:
                direction = "right "
            elif diff < 0:
                direction = "left "

        responses = [
            f"Roger, turning {direction}to heading {int(heading):03d}.",
            f"Copy, coming {direction}to {int(heading):03d}.",
            f"Wilco, heading {int(heading):03d}.",
        ]
        import random
        return random.choice(responses)

    def acknowledge_altitude(self, altitude: float, current_altitude: Optional[float] = None) -> str:
        action = "changing altitude"
        if current_altitude is not None:
            if altitude > current_altitude:
                action = "climbing"
            else:
                action = "descending"

        responses = [
            f"Roger, {action} to {int(altitude):,} feet.",
            f"Copy, {action} to flight level {int(altitude/100)} ({int(altitude):,} feet).",
            f"Wilco, {action} to {int(altitude):,}.",
        ]
        import random
        return random.choice(responses)

    def acknowledge_land(self, target: str) -> str:
        responses = [
            f"Roger, proceeding direct {target} for landing.",
            f"Copy, setting up approach to {target}.",
            f"Wilco, diverting to {target}. Configuring for approach.",
        ]
        import random
        return random.choice(responses)

    def report_status(self, state: Dict[str, Any]) -> str:
        alt = state.get("altitude", 0)
        hdg = state.get("heading", 0)
        spd = state.get("airspeed", 0)
        phase = state.get("phase", "UNKNOWN")
        dist = state.get("distance", 0)

        return (f"Current status: {phase} phase, "
                f"altitude {int(alt):,} feet, heading {int(hdg):03d}, "
                f"airspeed {int(spd)} knots, {dist:.1f} nm to destination.")

    def reject_command(self, reason: str) -> str:
        return f"Unable to comply. {reason}"

    def unknown_command(self) -> str:
        return "Say again? I didn't understand that command. Try 'heading 270', 'altitude 7000', or 'land at KHUT'."


class FlightController:
    """Interface to the flight simulation controller"""

    def __init__(self):
        self.override_active = False
        self.override_heading = None
        self.override_altitude = None
        self.override_land_target = None
        self.current_state = {
            "altitude": 5500,
            "heading": 350,
            "airspeed": 120,
            "phase": "CRUISE",
            "distance": 15.0
        }

        # Aircraft limits
        self.min_altitude = 1500  # ft AGL (ground at ~1400ft)
        self.max_altitude = 12000  # ft - service ceiling
        self.max_bank_angle = 30  # degrees

    def validate_command(self, cmd: FlightCommand) -> tuple[bool, str]:
        """Validate command against aircraft limits"""
        if cmd.action == "altitude":
            if cmd.value < self.min_altitude:
                return False, f"Altitude {cmd.value} ft is below minimum safe altitude ({self.min_altitude} ft)."
            if cmd.value > self.max_altitude:
                return False, f"Altitude {cmd.value} ft exceeds aircraft service ceiling ({self.max_altitude} ft)."

        if cmd.action == "heading":
            if cmd.value < 0 or cmd.value > 360:
                return False, f"Invalid heading {cmd.value}. Must be between 0 and 360."

        if cmd.action == "land":
            # For now, only KHUT is supported
            if cmd.target and cmd.target.upper() != "KHUT":
                return False, f"Airport {cmd.target} not in navigation database. Only KHUT available."

        return True, ""

    def execute_command(self, cmd: FlightCommand) -> Dict[str, Any]:
        """Execute the flight command"""
        result = {
            "success": False,
            "action": cmd.action,
            "message": ""
        }

        if cmd.action == "heading":
            self.override_heading = cmd.value
            self.override_active = True
            result["success"] = True
            result["value"] = cmd.value

        elif cmd.action == "altitude":
            self.override_altitude = cmd.value
            self.override_active = True
            result["success"] = True
            result["value"] = cmd.value

        elif cmd.action == "land":
            self.override_land_target = cmd.target or "KHUT"
            self.override_active = True
            result["success"] = True
            result["target"] = self.override_land_target

        elif cmd.action == "status":
            result["success"] = True
            result["state"] = self.current_state.copy()

        return result

    def update_state(self, state: Dict[str, Any]):
        """Update current flight state from telemetry"""
        self.current_state.update(state)

    def get_overrides(self) -> Dict[str, Any]:
        """Get current override commands for the flight controller"""
        return {
            "active": self.override_active,
            "heading": self.override_heading,
            "altitude": self.override_altitude,
            "land_target": self.override_land_target
        }


class LLMCommandServer:
    """WebSocket server for LLM flight commands"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8766, model_path: Optional[str] = None):
        self.host = host
        self.port = port
        self.parser = FlightCommandParser(model_path)
        self.responder = PilotResponder()
        self.controller = FlightController()
        self.clients = set()

    async def handle_client(self, websocket):
        """Handle a connected client"""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        print(f"[LLM Server] Client connected: {client_addr}")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self.process_message(data)
                    await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON"
                    }))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"[LLM Server] Client disconnected: {client_addr}")

    async def process_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message and return response"""
        msg_type = data.get("type", "")

        if msg_type == "command":
            text = data.get("text", "")
            print(f"[LLM Server] Command received: '{text}'")

            # Parse the command
            cmd = self.parser.parse(text)
            print(f"[LLM Server] Parsed: action={cmd.action}, value={cmd.value}, target={cmd.target}")

            # Handle unknown commands
            if cmd.action == "unknown":
                return {
                    "type": "response",
                    "message": self.responder.unknown_command(),
                    "action": None
                }

            # Validate against aircraft limits
            valid, reason = self.controller.validate_command(cmd)
            if not valid:
                return {
                    "type": "response",
                    "message": self.responder.reject_command(reason),
                    "action": None
                }

            # Execute the command
            result = self.controller.execute_command(cmd)

            # Generate pilot response
            if cmd.action == "heading":
                message = self.responder.acknowledge_heading(
                    cmd.value,
                    self.controller.current_state.get("heading")
                )
            elif cmd.action == "altitude":
                message = self.responder.acknowledge_altitude(
                    cmd.value,
                    self.controller.current_state.get("altitude")
                )
            elif cmd.action == "land":
                message = self.responder.acknowledge_land(cmd.target or "KHUT")
            elif cmd.action == "status":
                message = self.responder.report_status(result.get("state", {}))
            else:
                message = "Command acknowledged."

            return {
                "type": "response",
                "message": message,
                "action": result
            }

        elif msg_type == "telemetry":
            # Update flight state from viewer
            state = data.get("state", {})
            self.controller.update_state(state)
            return {"type": "ack"}

        return {"type": "error", "message": "Unknown message type"}

    async def broadcast_overrides(self):
        """Periodically broadcast override commands to connected telemetry clients"""
        while True:
            if self.controller.override_active:
                overrides = self.controller.get_overrides()
                message = json.dumps({"type": "override", "data": overrides})
                for client in self.clients:
                    try:
                        await client.send(message)
                    except:
                        pass
            await asyncio.sleep(0.5)

    async def run(self):
        """Start the WebSocket server"""
        print(f"[LLM Server] Starting on ws://{self.host}:{self.port}")

        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"[LLM Server] Ready to accept connections")
            # Start override broadcaster
            asyncio.create_task(self.broadcast_overrides())
            await asyncio.Future()  # Run forever


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AIDA LLM Command Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8766, help="Port to listen on")
    parser.add_argument("--model", default=None, help="Path to Phi-3 GGUF model")

    args = parser.parse_args()

    # Default model path
    if args.model is None:
        default_model = Path(__file__).parent.parent / "llm" / "models" / "Phi-3-mini-4k-instruct-q4.gguf"
        if default_model.exists():
            args.model = str(default_model)

    server = LLMCommandServer(args.host, args.port, args.model)

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[LLM Server] Shutting down...")


if __name__ == "__main__":
    main()
