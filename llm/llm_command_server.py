#!/usr/bin/env python3
"""
AIDA LLM Command Server
Processes natural language flight commands using Salesforce xLAM-2-8B

WebSocket server on port 8766 that:
1. Receives natural language commands from the viewer chatbot
2. Parses intent using xLAM-2-8B (optimized for function calling)
3. Sends structured commands to the flight controller
4. Returns pilot-style acknowledgements

Model: Salesforce/Llama-xLAM-2-8b-fc-r (Q4_K_M quantization)
- 8B parameters, 128K context
- State-of-the-art function calling performance
- Trained with APIGen-MT for multi-turn tool use
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
    action: str  # "heading", "altitude", "land", "status", "distance", "airport_info", "conversation", "unknown"
    value: Optional[float] = None
    target: Optional[str] = None  # e.g., "KHUT" for landing or distance query
    raw_text: str = ""
    conversation_response: Optional[str] = None  # For general conversation responses


class FlightCommandParser:
    """Parses natural language into flight commands using xLAM-2-8B"""

    # Define available tools for xLAM
    TOOLS = [
        {
            "name": "set_heading",
            "description": "Turn the aircraft to a specific heading. Use this when the pilot wants to change course.",
            "parameters": {
                "type": "object",
                "properties": {
                    "heading": {
                        "type": "integer",
                        "description": "Target heading in degrees (0-360)"
                    }
                },
                "required": ["heading"]
            }
        },
        {
            "name": "set_altitude",
            "description": "Climb or descend to a specific altitude. Use this when the pilot wants to change altitude.",
            "parameters": {
                "type": "object",
                "properties": {
                    "altitude": {
                        "type": "integer",
                        "description": "Target altitude in feet MSL"
                    }
                },
                "required": ["altitude"]
            }
        },
        {
            "name": "land_at_airport",
            "description": "Divert and land at a specific airport. Use this when the pilot wants to land somewhere.",
            "parameters": {
                "type": "object",
                "properties": {
                    "airport_code": {
                        "type": "string",
                        "description": "ICAO airport code (e.g., KHUT, SN65, KICT, KAAO, K50K)"
                    }
                },
                "required": ["airport_code"]
            }
        },
        {
            "name": "get_status",
            "description": "Get current flight status including position, altitude, heading, and airspeed.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "list_airports",
            "description": "List all available airports in the navigation database with their details.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "get_distance_to_airport",
            "description": "Calculate distance and heading from current position to a specific airport.",
            "parameters": {
                "type": "object",
                "properties": {
                    "airport_code": {
                        "type": "string",
                        "description": "ICAO airport code to calculate distance to"
                    }
                },
                "required": ["airport_code"]
            }
        },
        {
            "name": "get_airport_info",
            "description": "Get detailed information about a specific airport including elevation, runway, and location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "airport_code": {
                        "type": "string",
                        "description": "ICAO airport code"
                    }
                },
                "required": ["airport_code"]
            }
        },
        {
            "name": "return_to_cruise",
            "description": "Return to the planned cruise altitude. Use when pilot says 'return to cruise', 'resume cruise altitude', or wants to go back to the flight plan altitude.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    ]

    def __init__(self, model_path: Optional[str] = None):
        self.llm = None
        self.use_llm = False

        if LLAMA_AVAILABLE and model_path and Path(model_path).exists():
            print(f"[LLM Server] Loading xLAM-2-8B model: {model_path}")
            try:
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=4096,      # Larger context for tool definitions
                    n_threads=8,     # Use more threads (Xeon has 48)
                    n_gpu_layers=35, # Offload to RTX 4060 (8GB VRAM)
                    verbose=False
                )
                self.use_llm = True
                print("[LLM Server] xLAM-2-8B model loaded successfully")
            except Exception as e:
                print(f"[LLM Server] Failed to load LLM: {e}")
                self.use_llm = False
        else:
            print("[LLM Server] Using rule-based command parsing")

    def parse(self, text: str, flight_context: Optional[dict] = None) -> FlightCommand:
        """Parse natural language command into structured command.

        Args:
            text: Natural language command from pilot
            flight_context: Optional dict with current flight state:
                - altitude: Current altitude in feet
                - heading: Current heading in degrees
                - airspeed: Current airspeed in knots
                - phase: Current flight phase
                - origin: Origin airport code
                - destination: Destination airport code
                - cruise_altitude: Planned cruise altitude
                - target_altitude: Current target altitude
        """
        self._flight_context = flight_context or {}
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
        """Parse using xLAM-2-8B with native function calling"""

        # Build xLAM tool-calling prompt
        tools_json = json.dumps(self.TOOLS, indent=2)

        # Build flight context string if available
        ctx = self._flight_context
        context_str = ""
        if ctx:
            origin = ctx.get('origin', 'Unknown')
            dest = ctx.get('destination', 'Unknown')
            alt = ctx.get('altitude', 0)
            hdg = ctx.get('heading', 0)
            spd = ctx.get('airspeed', 0)
            phase = ctx.get('phase', 'Unknown')
            cruise = ctx.get('cruise_altitude', 0)
            dist = ctx.get('distance', 0)

            context_str = f"""
CURRENT FLIGHT STATUS:
- Route: {origin} → {dest}
- Phase: {phase}
- Altitude: {alt:.0f} ft (cruise altitude: {cruise:.0f} ft)
- Heading: {hdg:.0f}°
- Airspeed: {spd:.0f} kts
- Distance to destination: {dist:.1f} nm
"""

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|}}>

You are AIDA, an intelligent flight assistant for a Cessna 172 Skyhawk flight simulator. You help pilots with flight commands and questions.

Aircraft Info: You are flying a Cessna 172 Skyhawk - a single-engine, four-seat, high-wing aircraft. Cruise speed ~120 knots, service ceiling 13,500 ft, range ~640 nm.
{context_str}
You can use tools for specific actions. When using tools, respond with a JSON array:
[{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}]

Available tools:
{tools_json}

IMPORTANT Guidelines:
- For flight commands (turn, climb, descend, land at SPECIFIC airport), use the appropriate tool
- For "return to cruise" or "resume cruise altitude", use the return_to_cruise tool (NOT set_altitude)
- For distance/heading to airports IN THE DATABASE (SN65, KHUT, KICT, KAAO, K50K), use get_distance_to_airport
- For airport details, use get_airport_info
- ONLY use get_status when user explicitly asks for "status", "position", or "where am I"
- For general questions about the aircraft, aviation, what's around, or conversation, respond naturally WITHOUT using any tools
- If user asks about a location NOT in the database, respond that it's not in the nav database
- Keep responses concise and pilot-friendly
<|eot_id|><|start_header_id|>user<|end_header_id|>

{text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        try:
            response = self.llm(
                prompt,
                max_tokens=200,
                temperature=0.1,
                stop=["<|eot_id|>", "<|end_of_text|>"]
            )
            result_text = response["choices"][0]["text"].strip()
            print(f"[LLM Server] xLAM response: {result_text}")

            # Parse tool call from response
            # xLAM returns: [{"name": "tool_name", "arguments": {...}}]
            tool_match = re.search(r'\[\s*\{[^]]+\}\s*\]', result_text, re.DOTALL)
            if tool_match:
                try:
                    tool_calls = json.loads(tool_match.group())
                    if tool_calls and len(tool_calls) > 0:
                        tool_call = tool_calls[0]
                        tool_name = tool_call.get("name", "")
                        args = tool_call.get("arguments", {})

                        # Map tool calls to FlightCommand
                        if tool_name == "set_heading":
                            return FlightCommand(
                                action="heading",
                                value=args.get("heading"),
                                raw_text=text
                            )
                        elif tool_name == "set_altitude":
                            return FlightCommand(
                                action="altitude",
                                value=args.get("altitude"),
                                raw_text=text
                            )
                        elif tool_name == "land_at_airport":
                            return FlightCommand(
                                action="land",
                                target=args.get("airport_code"),
                                raw_text=text
                            )
                        elif tool_name == "get_status":
                            return FlightCommand(
                                action="status",
                                raw_text=text
                            )
                        elif tool_name == "list_airports":
                            return FlightCommand(
                                action="list_airports",
                                raw_text=text
                            )
                        elif tool_name == "get_distance_to_airport":
                            return FlightCommand(
                                action="distance",
                                target=args.get("airport_code"),
                                raw_text=text
                            )
                        elif tool_name == "get_airport_info":
                            return FlightCommand(
                                action="airport_info",
                                target=args.get("airport_code"),
                                raw_text=text
                            )
                        elif tool_name == "return_to_cruise":
                            # Get cruise altitude from flight context
                            cruise_alt = self._flight_context.get("cruise_altitude", 5500)
                            return FlightCommand(
                                action="altitude",
                                value=cruise_alt,
                                raw_text=text
                            )
                except json.JSONDecodeError as e:
                    print(f"[LLM Server] Failed to parse tool call JSON: {e}")

            # No tool call - treat as conversational response
            if result_text and not result_text.startswith('['):
                return FlightCommand(
                    action="conversation",
                    raw_text=text,
                    conversation_response=result_text
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

    # Available airports in the navigation database with coordinates
    AIRPORTS = {
        "SN65": {"name": "Lake Waltanna Airport", "elevation": 1448, "lat": 38.0833, "lon": -98.1167},
        "KHUT": {"name": "Hutchinson Regional Airport", "elevation": 1542, "lat": 38.0655, "lon": -97.8606},
        "KICT": {"name": "Wichita Eisenhower National", "elevation": 1333, "lat": 37.6499, "lon": -97.4331},
        "KAAO": {"name": "Colonel James Jabara Airport", "elevation": 1421, "lat": 37.7476, "lon": -97.2211},
        "K50K": {"name": "Pawnee Municipal Airport", "elevation": 2200, "lat": 37.8344, "lon": -99.3375},
    }

    def __init__(self, telemetry_url: str = "ws://localhost:8765"):
        self.telemetry_url = telemetry_url
        self.override_active = False
        self.override_heading = None
        self.override_altitude = None
        self.override_land_target = None
        self.current_state = {
            "altitude": 0,
            "heading": 0,
            "airspeed": 0,
            "phase": "UNKNOWN",
            "distance": 0,
            "origin": None,
            "destination": None,
            "lat": 0.0,
            "lon": 0.0,
            # Flight plan context
            "cruise_altitude": 0,
            "target_altitude": 0,
            "target_heading": 0,
        }
        self._telemetry_task = None
        self._telemetry_ws = None

        # Aircraft limits
        self.min_altitude = 1500  # ft AGL (ground at ~1400ft)
        self.max_altitude = 12000  # ft - service ceiling
        self.max_bank_angle = 30  # degrees

    @staticmethod
    def calculate_distance_and_heading(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
        """
        Calculate distance (nm) and heading (degrees) from point 1 to point 2.
        Uses haversine formula for distance and forward azimuth for heading.
        """
        import math

        # Convert to radians
        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        lon1_r = math.radians(lon1)
        lon2_r = math.radians(lon2)

        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r

        # Haversine formula for distance
        a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        earth_radius_nm = 3440.065  # Earth radius in nautical miles
        distance = earth_radius_nm * c

        # Forward azimuth (heading)
        x = math.sin(dlon) * math.cos(lat2_r)
        y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
        heading = math.degrees(math.atan2(x, y))
        heading = (heading + 360) % 360  # Normalize to 0-360

        return distance, heading

    async def start_telemetry_listener(self):
        """Start background task to listen for telemetry updates"""
        self._telemetry_task = asyncio.create_task(self._listen_telemetry())

    async def _listen_telemetry(self):
        """Listen for telemetry updates from flight server"""
        import websockets
        while True:
            try:
                async with websockets.connect(self.telemetry_url) as ws:
                    self._telemetry_ws = ws
                    print(f"[LLM Server] Connected to telemetry server at {self.telemetry_url}")
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            # Update current state from telemetry
                            self.current_state["altitude"] = data.get("altitude_ft", 0)
                            self.current_state["heading"] = data.get("heading_deg", 0)
                            self.current_state["airspeed"] = data.get("airspeed_kts", 0)
                            self.current_state["phase"] = data.get("phase", "UNKNOWN")
                            self.current_state["distance"] = data.get("distance_nm", 0)
                            self.current_state["origin"] = data.get("origin")
                            self.current_state["destination"] = data.get("destination")
                            self.current_state["lat"] = data.get("lat", 0.0)
                            self.current_state["lon"] = data.get("lon", 0.0)
                            # Flight plan context
                            self.current_state["cruise_altitude"] = data.get("cruise_altitude_ft", 0)
                            self.current_state["target_altitude"] = data.get("target_altitude_ft", 0)
                            self.current_state["target_heading"] = data.get("target_heading_deg", 0)
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                print(f"[LLM Server] Telemetry connection error: {e}, reconnecting...")
                await asyncio.sleep(2)  # Wait before reconnecting

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
            # Check if airport is in database
            if cmd.target and cmd.target.upper() not in self.AIRPORTS:
                available = ", ".join(self.AIRPORTS.keys())
                return False, f"Airport {cmd.target} not in navigation database. Available: {available}"

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

        elif cmd.action == "list_airports":
            result["success"] = True
            result["airports"] = self.AIRPORTS.copy()

        elif cmd.action == "distance":
            # Calculate distance to a specific airport
            target = cmd.target.upper() if cmd.target else None
            if target and target in self.AIRPORTS:
                airport = self.AIRPORTS[target]
                current_lat = self.current_state.get("lat", 0.0)
                current_lon = self.current_state.get("lon", 0.0)

                if current_lat != 0.0 and current_lon != 0.0:
                    distance, heading = self.calculate_distance_and_heading(
                        current_lat, current_lon,
                        airport["lat"], airport["lon"]
                    )
                    result["success"] = True
                    result["target"] = target
                    result["airport_name"] = airport["name"]
                    result["distance_nm"] = round(distance, 1)
                    result["heading_to"] = round(heading)
                else:
                    result["success"] = False
                    result["message"] = "Current position not available"
            else:
                result["success"] = False
                available = ", ".join(self.AIRPORTS.keys())
                result["message"] = f"Airport {target} not found. Available: {available}"

        elif cmd.action == "airport_info":
            # Get detailed info about an airport
            target = cmd.target.upper() if cmd.target else None
            if target and target in self.AIRPORTS:
                airport = self.AIRPORTS[target]
                result["success"] = True
                result["airport"] = {
                    "code": target,
                    "name": airport["name"],
                    "elevation": airport["elevation"],
                    "lat": airport["lat"],
                    "lon": airport["lon"]
                }
            else:
                result["success"] = False
                available = ", ".join(self.AIRPORTS.keys())
                result["message"] = f"Airport {target} not found. Available: {available}"

        elif cmd.action == "conversation":
            # General conversation - pass through the LLM's response
            result["success"] = True
            result["response"] = cmd.conversation_response

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

            # Parse the command with flight context
            cmd = self.parser.parse(text, flight_context=self.controller.current_state)
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
            elif cmd.action == "list_airports":
                airports = result.get("airports", {})
                apt_list = ", ".join([f"{code} ({info['name']})" for code, info in airports.items()])
                message = f"Available airports: {apt_list}"
            elif cmd.action == "distance":
                if result.get("success"):
                    dist = result.get("distance_nm", 0)
                    hdg = result.get("heading_to", 0)
                    target = result.get("target", "")
                    name = result.get("airport_name", "")
                    message = f"{target} ({name}) is {dist:.1f} nm away on heading {hdg:03d}°."
                else:
                    message = result.get("message", "Unable to calculate distance.")
            elif cmd.action == "airport_info":
                if result.get("success"):
                    apt = result.get("airport", {})
                    message = (f"{apt['code']} - {apt['name']}: "
                              f"Elevation {apt['elevation']} ft, "
                              f"Coordinates {apt['lat']:.4f}°N, {abs(apt['lon']):.4f}°W")
                else:
                    message = result.get("message", "Airport not found.")
            elif cmd.action == "conversation":
                # Use the LLM's natural language response
                message = result.get("response", "I'm not sure how to respond to that.")
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

        # Start telemetry listener to get real-time flight data
        await self.controller.start_telemetry_listener()

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
    parser.add_argument("--model", default=None, help="Path to LLM GGUF model")

    args = parser.parse_args()

    # Default model path - prefer xLAM-2-8B, fallback to Phi-3
    if args.model is None:
        xlam_model = Path(__file__).parent / "models" / "Llama-xLAM-2-8B-fc-r-Q4_K_M.gguf"
        phi3_model = Path(__file__).parent / "models" / "Phi-3-mini-4k-instruct-q4.gguf"

        if xlam_model.exists():
            args.model = str(xlam_model)
            print(f"[LLM Server] Using xLAM-2-8B model")
        elif phi3_model.exists():
            args.model = str(phi3_model)
            print(f"[LLM Server] Using Phi-3 model (fallback)")

    server = LLMCommandServer(args.host, args.port, args.model)

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[LLM Server] Shutting down...")


if __name__ == "__main__":
    main()
