# AIDA LLM Integration

Natural language command interface for autonomous flight control.

## Overview

This module enables voice/text commands to control the aircraft:

```
User: "Turn heading 270 degrees"
  │
  ▼
┌─────────────────┐
│  LLM Parser     │  ← Phi-3 Mini / Llama 3.2 (local)
│  (or Rule-based)│
└────────┬────────┘
         │ FlightCommand
         ▼
┌─────────────────┐
│  Validator      │  ← Flight envelope protection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Executor       │  ← Updates target states
└────────┬────────┘
         │ FlightTargets
         ▼
┌─────────────────┐
│  Expert + NN    │  ← Tracks targets
└─────────────────┘
```

## Supported Commands

| Command Type | Examples |
|--------------|----------|
| **Heading** | "Turn heading 270", "Fly heading 090" |
| **Altitude** | "Climb to 5000 feet", "Descend to 3000" |
| **Airspeed** | "Speed 80 knots", "Slow to 70" |
| **Turn** | "Turn left 30 degrees", "Right 45" |
| **Vertical Speed** | "Climb at 500 fpm", "Descend at 800" |
| **Flaps** | "Flaps 20", "Flaps full" |
| **Go Around** | "Go around", "Missed approach" |
| **Hold** | "Maintain current", "Hold altitude" |

## Installation

### 1. Install llama-cpp-python (for LLM support)

```bash
# With CUDA support (RTX 4060)
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
```

### 2. Download a model

Recommended models for RTX 4060 (8GB VRAM):

| Model | Size | Download |
|-------|------|----------|
| Phi-3 Mini Q4 | ~2.5 GB | [HuggingFace](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) |
| Llama 3.2 3B Q4 | ~2 GB | [HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct-GGUF) |

```bash
# Download to models directory
cd /home/AIDA/llm/models
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
```

## Usage

### Rule-Based Parser (No LLM required)

```python
from llm import RuleBasedCommandParser, FlightCommandExecutor

parser = RuleBasedCommandParser()
executor = FlightCommandExecutor()

# Parse command
command = parser.parse("Turn heading 270")
print(command.to_dict())
# {"command": "SET_HEADING", "heading_deg": 270, "confidence": 0.9}

# Execute
success, msg, targets = executor.execute(command)
print(msg)  # "Setting heading to 270°"
```

### LLM-Based Parser

```python
from llm import LLMCommandParser

parser = LLMCommandParser(model_path="models/Phi-3-mini-4k-instruct-q4.gguf")

# More flexible parsing
command = parser.parse("I'd like to turn left about 30 degrees please")
print(command.to_dict())
# {"command": "TURN", "turn_degrees": 30, "turn_direction": "left"}
```

### Integration with Flight Simulation

```python
from llm import LLMCommandParser, FlightCommandExecutor, TargetTrackingController

# Initialize
parser = LLMCommandParser(model_path="models/phi-3-mini.gguf")
executor = FlightCommandExecutor()
tracker = TargetTrackingController(executor)

# In simulation loop
while running:
    # Update executor with current state
    executor.update_state({
        "heading_deg": current_heading,
        "altitude_ft": current_altitude,
        "airspeed_kts": current_airspeed,
    })

    # Process any incoming commands
    if new_command_text:
        command = parser.parse(new_command_text)
        success, msg, targets = executor.execute(command)
        print(f"Command: {msg}")

    # Get expert action
    expert_action = expert_controller.compute_action(state)

    # Modify to track LLM targets
    modified_action = tracker.modify_expert_action(expert_action, current_state)

    # Apply to simulation
    sim.step(modified_action)
```

## Safety Features

### Flight Envelope Protection

All commands are validated against Cessna 172 limits:

- Stall speed: 45 kts (dirty)
- Vne: 140 kts
- Service ceiling: 14,000 ft
- Max climb rate: 1,000 fpm
- Max descent rate: 1,500 fpm
- Max bank angle: 60°

Commands exceeding these limits are rejected.

### Confidence Scoring

Each parsed command includes a confidence score:
- 0.95: LLM parse (high confidence)
- 0.90: Rule-based exact match
- 0.85: Rule-based partial match
- 0.00: Unknown command

## Files

```
llm/
├── __init__.py           # Module exports
├── flight_commands.py    # Command types, parsers, validator
├── command_executor.py   # Executor, target tracking
├── models/               # LLM model files (GGUF)
└── README.md             # This file
```

## Hardware Requirements

| Configuration | VRAM Usage | Performance |
|---------------|------------|-------------|
| Rule-based only | 0 GB | Instant |
| Phi-3 Mini (3.8B Q4) | ~2.5 GB | 80-100 tok/s |
| Llama 3.2 3B (Q4) | ~2 GB | 100+ tok/s |
| Mistral 7B (Q4) | ~4.5 GB | 40-60 tok/s |

RTX 4060 (8GB) can run 3-7B models alongside flight simulation.

## Future Enhancements

- [ ] Voice input via Whisper
- [ ] Waypoint/navigation commands
- [ ] Multi-turn conversation context
- [ ] Command confirmation for critical actions
- [ ] Integration with flight plan
