#!/usr/bin/env python3
"""
AIDA Unified Transformer Architecture

A single transformer model that handles both:
1. NLP Command Parsing: "climb to 5000 feet" -> {"command": "SET_ALTITUDE", "altitude_ft": 5000}
2. Trajectory Prediction: state sequence -> action sequence

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                   Unified Transformer                        │
    │                                                              │
    │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
    │  │ Text Encoder │    │ State Encoder│    │Shared Encoder│  │
    │  │  (Embedding) │    │   (Linear)   │    │ (Transformer)│  │
    │  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
    │         │                   │                   │           │
    │         └───────────────────┴───────────────────┘           │
    │                             │                               │
    │                    ┌────────┴────────┐                      │
    │                    │ Task Router     │                      │
    │                    │ (learned gate)  │                      │
    │                    └────────┬────────┘                      │
    │         ┌───────────────────┼───────────────────┐           │
    │         ▼                   ▼                   ▼           │
    │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
    │  │Command Head  │    │ Action Head  │    │ Value Head   │  │
    │  │(classif+reg) │    │  (control)   │    │  (critic)    │  │
    │  └──────────────┘    └──────────────┘    └──────────────┘  │
    └─────────────────────────────────────────────────────────────┘

Author: Kushal Koirala (with Claude Code)
Date: January 2026
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Supported task types for the unified model."""
    NLP_COMMAND = "nlp_command"      # Text -> Flight Command
    TRAJECTORY = "trajectory"         # State sequence -> Action sequence
    IMITATION = "imitation"          # State -> Expert action (BC)
    REINFORCEMENT = "reinforcement"  # State -> Action + Value (RL)
    ROUTE_GENERATION = "route_generation"  # Route spec -> Controller params


@dataclass
class ModelConfig:
    """Configuration for the unified transformer."""
    # Shared encoder
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # NLP settings
    vocab_size: int = 8000
    max_seq_len: int = 128
    num_command_types: int = 10  # SET_HEADING, SET_ALTITUDE, etc.

    # Flight state/action dimensions
    state_dim: int = 12   # x,y,z, u,v,w, phi,theta,psi, p,q,r
    action_dim: int = 7   # throttle, elevator, aileron, rudder, flaps, spoilers, brakes

    # Trajectory settings
    trajectory_seq_len: int = 64  # How many timesteps to process

    # Output heads
    use_value_head: bool = True  # For RL critic

    # Route generation settings
    route_input_dim: int = 10  # origin lat/lon/elev/rwy + dest lat/lon/elev/rwy + cruise_alt + distance
    max_waypoints: int = 8     # Maximum waypoints in a route
    waypoint_dim: int = 6      # lat, lon, alt, speed, heading, phase_type


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TextEncoder(nn.Module):
    """Encodes text input to d_model dimensional vectors."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        self.scale = math.sqrt(config.d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode token ids to vectors.

        Args:
            tokens: [batch, seq_len] token ids
        Returns:
            [batch, seq_len, d_model]
        """
        x = self.embedding(tokens) * self.scale
        return self.pos_encoder(x)


class StateEncoder(nn.Module):
    """Encodes flight state vectors to d_model dimensional vectors."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear = nn.Linear(config.state_dim, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, config.trajectory_seq_len, config.dropout)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Encode state vectors to d_model space.

        Args:
            states: [batch, seq_len, state_dim] or [batch, state_dim]
        Returns:
            [batch, seq_len, d_model] or [batch, 1, d_model]
        """
        if states.dim() == 2:
            states = states.unsqueeze(1)  # Add sequence dimension

        x = self.linear(states)
        x = self.norm(x)
        return self.pos_encoder(x)


class CommandHead(nn.Module):
    """Output head for NLP command parsing.

    Outputs:
    - command_type: Classification over command types
    - command_values: Regression for numeric values (heading, altitude, etc.)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Command type classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_command_types)
        )

        # Value regressors (one per value type)
        # heading_deg, altitude_ft, airspeed_kts, vs_fpm, turn_degrees, flap_position
        self.value_heads = nn.ModuleDict({
            'heading': nn.Linear(config.d_model, 1),
            'altitude': nn.Linear(config.d_model, 1),
            'airspeed': nn.Linear(config.d_model, 1),
            'vs': nn.Linear(config.d_model, 1),
            'turn': nn.Linear(config.d_model, 1),
            'flaps': nn.Linear(config.d_model, 1),
        })

        # Confidence estimator
        self.confidence = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, d_model] pooled encoder output
        Returns:
            dict with command_logits, values, confidence
        """
        return {
            'command_logits': self.classifier(x),
            'heading': torch.sigmoid(self.value_heads['heading'](x)) * 360,  # 0-360
            'altitude': torch.relu(self.value_heads['altitude'](x)) * 100,   # Scale to ft
            'airspeed': torch.sigmoid(self.value_heads['airspeed'](x)) * 100 + 40,  # 40-140 kts
            'vs': torch.tanh(self.value_heads['vs'](x)) * 1500,  # -1500 to +1500 fpm
            'turn': torch.tanh(self.value_heads['turn'](x)) * 180,  # -180 to +180 deg
            'flaps': torch.sigmoid(self.value_heads['flaps'](x)),  # 0-1
            'confidence': self.confidence(x),
        }


class ActionHead(nn.Module):
    """Output head for flight control actions.

    Outputs continuous control values:
    [throttle, elevator, aileron, rudder, flaps, spoilers, brakes]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.action_dim)
        )

        # Action bounds (for output normalization)
        # throttle: 0-1, surfaces: -1 to 1, flaps/spoilers/brakes: 0-1
        self.register_buffer('action_low', torch.tensor([0, -1, -1, -1, 0, 0, 0], dtype=torch.float32))
        self.register_buffer('action_high', torch.tensor([1, 1, 1, 1, 1, 1, 1], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model] or [batch, d_model]
        Returns:
            [batch, seq_len, action_dim] or [batch, action_dim] bounded actions
        """
        raw = self.mlp(x)

        # Apply appropriate activation for each action
        # Throttle, flaps, spoilers, brakes: sigmoid (0-1)
        # Elevator, aileron, rudder: tanh (-1 to 1)
        actions = torch.zeros_like(raw)
        actions[..., 0] = torch.sigmoid(raw[..., 0])      # throttle
        actions[..., 1] = torch.tanh(raw[..., 1])         # elevator
        actions[..., 2] = torch.tanh(raw[..., 2])         # aileron
        actions[..., 3] = torch.tanh(raw[..., 3])         # rudder
        actions[..., 4] = torch.sigmoid(raw[..., 4])      # flaps
        actions[..., 5] = torch.sigmoid(raw[..., 5])      # spoilers
        actions[..., 6] = torch.sigmoid(raw[..., 6])      # brakes

        return actions


class ValueHead(nn.Module):
    """Value function head for RL (critic network)."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, d_model] or [batch, seq_len, d_model]
        Returns:
            [batch, 1] or [batch, seq_len, 1] value estimates
        """
        return self.mlp(x)


class RouteEncoder(nn.Module):
    """Encodes route specification (origin, destination, constraints) to d_model vectors.

    Input format (normalized):
        - origin_lat, origin_lon (normalized to [-1, 1])
        - origin_elev_ft (normalized by /10000)
        - origin_runway_deg (normalized by /360)
        - dest_lat, dest_lon (normalized)
        - dest_elev_ft (normalized)
        - dest_runway_deg (normalized)
        - cruise_alt_ft (normalized by /10000)
        - distance_nm (normalized by /200)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Project route input to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(config.route_input_dim, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model)
        )

        # Learnable query tokens for waypoint generation
        self.waypoint_queries = nn.Parameter(
            torch.randn(config.max_waypoints, config.d_model) * 0.02
        )

    def forward(self, route_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode route specification.

        Args:
            route_spec: [batch, route_input_dim] normalized route parameters

        Returns:
            route_embed: [batch, 1, d_model] route embedding
            waypoint_queries: [batch, max_waypoints, d_model] query tokens
        """
        batch_size = route_spec.size(0)

        # Project route spec
        route_embed = self.input_proj(route_spec)  # [batch, d_model]
        route_embed = route_embed.unsqueeze(1)  # [batch, 1, d_model]

        # Expand waypoint queries for batch
        queries = self.waypoint_queries.unsqueeze(0).expand(batch_size, -1, -1)

        return route_embed, queries


class RouteGenerationHead(nn.Module):
    """Output head for route/controller parameter generation.

    Generates a sequence of waypoints that define a flight plan:
        - Waypoint positions (lat, lon, alt)
        - Target speeds at each waypoint
        - Headings
        - Phase type (takeoff, climb, cruise, descent, approach, landing)

    Also generates controller parameters:
        - Turn point distance
        - Glideslope angle
        - Pattern altitude
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Waypoint decoder
        self.waypoint_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.waypoint_dim)
        )

        # Controller parameters head (global route parameters)
        self.controller_params = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 8)  # tp_dist, gs_angle, pattern_alt, etc.
        )

        # Waypoint validity mask (learned end-of-sequence)
        self.validity_head = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Phase classifier for each waypoint (6 phases)
        self.phase_classifier = nn.Linear(config.d_model, 6)

    def forward(self, waypoint_features: torch.Tensor, route_embed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate route waypoints and controller parameters.

        Args:
            waypoint_features: [batch, max_waypoints, d_model] decoded waypoint features
            route_embed: [batch, d_model] pooled route embedding

        Returns:
            Dict with:
                - waypoints: [batch, max_waypoints, waypoint_dim] waypoint parameters
                - validity: [batch, max_waypoints, 1] waypoint validity mask
                - phase_logits: [batch, max_waypoints, 6] phase classification
                - controller_params: [batch, 8] global controller parameters
        """
        # Generate waypoints
        waypoints_raw = self.waypoint_mlp(waypoint_features)

        # Apply appropriate activations
        waypoints = torch.zeros_like(waypoints_raw)
        waypoints[..., 0] = torch.tanh(waypoints_raw[..., 0])  # lat delta (normalized)
        waypoints[..., 1] = torch.tanh(waypoints_raw[..., 1])  # lon delta (normalized)
        waypoints[..., 2] = torch.sigmoid(waypoints_raw[..., 2])  # alt (0-1, scale to ft)
        waypoints[..., 3] = torch.sigmoid(waypoints_raw[..., 3])  # speed (0-1, scale to kts)
        waypoints[..., 4] = torch.tanh(waypoints_raw[..., 4]) * math.pi  # heading (-π to π)
        waypoints[..., 5] = torch.sigmoid(waypoints_raw[..., 5])  # phase progress (0-1)

        # Validity mask (which waypoints are active)
        validity = self.validity_head(waypoint_features)

        # Phase classification
        phase_logits = self.phase_classifier(waypoint_features)

        # Global controller parameters (normalized 0-1 for training)
        # Denormalize using denormalize_controller_params() at inference time
        ctrl_raw = self.controller_params(route_embed)
        controller_params = torch.sigmoid(ctrl_raw)  # All outputs in [0, 1]

        return {
            'waypoints': waypoints,
            'validity': validity,
            'phase_logits': phase_logits,
            'controller_params': controller_params
        }


def denormalize_controller_params(ctrl_normalized: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Denormalize controller parameters from [0,1] to actual values.

    Args:
        ctrl_normalized: [batch, 8] tensor of normalized controller params

    Returns:
        Dictionary with named controller parameters in physical units
    """
    return {
        'tp_distance_nm': ctrl_normalized[..., 0] * 20.0,  # 0-20 nm
        'glideslope_deg': ctrl_normalized[..., 1] * 5.0 + 2.0,  # 2-7 deg
        'pattern_alt_ft': ctrl_normalized[..., 2] * 2000.0 + 500.0,  # 500-2500 ft
        'cruise_alt_ft': ctrl_normalized[..., 3] * 10000.0 + 2000.0,  # 2000-12000 ft
        'v_approach_kts': ctrl_normalized[..., 4] * 50.0 + 60.0,  # 60-110 kts
        'v_cruise_kts': ctrl_normalized[..., 5] * 60.0 + 80.0,  # 80-140 kts
        'turn_bank_deg': ctrl_normalized[..., 6] * 30.0 + 15.0,  # 15-45 deg
        'use_triangle_pattern': ctrl_normalized[..., 7],  # 0-1 (boolean)
    }


class UnifiedTransformer(nn.Module):
    """
    Unified transformer for flight control and NLP commands.

    Supports multiple task types:
    - NLP_COMMAND: Text input -> Command classification + value regression
    - TRAJECTORY: State sequence -> Action sequence (autoregressive)
    - IMITATION: Single state -> Expert action (behavioral cloning)
    - REINFORCEMENT: State -> Action + Value (PPO/SAC)
    - ROUTE_GENERATION: Route spec -> Controller parameters + waypoints
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()

        # Input encoders
        self.text_encoder = TextEncoder(self.config)
        self.state_encoder = StateEncoder(self.config)
        self.route_encoder = RouteEncoder(self.config)

        # Task type embedding (learned token to indicate task)
        self.task_embedding = nn.Embedding(len(TaskType), self.config.d_model)

        # Shared transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_encoder_layers)

        # Transformer decoder (for sequence-to-sequence tasks)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.config.num_decoder_layers)

        # Output heads
        self.command_head = CommandHead(self.config)
        self.action_head = ActionHead(self.config)
        self.route_head = RouteGenerationHead(self.config)
        if self.config.use_value_head:
            self.value_head = ValueHead(self.config)

        # Pooling for classification tasks
        self.pool = nn.AdaptiveAvgPool1d(1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        task: TaskType,
        text_tokens: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        target_actions: Optional[torch.Tensor] = None,
        route_spec: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with task-specific routing.

        Args:
            task: Which task to perform
            text_tokens: [batch, seq_len] for NLP tasks
            states: [batch, seq_len, state_dim] for trajectory tasks
            target_actions: [batch, seq_len, action_dim] for teacher forcing
            route_spec: [batch, route_input_dim] for route generation
            src_mask: Source attention mask
            tgt_mask: Target attention mask (causal)

        Returns:
            Dict with task-specific outputs
        """
        # Get task embedding
        task_idx = torch.tensor([list(TaskType).index(task)], device=self._get_device())
        task_emb = self.task_embedding(task_idx).unsqueeze(0)  # [1, 1, d_model]

        if task == TaskType.NLP_COMMAND:
            return self._forward_nlp(text_tokens, task_emb, src_mask)

        elif task == TaskType.TRAJECTORY:
            return self._forward_trajectory(states, target_actions, task_emb, src_mask, tgt_mask)

        elif task == TaskType.IMITATION:
            return self._forward_imitation(states, task_emb)

        elif task == TaskType.REINFORCEMENT:
            return self._forward_rl(states, task_emb)

        elif task == TaskType.ROUTE_GENERATION:
            return self._forward_route_generation(route_spec, task_emb)

        else:
            raise ValueError(f"Unknown task type: {task}")

    def _get_device(self) -> torch.device:
        return next(self.parameters()).device

    def _forward_nlp(
        self,
        tokens: torch.Tensor,
        task_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """NLP command parsing forward pass."""
        batch_size = tokens.size(0)

        # Encode text
        x = self.text_encoder(tokens)  # [batch, seq_len, d_model]

        # Prepend task embedding
        task_emb = task_emb.expand(batch_size, -1, -1)
        x = torch.cat([task_emb, x], dim=1)  # [batch, 1+seq_len, d_model]

        # Transformer encode
        encoded = self.encoder(x, src_key_padding_mask=mask)

        # Pool to single vector (use task token output)
        pooled = encoded[:, 0, :]  # [batch, d_model]

        # Command head
        return self.command_head(pooled)

    def _forward_trajectory(
        self,
        states: torch.Tensor,
        target_actions: Optional[torch.Tensor],
        task_emb: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Trajectory prediction forward pass (sequence-to-sequence)."""
        batch_size = states.size(0)

        # Encode states
        x = self.state_encoder(states)  # [batch, seq_len, d_model]

        # Prepend task embedding
        task_emb = task_emb.expand(batch_size, -1, -1)
        x = torch.cat([task_emb, x], dim=1)

        # Encode
        memory = self.encoder(x, src_key_padding_mask=src_mask)

        if target_actions is not None:
            # Teacher forcing: use target actions as decoder input
            # Shift right and embed
            tgt = self._embed_actions(target_actions)

            # Generate causal mask if not provided
            if tgt_mask is None:
                tgt_mask = self._generate_causal_mask(tgt.size(1))

            # Decode
            decoded = self.decoder(tgt, memory, tgt_mask=tgt_mask)

            # Action head
            actions = self.action_head(decoded)
        else:
            # Autoregressive generation
            actions = self._generate_actions(memory, states.size(1))

        return {'actions': actions}

    def _forward_imitation(
        self,
        states: torch.Tensor,
        task_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Behavioral cloning forward pass (single step)."""
        batch_size = states.size(0)

        # Ensure states have sequence dimension
        if states.dim() == 2:
            states = states.unsqueeze(1)

        # Encode
        x = self.state_encoder(states)

        # Add task embedding
        task_emb = task_emb.expand(batch_size, -1, -1)
        x = torch.cat([task_emb, x], dim=1)

        # Encode
        encoded = self.encoder(x)

        # Use task token for prediction
        hidden = encoded[:, 0, :]

        # Get action
        actions = self.action_head(hidden)

        return {'actions': actions}

    def _forward_rl(
        self,
        states: torch.Tensor,
        task_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """RL forward pass (actor-critic)."""
        batch_size = states.size(0)

        # Ensure states have sequence dimension
        if states.dim() == 2:
            states = states.unsqueeze(1)

        # Encode
        x = self.state_encoder(states)

        # Add task embedding
        task_emb = task_emb.expand(batch_size, -1, -1)
        x = torch.cat([task_emb, x], dim=1)

        # Encode
        encoded = self.encoder(x)

        # Use task token
        hidden = encoded[:, 0, :]

        # Get action and value
        actions = self.action_head(hidden)
        value = self.value_head(hidden) if self.config.use_value_head else None

        return {'actions': actions, 'value': value}

    def _forward_route_generation(
        self,
        route_spec: torch.Tensor,
        task_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Route generation forward pass.

        Given origin/destination and constraints, generates:
        - Waypoint sequence (lat, lon, alt, speed, heading, phase)
        - Controller parameters (tp_distance, glideslope, pattern_alt, etc.)

        Uses DETR-style learned queries for waypoint generation.

        Args:
            route_spec: [batch, route_input_dim] normalized route specification
            task_emb: [1, 1, d_model] task embedding

        Returns:
            Dict with waypoints, validity, phase_logits, controller_params
        """
        batch_size = route_spec.size(0)

        # Encode route specification and get waypoint queries
        route_embed, waypoint_queries = self.route_encoder(route_spec)
        # route_embed: [batch, 1, d_model]
        # waypoint_queries: [batch, max_waypoints, d_model]

        # Expand task embedding
        task_emb = task_emb.expand(batch_size, -1, -1)

        # Combine task embedding and route embedding as encoder input
        encoder_input = torch.cat([task_emb, route_embed], dim=1)  # [batch, 2, d_model]

        # Encode route context
        memory = self.encoder(encoder_input)  # [batch, 2, d_model]

        # Use waypoint queries as decoder input (DETR-style object queries)
        # Each query learns to predict one waypoint in the sequence
        waypoint_features = self.decoder(
            waypoint_queries,  # [batch, max_waypoints, d_model]
            memory             # [batch, 2, d_model]
        )  # [batch, max_waypoints, d_model]

        # Pool route embedding for controller params (use encoder output)
        pooled_route = memory[:, 0, :]  # Use task token output [batch, d_model]

        # Generate waypoints and controller parameters
        outputs = self.route_head(waypoint_features, pooled_route)

        return outputs

    def _embed_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Embed action vectors into d_model space."""
        # Simple linear projection
        if not hasattr(self, 'action_embedder'):
            self.action_embedder = nn.Linear(self.config.action_dim, self.config.d_model).to(actions.device)
        return self.action_embedder(actions)

    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.to(self._get_device())

    @torch.no_grad()
    def _generate_actions(self, memory: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Autoregressive action generation."""
        batch_size = memory.size(0)
        device = memory.device

        # Start with zeros
        actions = torch.zeros(batch_size, 1, self.config.action_dim, device=device)

        for _ in range(seq_len):
            # Embed current actions
            tgt = self._embed_actions(actions)

            # Causal mask
            tgt_mask = self._generate_causal_mask(tgt.size(1))

            # Decode
            decoded = self.decoder(tgt, memory, tgt_mask=tgt_mask)

            # Get next action
            next_action = self.action_head(decoded[:, -1:, :])

            # Append
            actions = torch.cat([actions, next_action], dim=1)

        # Remove initial zero token
        return actions[:, 1:, :]


# ============================================================================
# Training Utilities
# ============================================================================

class CommandLoss(nn.Module):
    """Loss function for NLP command parsing."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.config = config

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            outputs: Model outputs from command_head
            targets: Dict with 'command_type', 'heading', 'altitude', etc.
        """
        # Classification loss
        cls_loss = self.ce_loss(outputs['command_logits'], targets['command_type'])

        # Regression losses (only for relevant command types)
        reg_loss = 0.0
        if 'heading' in targets:
            reg_loss += self.mse_loss(outputs['heading'].squeeze(), targets['heading'])
        if 'altitude' in targets:
            reg_loss += self.mse_loss(outputs['altitude'].squeeze(), targets['altitude'])

        return cls_loss + 0.1 * reg_loss


class TrajectoryLoss(nn.Module):
    """Loss function for trajectory prediction."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor
    ) -> torch.Tensor:
        return self.mse(pred_actions, target_actions)


class ImitationLoss(nn.Module):
    """Behavioral cloning loss."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_actions: torch.Tensor,
        expert_actions: torch.Tensor
    ) -> torch.Tensor:
        return self.mse(pred_actions, expert_actions)


class RouteGenerationLoss(nn.Module):
    """Loss function for route generation task.

    Combines:
    - Waypoint position/parameter regression loss (MSE)
    - Waypoint validity classification loss (BCE)
    - Phase classification loss (CE)
    - Controller parameter regression loss (MSE)
    """

    def __init__(self, waypoint_weight: float = 1.0, validity_weight: float = 0.5,
                 phase_weight: float = 0.3, controller_weight: float = 1.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCELoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')

        self.waypoint_weight = waypoint_weight
        self.validity_weight = validity_weight
        self.phase_weight = phase_weight
        self.controller_weight = controller_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Model outputs from route_head
                - waypoints: [batch, max_waypoints, waypoint_dim]
                - validity: [batch, max_waypoints, 1]
                - phase_logits: [batch, max_waypoints, 6]
                - controller_params: [batch, 8]
            targets: Ground truth
                - waypoints: [batch, max_waypoints, waypoint_dim]
                - validity: [batch, max_waypoints, 1]
                - phase: [batch, max_waypoints] (class indices)
                - controller_params: [batch, 8]

        Returns:
            Dict with total loss and component losses
        """
        # Validity mask for waypoints (only compute loss on valid waypoints)
        validity_mask = targets['validity'].squeeze(-1)  # [batch, max_waypoints]

        # Waypoint loss (masked by validity)
        waypoint_loss = self.mse(outputs['waypoints'], targets['waypoints'])
        waypoint_loss = waypoint_loss.mean(dim=-1)  # [batch, max_waypoints]
        waypoint_loss = (waypoint_loss * validity_mask).sum() / (validity_mask.sum() + 1e-8)

        # Validity loss
        validity_loss = self.bce(outputs['validity'], targets['validity'])
        validity_loss = validity_loss.mean()

        # Phase classification loss (masked)
        phase_logits = outputs['phase_logits'].reshape(-1, 6)  # [batch*max_waypoints, 6]
        phase_targets = targets['phase'].reshape(-1)  # [batch*max_waypoints]
        phase_loss = self.ce(phase_logits, phase_targets)
        phase_loss = phase_loss.reshape(validity_mask.shape)
        phase_loss = (phase_loss * validity_mask).sum() / (validity_mask.sum() + 1e-8)

        # Controller params loss
        controller_loss = self.mse(outputs['controller_params'], targets['controller_params'])
        controller_loss = controller_loss.mean()

        # Total weighted loss
        total_loss = (
            self.waypoint_weight * waypoint_loss +
            self.validity_weight * validity_loss +
            self.phase_weight * phase_loss +
            self.controller_weight * controller_loss
        )

        return {
            'loss': total_loss,
            'waypoint_loss': waypoint_loss,
            'validity_loss': validity_loss,
            'phase_loss': phase_loss,
            'controller_loss': controller_loss
        }


# ============================================================================
# Simple Tokenizer (for NLP tasks)
# ============================================================================

class SimpleTokenizer:
    """Basic tokenizer for flight commands."""

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.next_idx = 4

        # Pre-populate with common flight terms
        flight_vocab = [
            'turn', 'heading', 'climb', 'descend', 'altitude', 'feet', 'ft',
            'knots', 'kts', 'speed', 'airspeed', 'left', 'right', 'degrees',
            'to', 'set', 'maintain', 'hold', 'go', 'around', 'flaps', 'up',
            'down', 'full', 'vertical', 'rate', 'fpm',
        ]
        for word in flight_vocab:
            self._add_word(word)

        # Add numbers 0-999
        for i in range(1000):
            self._add_word(str(i))

    def _add_word(self, word: str):
        if word not in self.word2idx and self.next_idx < self.vocab_size:
            self.word2idx[word] = self.next_idx
            self.idx2word[self.next_idx] = word
            self.next_idx += 1

    def encode(self, text: str, max_len: int = 128) -> torch.Tensor:
        """Tokenize text to tensor."""
        words = text.lower().split()
        tokens = [self.word2idx.get('<sos>')]

        for word in words:
            idx = self.word2idx.get(word, self.word2idx['<unk>'])
            tokens.append(idx)

        tokens.append(self.word2idx['<eos>'])

        # Pad or truncate
        if len(tokens) < max_len:
            tokens += [self.word2idx['<pad>']] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]

        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        """Decode token tensor to text."""
        words = []
        for idx in tokens.tolist():
            if idx == self.word2idx['<eos>']:
                break
            if idx not in [self.word2idx['<pad>'], self.word2idx['<sos>']]:
                words.append(self.idx2word.get(idx, '<unk>'))
        return ' '.join(words)


# ============================================================================
# Example Usage
# ============================================================================

def demo():
    """Demonstrate the unified transformer."""
    print("=" * 60)
    print("AIDA Unified Transformer Demo")
    print("=" * 60)

    # Create model
    config = ModelConfig(
        d_model=128,  # Smaller for demo
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=2,
    )
    model = UnifiedTransformer(config)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Demo 1: NLP Command
    print("\n--- NLP Command Parsing ---")
    tokenizer = SimpleTokenizer()
    text = "climb to 5000 feet"
    tokens = tokenizer.encode(text).unsqueeze(0)  # [1, seq_len]

    output = model(TaskType.NLP_COMMAND, text_tokens=tokens)
    print(f"Input: '{text}'")
    print(f"Command logits shape: {output['command_logits'].shape}")
    print(f"Predicted heading: {output['heading'].item():.1f}°")
    print(f"Predicted altitude: {output['altitude'].item():.0f} ft")
    print(f"Confidence: {output['confidence'].item():.2f}")

    # Demo 2: Imitation Learning
    print("\n--- Imitation Learning (BC) ---")
    state = torch.randn(1, config.state_dim)  # Random state
    output = model(TaskType.IMITATION, states=state)
    print(f"Input state shape: {state.shape}")
    print(f"Output action shape: {output['actions'].shape}")
    print(f"Actions: {output['actions'].squeeze().tolist()}")

    # Demo 3: Trajectory Prediction
    print("\n--- Trajectory Prediction ---")
    states = torch.randn(1, 16, config.state_dim)  # Sequence of states
    target_actions = torch.randn(1, 16, config.action_dim)
    output = model(TaskType.TRAJECTORY, states=states, target_actions=target_actions)
    print(f"Input states shape: {states.shape}")
    print(f"Output actions shape: {output['actions'].shape}")

    # Demo 4: RL (Actor-Critic)
    print("\n--- Reinforcement Learning ---")
    state = torch.randn(1, config.state_dim)
    output = model(TaskType.REINFORCEMENT, states=state)
    print(f"Actions shape: {output['actions'].shape}")
    print(f"Value: {output['value'].item():.3f}")

    # Demo 5: Route Generation
    print("\n--- Route Generation ---")
    # Create a sample route spec: KHUT -> KAAO (normalized)
    # Format: origin_lat, origin_lon, origin_elev, origin_rwy,
    #         dest_lat, dest_lon, dest_elev, dest_rwy, cruise_alt, distance
    route_spec = torch.tensor([[
        29.26 / 90,    # origin_lat (KHUT normalized)
        -95.66 / 180,  # origin_lon
        45 / 10000,    # origin_elev_ft
        160 / 360,     # origin_runway_deg
        29.42 / 90,    # dest_lat (KAAO)
        -96.19 / 180,  # dest_lon
        297 / 10000,   # dest_elev_ft
        170 / 360,     # dest_runway_deg
        3500 / 10000,  # cruise_alt_ft
        36 / 200,      # distance_nm
    ]])
    output = model(TaskType.ROUTE_GENERATION, route_spec=route_spec)
    print(f"Route spec shape: {route_spec.shape}")
    print(f"Waypoints shape: {output['waypoints'].shape}")
    print(f"Validity shape: {output['validity'].shape}")
    print(f"Phase logits shape: {output['phase_logits'].shape}")
    print(f"Controller params shape: {output['controller_params'].shape}")
    print(f"\nController parameters (normalized 0-1):")
    ctrl = output['controller_params'].squeeze()
    print(f"  Raw values: {ctrl.tolist()}")

    # Denormalize to physical units
    ctrl_denorm = denormalize_controller_params(output['controller_params'])
    print(f"\nController parameters (physical units):")
    print(f"  Turn point distance: {ctrl_denorm['tp_distance_nm'].item():.1f} nm")
    print(f"  Glideslope angle: {ctrl_denorm['glideslope_deg'].item():.1f}°")
    print(f"  Pattern altitude: {ctrl_denorm['pattern_alt_ft'].item():.0f} ft")
    print(f"  Cruise altitude: {ctrl_denorm['cruise_alt_ft'].item():.0f} ft")
    print(f"  Approach speed: {ctrl_denorm['v_approach_kts'].item():.0f} kts")
    print(f"  Cruise speed: {ctrl_denorm['v_cruise_kts'].item():.0f} kts")
    print(f"  Turn bank angle: {ctrl_denorm['turn_bank_deg'].item():.0f}°")
    print(f"  Triangle pattern: {ctrl_denorm['use_triangle_pattern'].item():.2f}")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    demo()
