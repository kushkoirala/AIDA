from dataclasses import dataclass, field
import numpy as np

@dataclass
class VehicleState:
    # Position in world frame (meters)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    # Linear velocity in world frame (m/s)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    # Orientation quaternion (w, x, y, z)
    orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    # Body angular rates p, q, r (rad/s)
    body_rates: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    # Control surface deflections (rad): [elevator, aileron, rudder]
    surfaces: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    # Battery state of charge [0,1]
    soc: float = 1.0
    # Battery bus voltage (V)
    voltage: float = 12.0
    # Estimated load factor (g)
    load_factor: float = 1.0

    def as_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                self.position,
                self.velocity,
                self.orientation,
                self.body_rates,
                self.surfaces,
                np.array([self.soc, self.voltage, self.load_factor], dtype=np.float32),
            ]
        )
