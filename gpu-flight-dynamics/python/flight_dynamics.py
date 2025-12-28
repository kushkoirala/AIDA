"""
GPU-Accelerated Flight Dynamics Simulator - Python Interface

This module provides a Pythonic interface to the CUDA flight dynamics
library, with support for NumPy, CuPy, and PyTorch tensor interoperability.

Author: Kushal Koirala
Date: 2025

NVIDIA Stack: CUDA, CuPy, PyTorch

Example Usage:
    from flight_dynamics import FlightSimulator, AircraftParams
    
    # Create simulator with 1000 parallel instances
    sim = FlightSimulator(n_instances=1000, use_gpu=True)
    
    # Run simulation
    for _ in range(1000):
        sim.set_controls(controls)
        sim.step()
    
    # Get states as PyTorch tensor for RL
    states = sim.get_states_tensor()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
from enum import IntEnum
import ctypes
import os

# Try to import GPU libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class StateIndex(IntEnum):
    """State vector indices"""
    X = 0       # Position North [m]
    Y = 1       # Position East [m]
    Z = 2       # Position Down [m]
    U = 3       # Body velocity X [m/s]
    V = 4       # Body velocity Y [m/s]
    W = 5       # Body velocity Z [m/s]
    PHI = 6     # Roll [rad]
    THETA = 7   # Pitch [rad]
    PSI = 8     # Yaw [rad]
    P = 9       # Roll rate [rad/s]
    Q = 10      # Pitch rate [rad/s]
    R = 11      # Yaw rate [rad/s]


class ControlIndex(IntEnum):
    """Control vector indices"""
    THROTTLE = 0    # 0 to 1
    AILERON = 1     # -1 to 1
    ELEVATOR = 2    # -1 to 1
    RUDDER = 3      # -1 to 1


# Dimensions
STATE_DIM = 12
CONTROL_DIM = 4


@dataclass
class MassProperties:
    """Aircraft mass and inertia properties"""
    mass: float = 1043.0      # kg
    Ixx: float = 1285.0       # kg·m²
    Iyy: float = 1825.0       # kg·m²
    Izz: float = 2667.0       # kg·m²
    Ixz: float = 0.0          # kg·m²


@dataclass
class Geometry:
    """Wing and control surface geometry"""
    S: float = 16.17          # Wing area [m²]
    b: float = 10.92          # Wingspan [m]
    c: float = 1.49           # Mean chord [m]
    e: float = 0.8            # Oswald efficiency
    
    @property
    def AR(self) -> float:
        """Aspect ratio"""
        return self.b ** 2 / self.S


@dataclass
class LongitudinalDerivatives:
    """Longitudinal aerodynamic stability derivatives"""
    CL0: float = 0.307
    CLa: float = 4.41         # 1/rad
    CLq: float = 3.9
    CLde: float = 0.43
    CLmax: float = 1.4
    CLmin: float = -0.5
    
    CD0: float = 0.027
    K: float = 0.054          # Induced drag factor
    CDa: float = 0.0
    
    Cm0: float = 0.04
    Cma: float = -0.613       # 1/rad (stable: negative)
    Cmq: float = -12.4
    Cmde: float = -1.122


@dataclass
class LateralDerivatives:
    """Lateral-directional stability derivatives"""
    CYb: float = -0.393
    CYp: float = -0.075
    CYr: float = 0.214
    CYda: float = 0.0
    CYdr: float = 0.187
    
    Clb: float = -0.0923      # Dihedral effect
    Clp: float = -0.484       # Roll damping
    Clr: float = 0.0798
    Clda: float = 0.229       # Aileron effectiveness
    Cldr: float = 0.0147
    
    Cnb: float = 0.0587       # Directional stability
    Cnp: float = -0.0278
    Cnr: float = -0.0937      # Yaw damping
    Cnda: float = -0.0216     # Adverse yaw
    Cndr: float = -0.0645


@dataclass
class PropulsionParams:
    """Propulsion system parameters"""
    thrust_max: float = 2500.0    # N
    thrust_min: float = 50.0      # N
    tau: float = 0.5              # Engine time constant [s]


@dataclass
class AircraftParams:
    """Complete aircraft parameters"""
    mass: MassProperties = field(default_factory=MassProperties)
    geom: Geometry = field(default_factory=Geometry)
    longi: LongitudinalDerivatives = field(default_factory=LongitudinalDerivatives)
    latdi: LateralDerivatives = field(default_factory=LateralDerivatives)
    prop: PropulsionParams = field(default_factory=PropulsionParams)


class FlightSimulator:
    """
    GPU-accelerated 6-DOF flight dynamics simulator.
    
    Supports parallel simulation of thousands of aircraft instances
    for reinforcement learning and Monte Carlo analysis.
    
    Args:
        n_instances: Number of parallel simulation instances
        params: Aircraft parameters (uses defaults if None)
        dt: Simulation time step [s]
        use_gpu: Use GPU acceleration if available
        integration_method: 0=Euler, 1=RK2, 2=RK4
    """
    
    def __init__(
        self,
        n_instances: int = 1,
        params: Optional[AircraftParams] = None,
        dt: float = 0.01,
        use_gpu: bool = True,
        integration_method: int = 2
    ):
        self.n_instances = n_instances
        self.params = params or AircraftParams()
        self.dt = dt
        self.integration_method = integration_method
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        
        # Select array backend
        self.xp = cp if self.use_gpu else np
        
        # Initialize state and control arrays
        self.states = self.xp.zeros((n_instances, STATE_DIM), dtype=np.float32)
        self.controls = self.xp.zeros((n_instances, CONTROL_DIM), dtype=np.float32)
        
        # Simulation time
        self.time = 0.0
        
        # Set default trim state
        self._init_trim()
        
        backend = "GPU (CuPy)" if self.use_gpu else "CPU (NumPy)"
        print(f"FlightSimulator: {n_instances} instances on {backend}")
    
    def _init_trim(self):
        """Initialize to trim condition"""
        # Altitude: 1000m
        self.states[:, StateIndex.Z] = -1000.0
        
        # Forward velocity: 50 m/s
        self.states[:, StateIndex.U] = 50.0
        
        # Trim throttle
        self.controls[:, ControlIndex.THROTTLE] = 0.4
    
    def reset(
        self,
        initial_state: Optional[np.ndarray] = None,
        initial_control: Optional[np.ndarray] = None
    ):
        """
        Reset simulation to initial conditions.
        
        Args:
            initial_state: Initial state [STATE_DIM] or [n_instances, STATE_DIM]
            initial_control: Initial control [CONTROL_DIM] or [n_instances, CONTROL_DIM]
        """
        self.time = 0.0
        
        if initial_state is not None:
            state = self.xp.asarray(initial_state, dtype=np.float32)
            if state.ndim == 1:
                self.states[:] = state
            else:
                self.states[:] = state
        else:
            self.states[:] = 0.0
            self._init_trim()
        
        if initial_control is not None:
            ctrl = self.xp.asarray(initial_control, dtype=np.float32)
            if ctrl.ndim == 1:
                self.controls[:] = ctrl
            else:
                self.controls[:] = ctrl
        else:
            self.controls[:] = 0.0
            self.controls[:, ControlIndex.THROTTLE] = 0.4
    
    def set_controls(self, controls: Union[np.ndarray, 'cp.ndarray', 'torch.Tensor']):
        """
        Set control inputs for all instances.
        
        Args:
            controls: Control array [n_instances, CONTROL_DIM]
        """
        if TORCH_AVAILABLE and isinstance(controls, torch.Tensor):
            if controls.is_cuda and self.use_gpu:
                self.controls = cp.asarray(controls)
            else:
                self.controls = self.xp.asarray(controls.cpu().numpy(), dtype=np.float32)
        else:
            self.controls = self.xp.asarray(controls, dtype=np.float32)
    
    def step(self):
        """Advance simulation by one time step using selected integration method."""
        if self.integration_method == 0:
            self._step_euler()
        elif self.integration_method == 1:
            self._step_rk2()
        else:
            self._step_rk4()
        
        self.time += self.dt
    
    def step_n(self, n_steps: int):
        """Advance simulation by multiple time steps."""
        for _ in range(n_steps):
            self.step()
    
    def _compute_derivatives(self, states: 'np.ndarray') -> 'np.ndarray':
        """Compute state derivatives for all instances."""
        xp = self.xp
        
        # Extract state components
        z = states[:, StateIndex.Z]
        u = states[:, StateIndex.U]
        v = states[:, StateIndex.V]
        w = states[:, StateIndex.W]
        phi = states[:, StateIndex.PHI]
        theta = states[:, StateIndex.THETA]
        psi = states[:, StateIndex.PSI]
        p = states[:, StateIndex.P]
        q = states[:, StateIndex.Q]
        r = states[:, StateIndex.R]
        
        # Controls
        throttle = self.controls[:, ControlIndex.THROTTLE]
        da = self.controls[:, ControlIndex.AILERON]
        de = self.controls[:, ControlIndex.ELEVATOR]
        dr = self.controls[:, ControlIndex.RUDDER]
        
        # Atmosphere
        altitude = xp.maximum(0.0, -z)
        T = 288.15 - 0.0065 * altitude
        rho = 1.225 * xp.power(T / 288.15, 4.256)
        
        # Airspeed and angles
        V = xp.sqrt(u**2 + v**2 + w**2)
        V = xp.maximum(V, 0.1)
        alpha = xp.arctan2(w, u)
        beta = xp.arcsin(xp.clip(v / V, -1, 1))
        
        # Dynamic pressure
        qbar = 0.5 * rho * V**2
        
        # Non-dimensional rates
        phat = p * self.params.geom.b / (2 * V)
        qhat = q * self.params.geom.c / (2 * V)
        rhat = r * self.params.geom.b / (2 * V)
        
        # Aerodynamic coefficients
        longi = self.params.longi
        latdi = self.params.latdi
        
        CL = longi.CL0 + longi.CLa * alpha + longi.CLq * qhat + longi.CLde * de
        CD = longi.CD0 + longi.K * CL**2
        Cm = longi.Cm0 + longi.Cma * alpha + longi.Cmq * qhat + longi.Cmde * de
        
        CY = latdi.CYb * beta + latdi.CYp * phat + latdi.CYr * rhat + latdi.CYdr * dr
        Cl = latdi.Clb * beta + latdi.Clp * phat + latdi.Clr * rhat + latdi.Clda * da
        Cn = latdi.Cnb * beta + latdi.Cnp * phat + latdi.Cnr * rhat + latdi.Cndr * dr
        
        # Forces in body frame
        S = self.params.geom.S
        L_aero = qbar * S * CL
        D_aero = qbar * S * CD
        Y_aero = qbar * S * CY
        
        cos_a = xp.cos(alpha)
        sin_a = xp.sin(alpha)
        
        Fx = -D_aero * cos_a + L_aero * sin_a + throttle * self.params.prop.thrust_max
        Fy = Y_aero
        Fz = -D_aero * sin_a - L_aero * cos_a
        
        # Moments
        b = self.params.geom.b
        c = self.params.geom.c
        L_mom = qbar * S * b * Cl
        M_mom = qbar * S * c * Cm
        N_mom = qbar * S * b * Cn
        
        # Gravity in body frame
        g = 9.80665
        gx = -g * xp.sin(theta)
        gy = g * xp.cos(theta) * xp.sin(phi)
        gz = g * xp.cos(theta) * xp.cos(phi)
        
        # Translational acceleration
        mass = self.params.mass.mass
        u_dot = Fx / mass - q*w + r*v + gx
        v_dot = Fy / mass - r*u + p*w + gy
        w_dot = Fz / mass - p*v + q*u + gz
        
        # Angular acceleration
        Ixx = self.params.mass.Ixx
        Iyy = self.params.mass.Iyy
        Izz = self.params.mass.Izz
        
        p_dot = L_mom / Ixx
        q_dot = M_mom / Iyy
        r_dot = N_mom / Izz
        
        # Euler angle rates
        sin_phi = xp.sin(phi)
        cos_phi = xp.cos(phi)
        cos_theta = xp.cos(theta)
        tan_theta = xp.tan(theta)
        
        phi_dot = p + (q * sin_phi + r * cos_phi) * tan_theta
        theta_dot = q * cos_phi - r * sin_phi
        psi_dot = (q * sin_phi + r * cos_phi) / xp.maximum(xp.abs(cos_theta), 0.001)
        
        # Position rates (body to NED)
        sin_theta = xp.sin(theta)
        cos_psi = xp.cos(psi)
        sin_psi = xp.sin(psi)
        
        R11 = cos_theta * cos_psi
        R12 = sin_phi * sin_theta * cos_psi - cos_phi * sin_psi
        R13 = cos_phi * sin_theta * cos_psi + sin_phi * sin_psi
        R21 = cos_theta * sin_psi
        R22 = sin_phi * sin_theta * sin_psi + cos_phi * cos_psi
        R23 = cos_phi * sin_theta * sin_psi - sin_phi * cos_psi
        R31 = -sin_theta
        R32 = sin_phi * cos_theta
        R33 = cos_phi * cos_theta
        
        x_dot = R11*u + R12*v + R13*w
        y_dot = R21*u + R22*v + R23*w
        z_dot = R31*u + R32*v + R33*w
        
        # Pack derivatives
        state_dot = xp.zeros_like(states)
        state_dot[:, StateIndex.X] = x_dot
        state_dot[:, StateIndex.Y] = y_dot
        state_dot[:, StateIndex.Z] = z_dot
        state_dot[:, StateIndex.U] = u_dot
        state_dot[:, StateIndex.V] = v_dot
        state_dot[:, StateIndex.W] = w_dot
        state_dot[:, StateIndex.PHI] = phi_dot
        state_dot[:, StateIndex.THETA] = theta_dot
        state_dot[:, StateIndex.PSI] = psi_dot
        state_dot[:, StateIndex.P] = p_dot
        state_dot[:, StateIndex.Q] = q_dot
        state_dot[:, StateIndex.R] = r_dot
        
        return state_dot
    
    def _step_euler(self):
        """Euler integration step."""
        state_dot = self._compute_derivatives(self.states)
        self.states += self.dt * state_dot
        self._normalize_angles()
    
    def _step_rk2(self):
        """RK2 (Heun) integration step."""
        xp = self.xp
        
        k1 = self._compute_derivatives(self.states)
        k2 = self._compute_derivatives(self.states + self.dt * k1)
        
        self.states += 0.5 * self.dt * (k1 + k2)
        self._normalize_angles()
    
    def _step_rk4(self):
        """RK4 integration step."""
        xp = self.xp
        dt = self.dt
        
        k1 = self._compute_derivatives(self.states)
        k2 = self._compute_derivatives(self.states + 0.5 * dt * k1)
        k3 = self._compute_derivatives(self.states + 0.5 * dt * k2)
        k4 = self._compute_derivatives(self.states + dt * k3)
        
        self.states += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self._normalize_angles()
    
    def _normalize_angles(self):
        """Normalize Euler angles to [-pi, pi]."""
        xp = self.xp
        pi = xp.pi
        
        for idx in [StateIndex.PHI, StateIndex.THETA, StateIndex.PSI]:
            self.states[:, idx] = xp.mod(self.states[:, idx] + pi, 2*pi) - pi
    
    def get_states(self) -> np.ndarray:
        """Get current states as NumPy array."""
        if self.use_gpu:
            return cp.asnumpy(self.states)
        return self.states.copy()
    
    def get_states_tensor(self, device: str = 'cuda') -> 'torch.Tensor':
        """Get current states as PyTorch tensor."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        if self.use_gpu and device == 'cuda':
            # Direct GPU transfer via DLPack
            return torch.as_tensor(self.states, device='cuda')
        else:
            return torch.from_numpy(self.get_states())
    
    def get_altitude(self) -> np.ndarray:
        """Get altitude for all instances [m]."""
        states = self.get_states()
        return -states[:, StateIndex.Z]
    
    def get_airspeed(self) -> np.ndarray:
        """Get true airspeed for all instances [m/s]."""
        states = self.get_states()
        u = states[:, StateIndex.U]
        v = states[:, StateIndex.V]
        w = states[:, StateIndex.W]
        return np.sqrt(u**2 + v**2 + w**2)
    
    def get_euler_angles_deg(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get Euler angles in degrees."""
        states = self.get_states()
        phi = np.degrees(states[:, StateIndex.PHI])
        theta = np.degrees(states[:, StateIndex.THETA])
        psi = np.degrees(states[:, StateIndex.PSI])
        return phi, theta, psi


def benchmark(n_instances: int = 1000, n_steps: int = 1000):
    """
    Run GPU vs CPU benchmark.
    
    This demonstrates the performance advantage of GPU acceleration.
    """
    import time
    
    print(f"\n{'='*60}")
    print(f"Flight Dynamics Benchmark")
    print(f"Instances: {n_instances}, Steps: {n_steps}")
    print(f"{'='*60}\n")
    
    # CPU benchmark
    print("Running CPU benchmark...")
    sim_cpu = FlightSimulator(n_instances, use_gpu=False, integration_method=0)
    
    start = time.perf_counter()
    sim_cpu.step_n(n_steps)
    cpu_time = time.perf_counter() - start
    
    print(f"CPU Time: {cpu_time*1000:.2f} ms")
    
    # GPU benchmark
    if CUPY_AVAILABLE:
        print("\nRunning GPU benchmark...")
        sim_gpu = FlightSimulator(n_instances, use_gpu=True, integration_method=0)
        
        # Warm-up
        sim_gpu.step_n(10)
        if CUPY_AVAILABLE:
            cp.cuda.Stream.null.synchronize()
        
        sim_gpu.reset()
        
        start = time.perf_counter()
        sim_gpu.step_n(n_steps)
        if CUPY_AVAILABLE:
            cp.cuda.Stream.null.synchronize()
        gpu_time = time.perf_counter() - start
        
        print(f"GPU Time: {gpu_time*1000:.2f} ms")
        print(f"\n🚀 Speedup: {cpu_time/gpu_time:.1f}x")
    else:
        print("\nCuPy not available - GPU benchmark skipped")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Run demo
    print("Flight Dynamics Simulator - Demo\n")
    
    # Create simulator
    sim = FlightSimulator(n_instances=10)
    
    # Run for 5 seconds
    print("Running 5 second simulation...")
    for i in range(500):
        # Apply varying controls
        controls = np.zeros((10, CONTROL_DIM), dtype=np.float32)
        controls[:, ControlIndex.THROTTLE] = 0.4
        controls[:, ControlIndex.ELEVATOR] = 0.05 * np.sin(i * 0.02)
        sim.set_controls(controls)
        sim.step()
        
        if i % 100 == 0:
            alt = sim.get_altitude()
            airspeed = sim.get_airspeed()
            phi, theta, psi = sim.get_euler_angles_deg()
            print(f"  t={sim.time:.1f}s: alt={alt[0]:.1f}m, V={airspeed[0]:.1f}m/s, "
                  f"pitch={theta[0]:.1f}°")
    
    print("\nRunning benchmark...")
    benchmark(n_instances=1000, n_steps=1000)
