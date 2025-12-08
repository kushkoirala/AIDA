import numpy as np

# Simple SoC + internal resistance battery model.
class Battery:
    def __init__(self, capacity_ah: float = 1.5, nominal_voltage: float = 12.0, r_internal: float = 0.05):
        self.capacity_ah = capacity_ah
        self.nominal_voltage = nominal_voltage
        self.r_internal = r_internal
        self.soc = 1.0
        self.voltage = nominal_voltage

    def step(self, current_a: float, dt: float):
        # Coulomb counting for SoC
        delta_ah = current_a * dt / 3600.0
        self.soc = np.clip(self.soc - delta_ah / self.capacity_ah, 0.0, 1.0)
        # Open-circuit voltage as simple linear function of SoC
        vocv = self.nominal_voltage * (0.9 + 0.1 * self.soc)
        self.voltage = vocv - current_a * self.r_internal
        return self.voltage
