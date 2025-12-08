"""Environments exposing Gymnasium-compatible interfaces."""

from gymnasium.envs.registration import register


# Register the flight environment with a stable Gymnasium ID so callers can
# rely on `gym.make("AIDA-Flight-v0")` after importing this module.
register(
	id="AIDA-Flight-v0",
	entry_point="aida_sim.env.flight_env:FlightEnv",
	max_episode_steps=1500,
)
