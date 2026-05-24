"""picoacq — PicoScope 5444D data acquisition for triloop captures."""
from .recorder import capture
from .simulator import simulate_capture

__version__ = "0.1.0"

__all__ = ["capture", "simulate_capture"]
