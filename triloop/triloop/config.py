"""triloop.config — small helpers for the loop-geometry config dict.

Users define their array as a Python dict in a notebook cell or in a
script.  We accept either a fully-specified dict or rely on the
default cube-vertex layout.
"""

import numpy as np


def default_loops_config():
    """Return a default loop-geometry config matching the cube-vertex
    layout described in the report.  Loop normals are 35.26° above the
    horizon, spaced 120° apart in azimuth, with L1 pointing N+up.
    """
    el = float(np.rad2deg(np.arctan(1.0 / np.sqrt(2.0))))   # 35.2644°
    return {
        "coordinate_frame": "ENU",
        "azimuth_convention": "cw_from_north",
        "elevation_convention": "above_horizon",
        "loops": [
            {"name": "L1", "channel": "A",
             "normal_az_deg":   0.0, "normal_el_deg": el,
             "gain": 1.0, "phase_offset_deg": 0.0},
            {"name": "L2", "channel": "B",
             "normal_az_deg": 120.0, "normal_el_deg": el,
             "gain": 1.0, "phase_offset_deg": 0.0},
            {"name": "L3", "channel": "C",
             "normal_az_deg": -120.0, "normal_el_deg": el,
             "gain": 1.0, "phase_offset_deg": 0.0},
        ],
    }


def validate_loops_config(cfg):
    """Raise a ValueError if the config dict is missing required keys."""
    if "loops" not in cfg or len(cfg["loops"]) != 3:
        raise ValueError("loops_config must list exactly 3 loops")
    needed = {"name", "channel", "normal_az_deg", "normal_el_deg",
              "gain", "phase_offset_deg"}
    for i, lp in enumerate(cfg["loops"]):
        missing = needed - set(lp.keys())
        if missing:
            raise ValueError(f"loop[{i}] missing keys: {missing}")
    return cfg
