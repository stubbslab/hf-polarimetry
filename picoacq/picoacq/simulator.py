"""picoacq.simulator — fallback acquisition path that doesn't need
hardware.  Generates a synthetic three-loop dataset using the
existing simulator from three_loop_array/code/three_loop.py.
"""

import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM_PATH = os.path.normpath(
    os.path.join(_HERE, "..", "..", "three_loop_array", "code")
)
sys.path.insert(0, _SIM_PATH)


def simulate_capture(duration_s=2.0, sample_rate=200_000.0,
                     carrier_hz=25_000.0, az_deg=273.0, el_deg=12.0,
                     pol="linear_vertical", snr_db=30.0,
                     faraday_rate_dps=0.0, seed=0):
    """Run the three_loop.simulate_wwv() simulator and return its output
    in the same dict-of-channels format that the real acquisition code
    produces."""
    from three_loop import simulate_wwv  # type: ignore
    t, B1, B2, B3, truth = simulate_wwv(
        duration_s=duration_s, sample_rate=sample_rate,
        f_RF=20.0e6, fs_offset=carrier_hz,
        az_deg=az_deg, el_deg=el_deg,
        pol=pol, amp=1.0, snr_db=snr_db,
        faraday_rate_dps=faraday_rate_dps,
        seed=seed,
    )
    return dict(
        t=t,
        channels={"A": B1.astype(np.float32),
                  "B": B2.astype(np.float32),
                  "C": B3.astype(np.float32)},
        sample_rate=sample_rate,
        scope_model="SIMULATOR",
        scope_serial="sim-0",
        truth=truth,
    )
