"""Generate the bundled example capture (`example_5band_wwv.h5`).

Synthesizes a 4-channel, 0.25-second capture at 15.625 MS/s containing all
five WWV bands (2.5 / 5 / 10 / 15 / 20 MHz) via bandpass undersampling.
Each band gets a different polarization, SNR, and slow envelope fade so
the multi-band tools (`triloop view`, `triloop analyze-multi`,
`triloop browse`) have something interesting to display.

Run from the project root after installing both packages:

    python examples/make_example_capture.py

The output file is ~58 MB and committed to the repo so first-time users
have something to run the analysis tools against without buying hardware.
"""

import os
import sys
from datetime import datetime, timezone

import numpy as np

# Local layout: triloop and three_loop_array next to this script's parent.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "triloop"))
sys.path.insert(0, os.path.join(ROOT, "picoacq"))
sys.path.insert(0, os.path.join(ROOT, "three_loop_array", "code"))

from triloop.io_hdf5 import write_capture
from triloop.config import default_loops_config
from three_loop import az_el_to_khat, perp_orthonormal_basis, LOOP_NORMALS

# --- capture parameters --------------------------------------------------
fs = 15.625e6                     # picoacq default sample rate
duration_s = 0.25                 # short to keep file size reasonable
n = int(fs * duration_s)
t = np.arange(n) / fs

# (rf_hz, polarization, amplitude, snr_dB, faraday_dps, fade_freq_hz, depth)
band_specs = [
    (2.5e6,  "linear_vertical",   1.00, 30,  8.0,  0.7, 0.6),
    (5.0e6,  "linear_vertical",   0.80, 28, 20.0,  1.4, 0.7),
    (10.0e6, "elliptical",        0.60, 25,  0.0,  2.5, 0.4),
    (15.0e6, "rcp",               0.40, 22,  0.0,  3.0, 0.5),
    (20.0e6, "rcp",               0.25, 18,  0.0,  5.0, 0.3),
]

az_src, el_src = 9.0, 35.0        # toy "Fort Collins -> APO" geometry
khat = az_el_to_khat(az_src, el_src)
p_hat, q_hat = perp_orthonormal_basis(khat)


def pol_vec(name, amp):
    if name == "linear_vertical":   return 0.0, amp
    if name == "linear_horizontal": return amp, 0.0
    if name == "rcp":  return amp / np.sqrt(2), -1j * amp / np.sqrt(2)
    if name == "lcp":  return amp / np.sqrt(2), +1j * amp / np.sqrt(2)
    if name == "elliptical": return 0.7 * amp, 0.3j * amp
    raise ValueError(name)


rng = np.random.default_rng(0)
Z_lab = np.zeros((3, n), dtype=np.complex64)

for f_rf, pol, amp, snr, fday, fade_f, fade_d in band_specs:
    A_p0, A_q0 = pol_vec(pol, amp)
    Omega = np.deg2rad(fday) * t
    A_p_t = (A_p0 * np.cos(Omega) - A_q0 * np.sin(Omega)).astype(np.complex64)
    A_q_t = (A_p0 * np.sin(Omega) + A_q0 * np.cos(Omega)).astype(np.complex64)
    phi = 2 * np.pi * rng.random()
    env = (1.0 + fade_d * np.cos(2 * np.pi * fade_f * t + phi)).astype(np.float32)
    env = np.clip(env, 0.05, None)
    A_p_t *= env
    A_q_t *= env
    carrier = np.exp(1j * 2 * np.pi * f_rf * t).astype(np.complex64)
    Z_lab += (A_p_t[None, :] * p_hat[:, None].astype(np.complex64)
              + A_q_t[None, :] * q_hat[:, None].astype(np.complex64)) * carrier[None, :]

B_loops = np.real(LOOP_NORMALS @ Z_lab).astype(np.float32)
B_whip  = np.real(Z_lab[2]).astype(np.float32) * 0.5

sig_rms = float(np.sqrt(np.mean(B_loops ** 2)))
worst_snr = min(s for _, _, _, s, _, _, _ in band_specs)
n_rms = sig_rms / 10 ** (worst_snr / 20.0)
B_loops += n_rms * rng.standard_normal(B_loops.shape).astype(np.float32)
B_whip  += n_rms * rng.standard_normal(n).astype(np.float32)

chans = {"A": B_loops[0], "B": B_loops[1], "C": B_loops[2], "D": B_whip}

settings = {
    "requested_duration_s": duration_s,
    "requested_sample_rate": fs,
    "channels": list(chans.keys()),
    "ranges_volts": {ch: 2.0 for ch in chans},
    "coupling": "DC",
    "used_simulator": True,
    "auto_range_enabled": True,
    "headroom": 3.0,
    "rf_bands_hz": [b[0] for b in band_specs],
    "auto_range_probe": {
        "peaks_volts": {"A": 0.42, "B": 0.45, "C": 0.51, "D": 0.18},
        "rms_volts":   {"A": 0.21, "B": 0.22, "C": 0.24, "D": 0.09},
        "headroom": 3.0, "probe_duration_s": 0.1,
    },
    "picoscope_ranges_volts": {"A": 1.0, "B": 2.0, "C": 2.0, "D": 0.5},
    "truth": {
        "az_deg": az_src, "el_deg": el_src,
        "bands": [{"rf_hz": b[0], "pol": b[1], "amp": b[2], "snr_db": b[3],
                   "faraday_dps": b[4], "fade_freq_hz": b[5],
                   "fade_depth": b[6]} for b in band_specs],
        "note": "synthetic data; not a real over-the-air recording",
    },
}

out = os.path.join(HERE, "example_5band_wwv.h5")
write_capture(out, channels=chans, sample_rate=fs,
              start_time_utc=datetime.now(timezone.utc).isoformat(),
              scope_model="SIMULATOR-multiband-example",
              scope_serial="sim-example-01",
              capture_settings=settings,
              loops_config=default_loops_config())
print(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")
