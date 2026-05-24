"""End-to-end test: simulate -> write HDF5 -> read -> analyze."""

import os, sys, tempfile
import numpy as np
import pytest

# make the in-tree triloop package importable
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

# Use the existing simulator from three_loop_array
sys.path.insert(0, os.path.join(_HERE, "..", "..", "three_loop_array", "code"))
from three_loop import simulate_wwv

from triloop import (read_capture, write_capture, analyze,
                      default_loops_config, beamform_grid)


def _make_capture(tmp_dir, pol="linear_vertical", faraday=0.0,
                  az=273.0, el=12.0, snr=40.0, carrier=25_000.0):
    t, B1, B2, B3, _ = simulate_wwv(
        duration_s=2.0, sample_rate=200_000.0,
        f_RF=20.0e6, fs_offset=carrier,
        az_deg=az, el_deg=el, pol=pol, amp=1.0, snr_db=snr,
        faraday_rate_dps=faraday, seed=0,
    )
    out = os.path.join(tmp_dir, "cap.h5")
    write_capture(out,
                  channels={"A": B1.astype(np.float32),
                            "B": B2.astype(np.float32),
                            "C": B3.astype(np.float32)},
                  sample_rate=200_000.0,
                  scope_model="SIMULATOR",
                  loops_config=default_loops_config())
    return out, dict(carrier=carrier, az=az, el=el, pol=pol, faraday=faraday)


def test_linear_vertical_recovery():
    with tempfile.TemporaryDirectory() as tmp:
        path, truth = _make_capture(tmp)
        cap = read_capture(path)
        assert cap["sample_rate"] == 200_000.0
        assert "A" in cap["channels"]
        res = analyze(cap["time"],
                      cap["channels"]["A"], cap["channels"]["B"], cap["channels"]["C"],
                      truth["carrier"], 2000.0, truth["az"], truth["el"],
                      loops_config=cap["loops_config"])
        # frequency recovered to better than 1 Hz
        assert abs(res.f_peak - truth["carrier"]) < 1.0
        # fully polarized
        assert np.median(res.pol_fraction) > 0.99
        # linear (ellipticity ≈ 0)
        assert abs(np.median(res.ellipticity_deg)) < 5.0


def test_rcp_recovery():
    with tempfile.TemporaryDirectory() as tmp:
        path, truth = _make_capture(tmp, pol="rcp")
        cap = read_capture(path)
        res = analyze(cap["time"],
                      cap["channels"]["A"], cap["channels"]["B"], cap["channels"]["C"],
                      truth["carrier"], 2000.0, truth["az"], truth["el"],
                      loops_config=cap["loops_config"])
        # ellipticity near ±45° for circular pol
        assert abs(abs(np.median(res.ellipticity_deg)) - 45) < 5.0


def test_beamforming_finds_direction():
    with tempfile.TemporaryDirectory() as tmp:
        path, truth = _make_capture(tmp, az=273.0, el=12.0)
        cap = read_capture(path)
        res = analyze(cap["time"],
                      cap["channels"]["A"], cap["channels"]["B"], cap["channels"]["C"],
                      truth["carrier"], 2000.0, truth["az"], truth["el"],
                      loops_config=cap["loops_config"])
        P, azg, elg = beamform_grid(res.z_loops, cap["loops_config"],
                                    az_grid_deg=np.arange(0, 360, 5),
                                    el_grid_deg=np.arange(0, 90, 5))
        # The three-loop single-point array has a broad cos²β beam;
        # we just check the recovered direction is roughly in the
        # right hemisphere.  (Front/back ambiguity is by construction.)
        i, j = np.unravel_index(np.argmax(P), P.shape)
        best_az = azg[j]; best_el = elg[i]
        def angdiff(a, b):
            d = (a - b + 180) % 360 - 180
            return abs(d)
        delta_az = min(angdiff(best_az, truth["az"]),
                       angdiff(best_az, (truth["az"] + 180) % 360))
        assert delta_az < 60.0
        # don't enforce elevation: the beam pattern is essentially flat
        # in elevation for a single-point array.


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
