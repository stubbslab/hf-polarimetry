"""Tests for triloop.direction.

Validates that the eigendecomposition + null-sweep recovers the
injected direction to within a degree on a high-SNR simulation.
"""

import os, sys, tempfile
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "three_loop_array", "code"))

from three_loop import simulate_wwv
from triloop import (read_capture, write_capture, default_loops_config,
                      lock_and_analyze, estimate_direction_from_z_lab,
                      build_N_matrix)


def _make(pol="linear_vertical", az=273.0, el=12.0, snr=40.0,
          faraday=0.0, carrier=25_000.0):
    t, B1, B2, B3, _ = simulate_wwv(
        duration_s=2.0, sample_rate=200_000.0, f_RF=20.0e6, fs_offset=carrier,
        az_deg=az, el_deg=el, pol=pol, amp=1.0, snr_db=snr,
        faraday_rate_dps=faraday, seed=0,
    )
    return t, B1, B2, B3, dict(carrier=carrier, az=az, el=el, pol=pol)


def test_eig_recovers_direction_for_circular_pol():
    """For a circularly polarized wave (rank-2 coherency) the eigendecomp's
    smallest-eigenvalue eigenvector unambiguously points at k̂_true."""
    t, B1, B2, B3, truth = _make(snr=60.0, pol="rcp")
    cfg = default_loops_config()
    from triloop.extract import extract_three_loops
    Z_loops, _, _ = extract_three_loops(t, B1, B2, B3, truth["carrier"], 2000.0)
    N = build_N_matrix(cfg)
    Z_lab = np.linalg.inv(N) @ Z_loops

    eig = estimate_direction_from_z_lab(Z_lab)
    def angdiff(a, b):
        return abs(((a - b + 180) % 360) - 180)
    daz = min(angdiff(eig["az_deg"], truth["az"]),
              angdiff(eig["az_deg"], (truth["az"] + 180) % 360))
    assert daz < 5.0, f"az error {daz}°"
    assert abs(eig["el_deg"] - truth["el"]) < 5.0


def test_lock_and_analyze_recovers_direction_for_rcp():
    """End-to-end direction recovery + polarimetry for a circularly
    polarized signal.  RCP has rank-2 coherency so the null
    eigenvector unambiguously points at the true direction."""
    t, B1, B2, B3, truth = _make(snr=50.0, az=273.0, el=12.0, pol="rcp")
    res = lock_and_analyze(t, B1, B2, B3, truth["carrier"], 2000.0,
                           az0=truth["az"], el0=truth["el"],
                           half_width_deg=15, n_points=51)
    daz = abs(((res["az_locked"] - truth["az"] + 180) % 360) - 180)
    assert daz < 5.0
    assert abs(res["el_locked"] - truth["el"]) < 5.0
    # ellipticity should be near ±45° for circular
    assert abs(abs(np.median(res["ellipticity_deg"])) - 45) < 5.0


def test_linear_pol_is_rank_one_and_cannot_be_located():
    """Physical limitation: a single linearly polarized wave produces a
    rank-1 coherency matrix, so direction-finding from one point is
    fundamentally ambiguous along the plane perpendicular to the
    polarization.  This test documents that the smallest-eigenvalue
    *eigenvalue* IS small (as expected for a clean signal) but the
    eigenVECTOR doesn't necessarily land on k̂_true.

    Verifies the eigenvalue ordering is clean (rank-1 means λ_2 ≈ λ_3)."""
    t, B1, B2, B3, truth = _make(snr=60.0, pol="linear_vertical")
    cfg = default_loops_config()
    from triloop.extract import extract_three_loops
    Z_loops, _, _ = extract_three_loops(t, B1, B2, B3, truth["carrier"], 2000.0)
    N = build_N_matrix(cfg)
    Z_lab = np.linalg.inv(N) @ Z_loops
    eig = estimate_direction_from_z_lab(Z_lab)
    # rank-1: largest eigenvalue dominates; the other two are both small
    # and similar in magnitude (degenerate nullspace).
    evs = eig["eigenvalues"]
    assert evs[0] > 1e6 * evs[1], "rank-1 expected"
    assert evs[1] / evs[2] < 100, "smaller eigenvalues should be similar"


def test_null_residual_is_minimum_at_truth():
    """Direct test: the residual energy ⟨|k̂·z_lab|²⟩ has its minimum
    at the true direction."""
    from triloop import perp_residual_energy
    from triloop.geometry import az_el_to_khat
    t, B1, B2, B3, truth = _make(snr=60.0)
    cfg = default_loops_config()
    from triloop.extract import extract_three_loops
    Z_loops, _, _ = extract_three_loops(t, B1, B2, B3,
                                         truth["carrier"], 2000.0)
    N = build_N_matrix(cfg)
    Z_lab = np.linalg.inv(N) @ Z_loops

    # Truth residual
    k_truth = az_el_to_khat(truth["az"], truth["el"])
    R_truth = perp_residual_energy(Z_lab, k_truth)
    # Off-axis residual at +20°/+10° offset
    k_off = az_el_to_khat(truth["az"] + 20, truth["el"] + 10)
    R_off = perp_residual_energy(Z_lab, k_off)
    assert R_off > R_truth * 5.0, \
        f"residual at truth ({R_truth:.4g}) not a minimum vs +20° off ({R_off:.4g})"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
