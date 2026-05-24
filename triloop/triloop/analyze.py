"""triloop.analyze — top-level analysis pipeline.

Read three loop signals, extract narrow-band complex baseband at a
target carrier, recover the lab-frame B vector, project perpendicular
to a chosen arrival direction, and return polarimetric and
direction-dependent observables.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import (
    build_N_matrix, az_el_to_khat, perp_projector, perp_orthonormal_basis
)
from .extract import extract_three_loops
from .stokes import compute_stokes
from .config import default_loops_config, validate_loops_config


@dataclass
class AnalysisResult:
    """Container for the output of analyze()."""
    f_peak: float
    snr_db_per_loop: np.ndarray
    z_loops: np.ndarray            # complex (3, N) per-loop baseband
    z_lab: np.ndarray              # complex (3, N) lab-frame B
    z_perp: np.ndarray             # complex (3, N) projected onto plane ⊥ k̂
    intensity: np.ndarray          # |B_⊥|² (N,)
    A_p: np.ndarray                # complex (N,) along p̂
    A_q: np.ndarray                # complex (N,) along q̂
    stokes_I: np.ndarray
    stokes_Q: np.ndarray
    stokes_U: np.ndarray
    stokes_V: np.ndarray
    pol_fraction: np.ndarray
    ellipticity_deg: np.ndarray
    position_angle_deg: np.ndarray
    amp_dominant: np.ndarray       # |z_dominant_pol| (N,)
    instant_phase: np.ndarray      # rad, unwrapped & detrended
    instant_freq: np.ndarray       # Hz, time-resolved
    instant_freq_mean: float
    khat: np.ndarray
    p_hat: np.ndarray
    q_hat: np.ndarray
    loops_config: dict
    sample_rate: float
    duration_s: float


def analyze(t, B1, B2, B3, f0, BW, az_deg, el_deg,
            loops_config: Optional[dict] = None) -> AnalysisResult:
    """Run the full three-loop analysis pipeline.

    Parameters
    ----------
    t : ndarray, time (s), uniform sampling
    B1, B2, B3 : ndarray, real loop signals matching loops_config order
    f0 : float, target carrier (Hz)
    BW : float, analysis bandwidth (Hz, ±BW/2 retained)
    az_deg, el_deg : initial direction estimate (deg)
    loops_config : dict, see config.default_loops_config().  If None,
                   the default cube-vertex layout is used.

    Returns
    -------
    AnalysisResult
    """
    if loops_config is None:
        loops_config = default_loops_config()
    validate_loops_config(loops_config)

    t = np.asarray(t, dtype=np.float64)
    sr = 1.0 / np.median(np.diff(t))
    duration = float(t[-1] - t[0])

    phase_offsets = [lp.get("phase_offset_deg", 0.0)
                     for lp in loops_config["loops"]]
    Z_loops, f_peak, snrs = extract_three_loops(
        t, B1, B2, B3, f0, BW, phase_offsets_deg=phase_offsets
    )
    return analyze_z_loops(t, Z_loops, f_peak, snrs,
                           az_deg, el_deg, loops_config=loops_config)


def analyze_z_loops(t, Z_loops, f_peak, snrs_db,
                    az_deg, el_deg, loops_config=None):
    """Analyze pre-extracted complex baseband loop signals.

    Useful when the baseband has already been computed (e.g. by
    :func:`triloop.bands.extract_bands`) and you don't want to re-extract.

    Parameters
    ----------
    t : ndarray, time stamps for the baseband samples (s).
    Z_loops : complex (3, N), already-mixed loop signals at the chosen band.
    f_peak : float, the band's RF / carrier frequency (Hz) for reporting.
    snrs_db : array-like (3,), per-loop SNR estimates (dB).
    az_deg, el_deg : initial direction estimate (deg).
    loops_config : optional dict.

    Returns
    -------
    AnalysisResult
    """
    if loops_config is None:
        loops_config = default_loops_config()
    validate_loops_config(loops_config)
    t = np.asarray(t, dtype=np.float64)
    sr = 1.0 / np.median(np.diff(t))
    duration = float(t[-1] - t[0])
    snrs = np.asarray(snrs_db, dtype=np.float64)

    N_mat  = build_N_matrix(loops_config)
    Ninv   = np.linalg.inv(N_mat)
    Z_lab  = Ninv @ Z_loops

    khat   = az_el_to_khat(az_deg, el_deg)
    P_perp = perp_projector(khat)
    Z_perp = P_perp @ Z_lab

    intensity = np.real(np.sum(Z_perp * np.conj(Z_perp), axis=0))

    p_hat, q_hat = perp_orthonormal_basis(khat)
    A_p = p_hat @ Z_perp
    A_q = q_hat @ Z_perp
    s   = compute_stokes(A_p, A_q)

    C = np.array([
        [np.mean(np.abs(A_p) ** 2),               np.mean(np.conj(A_p) * A_q)],
        [np.mean(A_p * np.conj(A_q)),             np.mean(np.abs(A_q) ** 2)],
    ])
    _, V_eig = np.linalg.eigh(C)
    u_dom = V_eig[:, -1]
    z_dom = u_dom[0] * A_p + u_dom[1] * A_q
    amp_dom = np.abs(z_dom)
    phase   = np.unwrap(np.angle(z_dom))
    slope, intercept = np.polyfit(t, phase, 1)
    phase   = phase - (slope * t + intercept)
    inst_freq_t = (np.gradient(np.unwrap(np.angle(z_dom)), t) / (2 * np.pi)
                   + f_peak)
    inst_freq_mean = (slope / (2 * np.pi)) + f_peak

    return AnalysisResult(
        f_peak=float(f_peak),
        snr_db_per_loop=snrs,
        z_loops=Z_loops, z_lab=Z_lab, z_perp=Z_perp,
        intensity=intensity,
        A_p=A_p, A_q=A_q,
        stokes_I=s["I"], stokes_Q=s["Q"], stokes_U=s["U"], stokes_V=s["V"],
        pol_fraction=s["pol_fraction"],
        ellipticity_deg=s["ellipticity_deg"],
        position_angle_deg=s["position_angle_deg"],
        amp_dominant=amp_dom,
        instant_phase=phase,
        instant_freq=inst_freq_t,
        instant_freq_mean=float(inst_freq_mean),
        khat=khat, p_hat=p_hat, q_hat=q_hat,
        loops_config=loops_config,
        sample_rate=float(sr),
        duration_s=duration,
    )
