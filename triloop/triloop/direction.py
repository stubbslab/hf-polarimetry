"""triloop.direction — null-eigenvector direction finding.

For a single plane wave arriving from k̂_true through isotropic noise,
the lab-frame coherency matrix
    C = ⟨z_lab(t) · z_lab(t)^H⟩
is a 3x3 Hermitian PSD matrix with eigenvalues λ_1 ≥ λ_2 ≥ λ_3 ≥ 0,
where λ_3 → 0 in the noise-free limit and its eigenvector is ±k̂_true.

The two largest-eigenvalue eigenvectors span the plane perpendicular
to k̂_true and provide the polarization basis for everything
downstream.

This module exposes:

  estimate_direction_from_z_lab(Z_lab) -> dict
    Returns the unit vector k̂ pointing at the source (one of the two
    sign choices), eigenvalues, and the perpendicular-plane basis.

  null_search(Z_lab, az0, el0, half_width_deg=30, points=121) -> dict
    Performs a 2-D fine sweep around an initial guess in (az, el) and
    returns the direction that minimises the residual perp-direction
    energy (a software null finder, sharper than maximum-finding).

  lock_and_analyze(t, B1, B2, B3, f0, BW, az0=None, el0=None, ...)
    Top-level convenience: extract complex baseband, run the
    eigendecomposition direction estimator, refine via null sweep,
    and return a full polarimetric AnalysisResult locked to the
    estimated direction.
"""

import numpy as np

from .geometry import (
    az_el_to_khat, perp_projector, perp_orthonormal_basis, build_N_matrix
)
from .extract import extract_three_loops
from .stokes import compute_stokes
from .config import default_loops_config, validate_loops_config


# --------------------- core: eigendecomposition --------------------------

def coherency_matrix(Z_lab):
    """Compute the time-averaged 3x3 lab-frame coherency matrix
    C = (1/T) Σ_t z(t) z(t)^H from a complex (3, N) array.
    """
    Z = np.asarray(Z_lab, dtype=np.complex128)
    if Z.shape[0] != 3:
        raise ValueError("Z_lab must be (3, N)")
    return (Z @ Z.conj().T) / Z.shape[1]


def estimate_direction_from_z_lab(Z_lab):
    """Estimate k̂_true as the smallest-eigenvalue eigenvector of the
    coherency matrix.

    Returns
    -------
    dict with keys:
        khat          : (3,) real unit vector toward source (or away —
                        front/back ambiguity is unresolved here)
        az_deg, el_deg
        eigenvalues   : (3,) real, descending order
        eigenvectors  : (3, 3) complex; columns are eigenvectors
                        ordered to match eigenvalues
        null_strength : λ_3 / λ_1, dimensionless ratio.  Small means
                        the null is deep — confidence in the direction
                        estimate.
        p_hat, q_hat  : complex 3-vectors spanning the plane ⊥ k̂
                        (the two largest-eigenvalue eigenvectors)
    """
    C = coherency_matrix(Z_lab)
    # eigh returns ascending; sort descending
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]

    null_vec = V[:, 2]                # smallest eigenvalue's eigenvector
    # Eigenvectors of a Hermitian matrix can be complex; the physical
    # k̂ is real.  Take the dominant real component, then sign by hemisphere.
    if np.max(np.abs(null_vec.imag)) > 1e-6 * np.max(np.abs(null_vec)):
        # Multi-mode signals can give a complex null; project to real.
        # Pick the phase that maximises Re(null_vec)·Re(null_vec).
        phase = np.exp(-1j * np.angle(np.sum(null_vec * null_vec)))
        null_vec = null_vec * phase
    khat = null_vec.real
    khat /= np.linalg.norm(khat)

    # Convention: pick the sign with positive elevation (upward sky).
    if khat[2] < 0:
        khat = -khat

    az = np.rad2deg(np.arctan2(khat[0], khat[1])) % 360.0
    el = np.rad2deg(np.arctan2(khat[2], np.hypot(khat[0], khat[1])))

    # Polarization basis = two largest eigenvectors (also complex in general)
    p_hat = V[:, 0]
    q_hat = V[:, 1]

    return dict(
        khat=khat,
        az_deg=float(az), el_deg=float(el),
        eigenvalues=w.real,
        eigenvectors=V,
        null_strength=float(w[2] / max(w[0], 1e-30)),
        p_hat=p_hat, q_hat=q_hat,
    )


# --------------------- 2-D null sweep refinement -------------------------

def perp_residual_energy(Z_lab, khat):
    """Average ⟨|k̂ · z_lab(t)|²⟩ — the energy in the supposed null
    direction.  This is the quantity to MINIMISE."""
    k = np.asarray(khat, dtype=np.float64).reshape(3)
    proj = k @ Z_lab                          # complex (N,)
    return float(np.mean(np.abs(proj) ** 2))


def null_search(Z_lab, az0, el0, half_width_deg=30.0, n_points=121):
    """2-D sweep around (az0, el0) minimising perp-residual energy.

    Returns a dict including the residual map and the best-fit
    (az_deg, el_deg) on the grid (no sub-grid interpolation -- caller
    can rerun with a smaller window if needed).
    """
    az_grid = np.linspace(az0 - half_width_deg, az0 + half_width_deg, n_points)
    el_grid = np.linspace(max(0.0, el0 - half_width_deg),
                          min(90.0, el0 + half_width_deg), n_points)
    R = np.empty((len(el_grid), len(az_grid)))
    for i, el in enumerate(el_grid):
        for j, az in enumerate(az_grid):
            khat = az_el_to_khat(az, el)
            R[i, j] = perp_residual_energy(Z_lab, khat)
    i_min, j_min = np.unravel_index(np.argmin(R), R.shape)
    return dict(
        az_grid=az_grid, el_grid=el_grid, residual=R,
        az_best=float(az_grid[j_min]),
        el_best=float(el_grid[i_min]),
        residual_min=float(R[i_min, j_min]),
        residual_max=float(R.max()),
    )


def parabolic_refine(R, az_grid, el_grid):
    """Sub-grid refinement: fit a 2-D paraboloid to the 3×3 cells around
    the residual minimum and return the analytic minimum location."""
    i, j = np.unravel_index(np.argmin(R), R.shape)
    if not (0 < i < R.shape[0] - 1 and 0 < j < R.shape[1] - 1):
        return float(az_grid[j]), float(el_grid[i])  # at edge — bail out
    # local 3x3 patch
    p = R[i-1:i+2, j-1:j+2]
    # 2D parabolic fit: dx in az, dy in el
    daz = (p[1, 2] - p[1, 0]) / 2.0
    dex = (p[2, 1] - p[0, 1]) / 2.0
    a   = (p[1, 2] - 2 * p[1, 1] + p[1, 0])
    b   = (p[2, 1] - 2 * p[1, 1] + p[0, 1])
    if abs(a) < 1e-30 or abs(b) < 1e-30:
        return float(az_grid[j]), float(el_grid[i])
    dx = -daz / a; dy = -dex / b
    daz_step = az_grid[1] - az_grid[0]
    del_step = el_grid[1] - el_grid[0]
    return float(az_grid[j] + dx * daz_step), float(el_grid[i] + dy * del_step)


# --------------------- top-level convenience ----------------------------

def lock_and_analyze(t, B1, B2, B3, f0, BW,
                     az0=None, el0=None,
                     loops_config=None,
                     refine=True, half_width_deg=20.0, n_points=81):
    """End-to-end: extract complex baseband, eigendecompose for an
    initial direction, optionally refine via 2-D null search, then
    return a polarimetric analysis locked to the estimated direction.

    Parameters
    ----------
    t, B1, B2, B3 : ndarray
        Loop signals.
    f0, BW : float
        Carrier and analysis bandwidth (Hz).
    az0, el0 : float, optional
        Initial guess for the null search.  If None, the eigendecomposition
        result is used as the initial guess.
    loops_config : dict
        Loop geometry (see triloop.config.default_loops_config).
    refine : bool
        Run the 2-D null sweep after the eigendecomposition.
    half_width_deg, n_points : null-sweep grid parameters.

    Returns
    -------
    dict containing:
        z_loops, z_lab           complex (3, N) per-channel baseband
        eig_result               from estimate_direction_from_z_lab()
        sweep                    from null_search() (if refine=True)
        az_locked, el_locked     float, the final locked direction
        khat                     (3,)
        intensity                |B_⊥|² time series at the locked dir
        A_p, A_q                 complex polarization components
        stokes                   from compute_stokes(A_p, A_q)
        ...                      same fields as analyze()
    """
    if loops_config is None:
        loops_config = default_loops_config()
    validate_loops_config(loops_config)

    # 1) per-loop complex baseband
    phase_offsets = [lp.get("phase_offset_deg", 0.0)
                     for lp in loops_config["loops"]]
    Z_loops, f_peak, snrs = extract_three_loops(
        t, B1, B2, B3, f0, BW, phase_offsets_deg=phase_offsets
    )

    # 2) lab-frame complex B
    N_mat = build_N_matrix(loops_config)
    Z_lab = np.linalg.inv(N_mat) @ Z_loops

    # 3) eigendecomposition direction
    eig = estimate_direction_from_z_lab(Z_lab)

    # 4) optional 2-D null sweep
    sweep = None
    if refine:
        seed_az = eig["az_deg"] if az0 is None else float(az0)
        seed_el = eig["el_deg"] if el0 is None else float(el0)
        sweep = null_search(Z_lab, seed_az, seed_el,
                            half_width_deg=half_width_deg,
                            n_points=n_points)
        az_locked, el_locked = parabolic_refine(
            sweep["residual"], sweep["az_grid"], sweep["el_grid"])
    else:
        az_locked, el_locked = eig["az_deg"], eig["el_deg"]

    # 5) lock onto the direction and analyze polarization
    khat = az_el_to_khat(az_locked, el_locked)
    P_perp = perp_projector(khat)
    Z_perp = P_perp @ Z_lab
    intensity = np.real(np.sum(Z_perp * np.conj(Z_perp), axis=0))

    p_hat, q_hat = perp_orthonormal_basis(khat)
    A_p = p_hat @ Z_perp
    A_q = q_hat @ Z_perp
    stokes = compute_stokes(A_p, A_q)

    return dict(
        z_loops=Z_loops, z_lab=Z_lab,
        f_peak=f_peak, snr_db_per_loop=snrs,
        eig=eig, sweep=sweep,
        az_locked=az_locked, el_locked=el_locked, khat=khat,
        p_hat=p_hat, q_hat=q_hat,
        intensity=intensity,
        A_p=A_p, A_q=A_q,
        stokes_I=stokes["I"], stokes_Q=stokes["Q"],
        stokes_U=stokes["U"], stokes_V=stokes["V"],
        pol_fraction=stokes["pol_fraction"],
        ellipticity_deg=stokes["ellipticity_deg"],
        position_angle_deg=stokes["position_angle_deg"],
    )
