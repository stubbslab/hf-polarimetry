"""triloop.beamform — sweep P(az, el) over a grid for direction finding."""

import numpy as np

from .geometry import build_N_matrix, az_el_to_khat


def beamform_grid(z_loops, loops_config,
                  az_grid_deg=None, el_grid_deg=None):
    """Compute time-averaged P(az, el) = <|B_⊥|²> over a 2D grid.

    Parameters
    ----------
    z_loops : complex ndarray, shape (3, N).  Per-loop complex baseband.
    loops_config : dict (see triloop.config.default_loops_config).
    az_grid_deg : ndarray, azimuth grid in degrees.  Defaults to
                  np.arange(0, 360, 5).
    el_grid_deg : ndarray, elevation grid in degrees.  Defaults to
                  np.arange(-5, 90, 5).  (Slightly negative to allow for
                  sub-horizon arrivals if you don't trust your azimuth.)

    Returns
    -------
    P : ndarray of shape (len(el_grid), len(az_grid)).
    az_grid : ndarray of azimuths (deg)
    el_grid : ndarray of elevations (deg)
    """
    if az_grid_deg is None:
        az_grid_deg = np.arange(0.0, 360.0, 5.0)
    if el_grid_deg is None:
        el_grid_deg = np.arange(-5.0, 90.0, 5.0)

    N_mat = build_N_matrix(loops_config)
    Ninv = np.linalg.inv(N_mat)
    z_lab = Ninv @ z_loops               # (3, N) lab-frame complex B
    # Time-averaged covariance (3x3 Hermitian)
    Sigma = (z_lab @ z_lab.conj().T) / z_lab.shape[1]

    P = np.zeros((len(el_grid_deg), len(az_grid_deg)))
    for i, el in enumerate(el_grid_deg):
        for j, az in enumerate(az_grid_deg):
            k = az_el_to_khat(az, el)
            # tr[(I - kk^T) Σ] = tr Σ - k^T Σ k
            P[i, j] = np.real(np.trace(Sigma) - k @ Sigma @ k)
    return P, az_grid_deg, el_grid_deg


def best_direction(P, az_grid, el_grid):
    """Return the (az, el) of the maximum of P."""
    i, j = np.unravel_index(np.argmax(P), P.shape)
    return float(az_grid[j]), float(el_grid[i])
