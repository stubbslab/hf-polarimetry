"""triloop.stokes — polarization parameters from two complex baseband
components in the plane perpendicular to the arrival direction."""

import numpy as np


def compute_stokes(A_p, A_q):
    """Return Stokes (I, Q, U, V) and derived quantities from the two
    complex polarization components (numpy arrays of equal length).

    Returns a dict with:
      I, Q, U, V          : ndarray time series (real)
      pol_fraction        : √(Q²+U²+V²) / I, in [0, 1]
      ellipticity_deg     : ½·arctan(V / √(Q²+U²)) in degrees
                            (0 = linear, ±45 = circular)
      position_angle_deg  : ½·arctan2(U, Q) in degrees, in (-90, 90)
                            (linear-polarization axis orientation
                             relative to p̂)
    """
    A_p = np.asarray(A_p)
    A_q = np.asarray(A_q)
    I = np.abs(A_p) ** 2 + np.abs(A_q) ** 2
    Q = np.abs(A_p) ** 2 - np.abs(A_q) ** 2
    U = 2.0 * np.real(np.conj(A_p) * A_q)
    V = 2.0 * np.imag(np.conj(A_p) * A_q)

    pol_frac = np.sqrt(Q ** 2 + U ** 2 + V ** 2) / np.maximum(I, 1e-30)
    ellipticity = 0.5 * np.arctan2(V, np.sqrt(Q ** 2 + U ** 2))
    posn_angle  = 0.5 * np.arctan2(U, Q)

    return dict(
        I=I, Q=Q, U=U, V=V,
        pol_fraction=pol_frac,
        ellipticity_deg=np.rad2deg(ellipticity),
        position_angle_deg=np.rad2deg(posn_angle),
    )
