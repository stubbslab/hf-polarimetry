"""triloop.geometry — array geometry and direction transforms.

Lab frame: right-handed (East, North, Up).  Azimuth measured clockwise
from North.  Elevation measured above the horizontal.  See the report.
"""

import numpy as np

# Default cube-vertex geometry: loop normals tilt arccos(1/√3) = 54.7356°
# from zenith (i.e. 35.26° above horizon) and are spaced 120° apart in
# azimuth, with L1 pointing N + up.
_DEFAULT_NORMAL_EL_DEG = float(np.rad2deg(np.arctan(1.0 / np.sqrt(2.0))))


def _normal_from_az_el(az_deg, el_deg):
    """Unit vector pointing in azimuth/elevation, in (E, N, Up).
    az measured CW from N, el measured above horizon."""
    a = np.deg2rad(az_deg); e = np.deg2rad(el_deg)
    return np.array([
        np.cos(e) * np.sin(a),     # East
        np.cos(e) * np.cos(a),     # North
        np.sin(e),                 # Up
    ], dtype=np.float64)


def build_N_matrix(loops_config):
    """Build the 3x3 geometry matrix N from a loops config dict.
    Rows of N are the loop-normal unit vectors in lab coordinates,
    in the order that the loops appear in cfg['loops'].

    The user's per-loop scalar `gain` is folded in here — i.e. each row
    of N is multiplied by the gain so that B_i = N[i] · B_lab matches
    the recorded loop signal directly.  Per-loop phase_offset_deg is
    NOT folded into N (it's applied to the complex baseband instead;
    see analyze.py).
    """
    rows = []
    for lp in loops_config["loops"]:
        nhat = _normal_from_az_el(lp["normal_az_deg"], lp["normal_el_deg"])
        rows.append(nhat * float(lp["gain"]))
    return np.asarray(rows, dtype=np.float64)


# Backwards-compatible default
LOOP_NORMALS_DEFAULT = build_N_matrix({
    "loops": [
        {"normal_az_deg":   0.0, "normal_el_deg": _DEFAULT_NORMAL_EL_DEG, "gain": 1.0},
        {"normal_az_deg": 120.0, "normal_el_deg": _DEFAULT_NORMAL_EL_DEG, "gain": 1.0},
        {"normal_az_deg": -120.0, "normal_el_deg": _DEFAULT_NORMAL_EL_DEG, "gain": 1.0},
    ]
})


def az_el_to_khat(az_deg, el_deg):
    """Unit vector pointing TOWARD the source, in (E, N, Up)."""
    return _normal_from_az_el(az_deg, el_deg)


def perp_projector(khat):
    """3x3 projector P = I - khat khat^T."""
    k = np.asarray(khat, dtype=np.float64).reshape(3)
    return np.eye(3) - np.outer(k, k)


def perp_orthonormal_basis(khat):
    """Return two orthonormal vectors p̂, q̂ spanning the plane ⊥ khat.

    p̂ lies in the local horizontal plane and is perpendicular to k̂
    (the natural "horizontal polarization axis" of the wave).
    q̂ = k̂ × p̂ is the orthogonal in-plane axis (generally upward).
    Falls back to East as the reference if k̂ is straight up.
    """
    k = np.asarray(khat, dtype=np.float64).reshape(3)
    z = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(k, z)) > 0.999:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = z
    p = np.cross(k, ref); p /= np.linalg.norm(p)
    q = np.cross(k, p);   q /= np.linalg.norm(q)
    return p, q
