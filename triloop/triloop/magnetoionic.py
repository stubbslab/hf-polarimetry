"""triloop.magnetoionic — Appleton-Hartree O/X mode polarization for HF
ionospheric propagation.

The propagation eigenmodes of the cold magnetised plasma are computed at
the *exit point* of the ray (where X = f_p^2/f^2 → 0).  Because the
medium is locally homogeneous and the modes are an orthogonal basis, the
amplitude and shape of each mode is preserved adiabatically from exit
to receiver — no further "Faraday rotation" of the mode envelopes
happens once the wave has left the ionosphere.

Sign conventions
----------------
* k̂ points along the **Poynting vector**, i.e. *from source toward
  receiver*.  This is opposite to ``triloop.geometry.az_el_to_khat``,
  which builds a unit vector toward the source.  Use
  :func:`khat_propagation` to convert.
* (p̂, q̂) is a right-handed perpendicular-plane basis with
  p̂ = projection of B̂ onto the plane ⊥ k̂, normalised.
  q̂ = k̂ × p̂.  This makes p̂ the natural axis along which the
  ordinary (O) mode is linearly polarised in the QT limit.
* ρ = E_q / E_p is the Appleton-Hartree polarization parameter for a
  wave E = E_p p̂ + ρ E_p q̂.  At X = 0:

        ρ_{O,X} = -i q ± i √(q² + 1),       q ≡ Y_T² / (2 Y_L (1-X))

  where Y = f_H/f, Y_L = Y cosθ, Y_T = Y sinθ.

Limiting cases
--------------
* QL (θ → 0):  Y_T → 0, q → 0, so ρ_O → +i, ρ_X → -i (pure RCP/LCP).
* QT (θ → 90°):  Y_L → 0, q → ∞.  Then ρ_O → 0 (linear along p̂),
  ρ_X → -i∞ (linear along q̂; the i is a fixed phase, not rotation).
* The two modes are always orthogonal under the Hermitian inner product:
  ⟨u_O, u_X⟩ = 1 + ρ_O ρ_X̄ = 0.  In cold lossless plasma both ρ values
  are purely imaginary, ρ_O = -iq + i√(q²+1) and ρ_X = -iq - i√(q²+1),
  so ρ_O ρ_X̄ = -|ρ_X|² ρ_O / ρ_O = -1 → orthogonality is automatic.
  (Note: ρ_O ρ_X = +1 in this purely-imaginary convention, not -1 as
  in some texts that put the i outside.)

The B-field lookup falls back to a fixed mid-latitude WMM-style
approximation if pyIGRF isn't installed.  For careful work, install
``pyIGRF`` and pass an explicit epoch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# Electron gyrofrequency at |B| = 50 μT (mid-latitude default)
F_H_DEFAULT_HZ = 1.4e6
# Gyromagnetic ratio for electrons:  f_H[Hz] = GAMMA_E * |B|[T]
GAMMA_E_HZ_PER_T = 2.799249e10


# --------------------------------------------------------------- modes

def appleton_hartree_polarization(theta_rad: float, f_hz: float, *,
                                  X: float = 0.0,
                                  f_H_hz: float = F_H_DEFAULT_HZ
                                  ) -> Tuple[complex, complex]:
    """Return (ρ_O, ρ_X), the complex E_q/E_p ratios for the ordinary
    and extraordinary cold-plasma modes.

    Caller's responsibility: pass θ in radians and stay in the cold,
    collisionless, lossless approximation (no electron-collision term Z).
    """
    Y   = f_H_hz / f_hz
    Y_L = Y * np.cos(theta_rad)
    Y_T = Y * np.sin(theta_rad)
    one_minus_X = 1.0 - X
    if abs(Y_L) < 1e-14 * Y_T**2:
        # Exact QT: O mode is linear along p̂ (ρ=0), X is linear along q̂
        # (ρ ↔ ∞).  Return a finite large stand-in so the unit-vector
        # construction still produces u_X ≈ (0, 1).
        return 0.0 + 0.0j, complex(0.0, -1e15)
    q = (Y_T * Y_T) / (2.0 * Y_L * one_minus_X)
    disc = np.sqrt(q * q + 1.0)
    return complex(0.0, -q + disc), complex(0.0, -q - disc)


def mode_unit_vectors(rho_O: complex, rho_X: complex
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (u_O, u_X), each a 2-component complex unit vector in the
    (p̂, q̂) basis, suitable for matched-filter projection."""
    u_O = np.array([1.0 + 0j, rho_O], dtype=np.complex128)
    u_X = np.array([1.0 + 0j, rho_X], dtype=np.complex128)
    u_O /= np.sqrt(np.real(np.vdot(u_O, u_O)))
    u_X /= np.sqrt(np.real(np.vdot(u_X, u_X)))
    return u_O, u_X


def mode_axial_ratio(rho: complex) -> float:
    """Axial ratio b/a ∈ [0, 1] of the polarization ellipse for E ∝ (1, ρ).
    0 = linear, 1 = circular."""
    # Ellipse axes are determined by the eigen-decomposition of the
    # 2×2 coherency.  For a fully polarized state, axial ratio reduces
    # to tan(χ) where χ is the ellipticity angle:
    s = 2.0 * np.imag(rho) / (1.0 + abs(rho) ** 2)
    chi = 0.5 * np.arcsin(np.clip(s, -1.0, 1.0))
    return float(np.tan(np.abs(chi)))


def mode_ellipticity_deg(rho: complex) -> float:
    """Signed ellipticity angle χ in degrees, χ ∈ [-45°, +45°].
    χ = 0 is linear; χ = ±45° is circular (sign = handedness)."""
    s = 2.0 * np.imag(rho) / (1.0 + abs(rho) ** 2)
    return float(np.degrees(0.5 * np.arcsin(np.clip(s, -1.0, 1.0))))


def mode_orientation_deg(rho: complex) -> float:
    """Orientation angle ψ of the polarization-ellipse major axis,
    measured from p̂ toward q̂ in degrees, ψ ∈ (-90°, 90°]."""
    num = 2.0 * np.real(rho)
    den = 1.0 - abs(rho) ** 2
    return float(np.degrees(0.5 * np.arctan2(num, den)))


def circular_energy_fractions(axial_ratio: float) -> Tuple[float, float]:
    """Decompose a fully polarized elliptical wave into RCP+LCP energy
    fractions, given the (signed) axial ratio b/a in [-1, 1] or
    magnitude in [0, 1].

    For a fully polarised wave with major-axis amplitude a and
    minor-axis amplitude |b| (so axial ratio r = |b|/a), the two
    circular components carry

        P_dominant / P_total = (1 + r)^2 / (2 (1 + r^2))
        P_other     / P_total = (1 - r)^2 / (2 (1 + r^2))

    where 'dominant' is the handedness of the wave's circular
    component that matches the rotation sense of the ellipse and
    'other' is the opposite handedness.  At r = 0 (linear) both
    fractions are 1/2; at r = 1 (circular) the dominant fraction is
    1 and the other is 0.

    Parameters
    ----------
    axial_ratio : float in [0, 1] (or [-1, 1] for signed)

    Returns
    -------
    (P_dom_frac, P_other_frac), both in [0, 1] and summing to 1.
    """
    r = abs(float(axial_ratio))
    if r > 1.0:
        # Use the conjugate ellipse if the user passed >1 by accident
        r = 1.0 / r
    denom = 2.0 * (1.0 + r * r)
    P_dom = (1.0 + r) ** 2 / denom
    P_other = (1.0 - r) ** 2 / denom
    return float(P_dom), float(P_other)


def cross_pol_discrimination_db(axial_ratio: float) -> float:
    """Convenience: 10 log10(P_dom / P_other) in dB.  Returns +inf for
    a perfect circular wave (axial_ratio = 1)."""
    P_dom, P_other = circular_energy_fractions(axial_ratio)
    if P_other <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(P_dom / P_other))


# ------------------------------------------------------------ geometry

def khat_propagation(az_src_deg: float, el_src_deg: float) -> np.ndarray:
    """Return propagation-direction unit vector (Poynting, source→rx) in
    local ENU at the receiver, given the *source's* azimuth (deg from N
    CW) and elevation (deg above horizon).  This is the negative of the
    'toward source' unit vector that ``geometry.az_el_to_khat`` returns."""
    az = np.deg2rad(az_src_deg)
    el = np.deg2rad(el_src_deg)
    to_src = np.array([np.cos(el) * np.sin(az),
                       np.cos(el) * np.cos(az),
                       np.sin(el)])
    return -to_src


def igrf_b_hat_enu(lat_deg: float, lon_deg: float, alt_km: float,
                   year: float = 2026.5
                   ) -> Tuple[np.ndarray, float]:
    """Return (B̂ in local ENU, |B| in Tesla) at (lat, lon, alt).

    Uses pyIGRF if installed; otherwise falls back to a fixed
    mid-latitude WMM-style approximation: declination ≈ 9° E,
    inclination scaling linearly with latitude, |B| ≈ 50 μT.
    The fallback is good to ~5 % across the continental US, sufficient
    for HF mode-polarization sanity checks but not for tight calibration.
    """
    try:
        import pyIGRF
        # pyIGRF.igrf_value(lat, lon, alt_km, year) →
        #   (D_deg, I_deg, H_nT, X_nT, Y_nT, Z_nT, F_nT)
        # X = geographic North, Y = East, Z = Down (geomagnetic conv.)
        _, _, _, X_nT, Y_nT, Z_nT, F_nT = pyIGRF.igrf_value(
            lat_deg, lon_deg, alt_km, year)
        B_enu = np.array([Y_nT, X_nT, -Z_nT])   # E, N, U
        return B_enu / F_nT, F_nT * 1e-9
    except ImportError:
        D = np.deg2rad(9.0)
        I = np.deg2rad(60.0 + 0.6 * (lat_deg - 32.0))   # crude lat scaling
        F = 50e-6
        B_E = F * np.cos(I) * np.sin(D)
        B_N = F * np.cos(I) * np.cos(D)
        B_U = -F * np.sin(I)
        B = np.array([B_E, B_N, B_U])
        return B / F, F


@dataclass
class ExitGeometry:
    mid_lat_deg: float
    mid_lon_deg: float
    h_km: float
    range_km: float
    el_rx_deg: float            # ray elevation at the receiver (flat-Earth)
    el_rx_spherical_deg: float  # ray elevation at the receiver (spherical Earth)
    az_rx_to_src_deg: float     # great-circle bearing from rx -> source
    k_hat_enu: np.ndarray       # (3,) Poynting direction at exit, exit ENU
    B_hat_enu: np.ndarray       # (3,) B unit vector at exit, exit ENU
    theta_rad: float            # angle between k_hat and B_hat
    theta_deg: float
    B_magnitude_T: float
    f_H_hz: float


def _path_point_geometry(tx_lat_deg, tx_lon_deg, rx_lat_deg, rx_lon_deg,
                         h_km, leg, igrf_year):
    """Internal helper: compute (k̂_Poynting in local ENU at the
    sample point, B̂ in the same frame, sample lat/lon).

    leg='upgoing':  sample point is the *entry* (quarter-path from TX
                    on the upgoing leg).
    leg='descending':  sample point is the *exit* (quarter-path from
                    RX on the descending leg, i.e. three-quarter-path
                    from TX -- equivalent to the midpoint of the upper
                    portion of the ray).

    For a flat-mirror reflection the entry, apex, and exit are at the
    same altitude h_km; the entry is at quarter-path TX→RX, the apex
    at half-path, the exit at three-quarter-path.  In the upgoing leg
    k̂ points up-and-toward-RX; in the descending leg k̂ points
    down-and-toward-RX.  Both expressed in local ENU at the sample
    point.
    """
    if leg not in ("upgoing", "descending"):
        raise ValueError(f"leg must be 'upgoing' or 'descending', got {leg!r}")

    mid_lat = 0.5 * (tx_lat_deg + rx_lat_deg)
    cos_lat = np.cos(np.deg2rad(mid_lat))
    dN_full = (rx_lat_deg - tx_lat_deg) * 111.0
    dE_full = (rx_lon_deg - tx_lon_deg) * 111.0 * cos_lat
    range_km = float(np.hypot(dN_full, dE_full))

    if leg == "upgoing":
        # Entry: 1/4 of the way TX -> RX
        sample_lat = 0.75 * tx_lat_deg + 0.25 * rx_lat_deg
        sample_lon = 0.75 * tx_lon_deg + 0.25 * rx_lon_deg
        # Vector from TX (alt 0) to entry (alt h_km), expressed in
        # the *entry's* local ENU.  Flat-earth: entry-ENU ≈ TX-ENU
        # to within the ~2° rotation corresponding to a quarter of the
        # path's arc.
        sample_to_apex_E = (dE_full / 4.0)
        sample_to_apex_N = (dN_full / 4.0)
        # k_hat at entry points from TX→entry, i.e. up and toward apex
        delta = np.array([dE_full / 4.0, dN_full / 4.0, h_km])
    else:  # descending leg, the exit point
        sample_lat = 0.25 * tx_lat_deg + 0.75 * rx_lat_deg
        sample_lon = 0.25 * tx_lon_deg + 0.75 * rx_lon_deg
        # k_hat at exit points from apex→RX, expressed in exit ENU.
        # apex is at midpoint, exit is 3/4 of the way; vector from
        # exit to RX has horizontal component (dE_full/4, dN_full/4)
        # and vertical -h_km.  k_hat is along that.
        delta = np.array([dE_full / 4.0, dN_full / 4.0, -h_km])
    k_hat = delta / np.linalg.norm(delta)
    B_hat, B_mag = igrf_b_hat_enu(sample_lat, sample_lon, h_km,
                                  year=igrf_year)
    return k_hat, B_hat, B_mag, sample_lat, sample_lon, range_km


def haversine_km(lat1_deg, lon1_deg, lat2_deg, lon2_deg, R_e_km=6371.0):
    """Great-circle distance in km via haversine."""
    p1, p2 = np.deg2rad([lat1_deg, lat2_deg])
    dl = np.deg2rad(lon2_deg - lon1_deg)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R_e_km * np.arcsin(np.sqrt(a)))


def initial_bearing_deg(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """Initial great-circle bearing from (lat1,lon1) toward (lat2,lon2),
    in degrees from N CW.  Use this for the apparent source bearing at
    the receiver: pass (rx_lat, rx_lon, tx_lat, tx_lon)."""
    p1, p2 = np.deg2rad([lat1_deg, lat2_deg])
    dl = np.deg2rad(lon2_deg - lon1_deg)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)


def spherical_elevation_at_rx_deg(range_km: float, h_km: float,
                                  R_e_km: float = 6371.0) -> float:
    """Elevation angle at receiver for a single-hop reflection at
    altitude h_km on a spherical Earth.  Davies eq. 4.5 (1990).

    More accurate than the flat-Earth ``arctan(h/(range/2))`` for
    long paths (e.g. 1400 km hops to Boston, where flat-Earth
    over-estimates by ~3.5 deg).
    """
    arc_half = (range_km / 2.0) / R_e_km
    num = np.cos(arc_half) - R_e_km / (R_e_km + h_km)
    den = np.sin(arc_half)
    return float(np.degrees(np.arctan2(num, den)))


def exit_point_geometry(tx_lat_deg: float, tx_lon_deg: float,
                        rx_lat_deg: float, rx_lon_deg: float,
                        h_km: float = 250.0,
                        igrf_year: float = 2026.5) -> ExitGeometry:
    """Compute the *descending-leg exit-point* geometry for a 1-hop F2
    reflection from TX to RX.  The exit point sits at three-quarter
    path (from TX) at altitude h_km, and the wave there is descending
    toward RX.

    Flat-Earth approximation: adequate for mid-latitude HF first-pass
    work (errors of order h/R_Earth ≈ 4 % in the propagation-direction
    estimate at h=250 km).  Replace with a proper ray-trace through an
    IRI profile if you need degree-level direction accuracy.

    The reported ``el_rx_deg`` is the flat-Earth elevation
    (consistent with the k_hat used for mode projection); a more
    accurate spherical-Earth value is in ``el_rx_spherical_deg``.
    The ``az_rx_to_src_deg`` is the proper great-circle initial
    bearing.  For paths > ~1500 km the flat-Earth and spherical
    answers differ enough that the user should report the
    spherical-Earth value to operators.
    """
    k_hat, B_hat, B_mag, slat, slon, range_km = _path_point_geometry(
        tx_lat_deg, tx_lon_deg, rx_lat_deg, rx_lon_deg, h_km,
        leg="descending", igrf_year=igrf_year)

    range_haver = haversine_km(tx_lat_deg, tx_lon_deg, rx_lat_deg, rx_lon_deg)
    el_flat = float(np.degrees(np.arctan2(h_km, range_haver / 2.0)))
    el_sph  = spherical_elevation_at_rx_deg(range_haver, h_km)
    bearing = initial_bearing_deg(rx_lat_deg, rx_lon_deg,
                                  tx_lat_deg, tx_lon_deg)
    cos_theta = float(np.dot(k_hat, B_hat))
    theta = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    return ExitGeometry(
        mid_lat_deg=slat, mid_lon_deg=slon, h_km=h_km,
        range_km=range_haver,
        el_rx_deg=el_flat,
        el_rx_spherical_deg=el_sph,
        az_rx_to_src_deg=bearing,
        k_hat_enu=k_hat,
        B_hat_enu=B_hat,
        theta_rad=theta,
        theta_deg=float(np.degrees(theta)),
        B_magnitude_T=B_mag,
        f_H_hz=GAMMA_E_HZ_PER_T * B_mag,
    )


@dataclass
class EntryGeometry:
    entry_lat_deg: float
    entry_lon_deg: float
    h_km: float
    range_km: float
    k_hat_enu: np.ndarray       # (3,) Poynting at entry, entry ENU (upgoing)
    B_hat_enu: np.ndarray       # (3,) B̂ at entry, entry ENU
    theta_rad: float
    theta_deg: float
    B_magnitude_T: float
    f_H_hz: float


def entry_point_geometry(tx_lat_deg: float, tx_lon_deg: float,
                         rx_lat_deg: float, rx_lon_deg: float,
                         h_km: float = 250.0,
                         igrf_year: float = 2026.5) -> EntryGeometry:
    """Compute the *upgoing-leg entry-point* geometry for a 1-hop F2
    reflection from TX to RX.  The entry sits at quarter path (from
    TX) at altitude h_km; the wave there is climbing toward the apex.

    Symmetry: the horizontal components of k_hat at entry and exit are
    identical (both legs travel the same horizontal direction), and
    the vertical components have opposite sign (one upgoing, one
    descending).  Code-side sanity:
        k_entry[:2] == k_exit[:2]           # within float tolerance
        k_entry[2]  == -k_exit[2]
    Note: theta_entry and theta_exit do NOT generally sum to 180 deg.
    They would only if horizontal-k were perpendicular to
    horizontal-B, which is not the case for most geographic paths.
    """
    k_hat, B_hat, B_mag, slat, slon, range_km = _path_point_geometry(
        tx_lat_deg, tx_lon_deg, rx_lat_deg, rx_lon_deg, h_km,
        leg="upgoing", igrf_year=igrf_year)
    cos_theta = float(np.dot(k_hat, B_hat))
    theta = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return EntryGeometry(
        entry_lat_deg=slat, entry_lon_deg=slon, h_km=h_km,
        range_km=range_km,
        k_hat_enu=k_hat, B_hat_enu=B_hat,
        theta_rad=theta, theta_deg=float(np.degrees(theta)),
        B_magnitude_T=B_mag,
        f_H_hz=GAMMA_E_HZ_PER_T * B_mag,
    )


def gc_intermediate_point(lat1_deg, lon1_deg, lat2_deg, lon2_deg,
                          fraction):
    """Point that fraction of the way along the great circle from
    point 1 to point 2.  fraction=0.5 gives the midpoint.  Returns
    (lat_deg, lon_deg)."""
    p1 = np.deg2rad([lat1_deg, lon1_deg])
    p2 = np.deg2rad([lat2_deg, lon2_deg])
    e1 = np.array([np.cos(p1[0]) * np.cos(p1[1]),
                   np.cos(p1[0]) * np.sin(p1[1]),
                   np.sin(p1[0])])
    e2 = np.array([np.cos(p2[0]) * np.cos(p2[1]),
                   np.cos(p2[0]) * np.sin(p2[1]),
                   np.sin(p2[0])])
    e = (1 - fraction) * e1 + fraction * e2
    e /= np.linalg.norm(e)
    return (float(np.degrees(np.arcsin(e[2]))),
            float(np.degrees(np.arctan2(e[1], e[0]))))


def multi_hop_geometry(tx_lat_deg, tx_lon_deg, rx_lat_deg, rx_lon_deg,
                       n_hops: int = 1, h_km: float = 250.0,
                       igrf_year: float = 2026.5) -> ExitGeometry:
    """Geometry for the *last hop* of an n-hop F2 ground-bounce path,
    which is the wave that physically arrives at the receiver.

    n_hops = 1 is identical to ``exit_point_geometry``.  For
    n_hops = 2, the last hop runs from the great-circle midpoint
    (a ground bounce) to the receiver; for n_hops = 3 it runs from
    the 2/3 point; etc.

    Use this for paths beyond ~2200 km (1-hop F2 limit at 250 km
    altitude) where multi-hop propagation is required.  At 30 deg
    elevation the 1-hop horizon is ~880 km; at 10 deg it's ~2200 km.
    """
    if n_hops < 1:
        raise ValueError("n_hops must be >= 1")
    last_hop_start_frac = (n_hops - 1) / n_hops
    last_lat, last_lon = gc_intermediate_point(
        tx_lat_deg, tx_lon_deg, rx_lat_deg, rx_lon_deg,
        fraction=last_hop_start_frac)
    return exit_point_geometry(last_lat, last_lon,
                               rx_lat_deg, rx_lon_deg,
                               h_km=h_km, igrf_year=igrf_year)


def integrated_faraday_rotation_rad(B_cos_theta_avg_T: float,
                                    TEC_electrons_per_m2: float,
                                    f_hz: float) -> float:
    """Path-integrated Faraday rotation Ω in radians (one-way), via the
    standard radio-science formula

        Ω = (e^3 / (8 π² ε₀ m_e² c)) · f^(-2) · ∫ N_e B cos θ ds
          ≈ 2.36e4 · TEC[e/m²] · ⟨B cos θ⟩[T] · f^(-2)[Hz]
          ≈ 2.36e-12 · TEC[TECU] · ⟨B cos θ⟩[T] · f^(-2)[Hz]

    where ``B_cos_theta_avg_T`` is the path-averaged longitudinal
    component of **B** in Tesla, and TEC is in electrons per square
    metre.  See Yeh & Liu (1972) eq. 4.78, Davies (1990) eq. 6.17.

    Parameters
    ----------
    B_cos_theta_avg_T : average of |B|·cos(θ) over the ray path, Tesla.
    TEC_electrons_per_m2 : line-of-sight TEC, in electrons / m²
        (1 TECU = 1e16 e/m²).
    f_hz : carrier frequency, Hz.

    Returns
    -------
    float, accumulated rotation in radians.  Many turns at low HF.
    """
    # Constant factor: K = e³ / (8 π² ε₀ m_e² c)  ≈ 2.3649e4 SI
    K = 2.3649e4
    return K * TEC_electrons_per_m2 * B_cos_theta_avg_T / (f_hz ** 2)


# ------------------------------------------------------------- bundle

@dataclass
class ModeAtExit:
    f_hz: float
    f_H_hz: float
    theta_rad: float
    rho_O: complex
    rho_X: complex
    u_O: np.ndarray            # (2,) complex
    u_X: np.ndarray            # (2,) complex
    ellipticity_O_deg: float
    ellipticity_X_deg: float
    orientation_O_deg: float
    orientation_X_deg: float
    axial_ratio_O: float
    axial_ratio_X: float


def modes_at_exit(geom: ExitGeometry, f_hz: float, X: float = 0.0
                  ) -> ModeAtExit:
    """Compute the O/X mode polarization parameters at the exit point
    for a wave of frequency f_hz."""
    rO, rX = appleton_hartree_polarization(
        geom.theta_rad, f_hz, X=X, f_H_hz=geom.f_H_hz)
    uO, uX = mode_unit_vectors(rO, rX)
    return ModeAtExit(
        f_hz=f_hz, f_H_hz=geom.f_H_hz, theta_rad=geom.theta_rad,
        rho_O=rO, rho_X=rX, u_O=uO, u_X=uX,
        ellipticity_O_deg=mode_ellipticity_deg(rO),
        ellipticity_X_deg=mode_ellipticity_deg(rX),
        orientation_O_deg=mode_orientation_deg(rO),
        orientation_X_deg=mode_orientation_deg(rX),
        axial_ratio_O=mode_axial_ratio(rO),
        axial_ratio_X=mode_axial_ratio(rX),
    )
