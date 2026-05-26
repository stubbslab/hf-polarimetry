r"""triloop.multiband — run the analysis pipeline on every RF band of a
multi-band capture (e.g.\ all 5 WWV bands at once)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .bands import extract_bands, BandExtraction
from .analyze import analyze_z_loops, AnalysisResult
from .config import default_loops_config
from .geometry import perp_orthonormal_basis, az_el_to_khat
from .magnetoionic import (
    exit_point_geometry, modes_at_exit, ExitGeometry, ModeAtExit,
)


@dataclass
class ModeSeparation:
    """O/X mode time series at one band, from matched-filter projection
    of the antenna's lab-frame perp-plane signal onto the predicted
    Appleton-Hartree mode unit vectors at the exit point.
    """
    rf_hz: float
    geometry: ExitGeometry
    mode: ModeAtExit
    a_O: np.ndarray         # (N,) complex64, O-mode amplitude time series
    a_X: np.ndarray         # (N,) complex64, X-mode amplitude
    delta_phase_rad: np.ndarray  # (N,), arg(a_O) - arg(a_X), unwrapped


@dataclass
class MultiBandResult:
    rf_hz: float
    extraction: BandExtraction
    analysis: AnalysisResult
    snrs_db: np.ndarray
    mode: Optional[ModeSeparation] = None


def _project_onto_modes(ar: AnalysisResult, geom: ExitGeometry,
                        f_rf: float) -> ModeSeparation:
    """Given an AnalysisResult (which carries A_p, A_q in the receiver's
    horizontal-first perp-plane basis) and the path geometry, rotate
    into the magnetoionic (p̂_B, q̂_B) basis where p̂_B is the projection
    of B onto the perp-plane, then matched-filter against the O and X
    mode unit vectors.

    Mathematics:
      The receiver basis (p̂_R, q̂_R) and the magnetoionic basis
      (p̂_B, q̂_B) span the same 2-D plane (the plane perpendicular to
      k_hat at the receiver), so they're related by a 2×2 rotation:
          (A_pB, A_qB) = R(α) (A_pR, A_qR)
      where α is the angle from p̂_R to p̂_B in the perp-plane.
      Then:
          a_O = u_O† (A_pB, A_qB)
          a_X = u_X† (A_pB, A_qB)

    The k_hat used for the receiver basis is the toward-source one
    (azimuth/elevation of the source as seen from the receiver).  The
    magnetoionic geom.k_hat_enu is the Poynting direction at the exit
    point, which is the same line of sight but expressed in exit-ENU.
    For mode projection at the receiver we use the receiver's k̂; the
    p̂_B direction is the projection of B onto that perp-plane.  We
    take B from geom.B_hat_enu (exit-ENU); since the exit and receiver
    ENU frames differ only by ~5° rotation for FC->APO, this is fine
    for first-pass mode separation.  For careful work, transform B to
    receiver-ENU first.
    """
    # Receiver-side k_hat (toward source = -Poynting at receiver)
    az_src = geom.az_rx_to_src_deg
    el_src = geom.el_rx_deg
    k_to_src = az_el_to_khat(az_src, el_src)   # toward source
    p_hat_R, q_hat_R = perp_orthonormal_basis(k_to_src)

    # B at exit, projected onto receiver's perp-plane and unitized.
    # Using exit-ENU B in receiver's perp-plane basis is approximate but
    # adequate; both ENU frames are within a few degrees for any single
    # 1-hop mid-latitude path.
    B = geom.B_hat_enu
    B_perp = B - np.dot(B, k_to_src) * k_to_src
    B_perp_norm = np.linalg.norm(B_perp)
    if B_perp_norm < 1e-12:
        # Pure QL: any orthogonal pair works; default to receiver basis
        p_hat_B = p_hat_R
        q_hat_B = q_hat_R
    else:
        p_hat_B = B_perp / B_perp_norm
        q_hat_B = np.cross(k_to_src, p_hat_B)
        q_hat_B = q_hat_B / np.linalg.norm(q_hat_B)

    # Rotation angle α: p̂_R → p̂_B in the perp-plane
    cos_alpha = float(np.dot(p_hat_R, p_hat_B))
    sin_alpha = float(np.dot(q_hat_R, p_hat_B))
    # (A_pB, A_qB) = R(α)(A_pR, A_qR);  R(α) = [[c, s],[-s, c]] for our sign
    A_pB = ar.A_p * cos_alpha + ar.A_q * sin_alpha
    A_qB = -ar.A_p * sin_alpha + ar.A_q * cos_alpha

    # Recompute mode at exit for *this* frequency (modes_at_exit takes f_hz)
    mode = modes_at_exit(geom, f_rf)
    uO, uX = mode.u_O, mode.u_X    # 2-vectors in (p_B, q_B) basis

    # Hermitian inner product gives the matched-filter amplitude
    # a_M(t) = u_M† (A_pB(t), A_qB(t))
    A_pq = np.stack([A_pB, A_qB], axis=0)   # shape (2, N), complex
    a_O = (np.conj(uO[0]) * A_pq[0] + np.conj(uO[1]) * A_pq[1]).astype(
        np.complex64
    )
    a_X = (np.conj(uX[0]) * A_pq[0] + np.conj(uX[1]) * A_pq[1]).astype(
        np.complex64
    )
    delta_phase = np.unwrap(np.angle(a_O) - np.angle(a_X))

    return ModeSeparation(
        rf_hz=f_rf, geometry=geom, mode=mode,
        a_O=a_O, a_X=a_X, delta_phase_rad=delta_phase.astype(np.float32),
    )


def _per_loop_snr_from_baseband(z_loops_full, sr):
    """Quick FFT-based SNR estimate per loop on a band-extracted complex
    baseband: peak / median-of-far-bins, in dB."""
    out = []
    for z in z_loops_full:
        n = z.size
        nfft = 1 << int(np.ceil(np.log2(min(n, 1 << 16))))
        nfft = min(nfft, n)
        S = np.abs(np.fft.fft(z[:nfft]))
        peak = float(S.max())
        # noise floor = median of bins outside the central ±10% of the
        # window, where the band's tone lives after mixing
        ix = np.arange(nfft)
        center_band = (np.abs(ix - nfft // 2) > nfft // 10)
        noise = float(np.median(S[center_band])) if np.any(center_band) else float("nan")
        if noise <= 0 or not np.isfinite(noise):
            out.append(float("nan"))
        else:
            out.append(20.0 * np.log10(peak / noise))
    return np.array(out)


def analyze_all_bands(cap, az_deg, el_deg, *,
                      loops_channels=("A", "B", "C"),
                      bw_hz=4000.0, decim_rate_hz=20_000.0,
                      rf_bands=None, loops_config=None,
                      tx_lat_deg=None, tx_lon_deg=None,
                      rx_lat_deg=None, rx_lon_deg=None,
                      reflection_height_km=250.0) -> dict:
    """Run analyze_z_loops on every RF band declared in the file.

    If TX and RX coordinates are supplied (all four of
    ``tx_lat_deg, tx_lon_deg, rx_lat_deg, rx_lon_deg``), each
    MultiBandResult also carries a ModeSeparation with the
    matched-filter projections onto the predicted O/X modes at the
    descending exit point.

    Returns a dict mapping rf_hz -> MultiBandResult.
    """
    if loops_config is None:
        loops_config = cap.get("loops_config") or default_loops_config()

    extr = extract_bands(cap, rf_bands=rf_bands, bw_hz=bw_hz,
                         decim_rate_hz=decim_rate_hz,
                         channels=list(set(loops_channels) | set(cap["channels"].keys())))

    coords_supplied = all(c is not None for c in
                          (tx_lat_deg, tx_lon_deg, rx_lat_deg, rx_lon_deg))
    geom = (exit_point_geometry(tx_lat_deg, tx_lon_deg,
                                rx_lat_deg, rx_lon_deg,
                                h_km=reflection_height_km)
            if coords_supplied else None)

    results = {}
    for f_rf, be in extr.items():
        Z_loops = np.array([be.z[ch] for ch in loops_channels])
        # The extracted baseband is centred at DC — there's no "carrier
        # frequency within the slice" to lock to; pass the RF frequency in
        # so the AnalysisResult.f_peak field carries the band's identity.
        snrs = _per_loop_snr_from_baseband(Z_loops, be.decim_rate_hz)
        ar = analyze_z_loops(be.t, Z_loops, f_rf, snrs,
                             az_deg, el_deg, loops_config=loops_config)
        mode_sep = (_project_onto_modes(ar, geom, f_rf)
                    if geom is not None else None)
        results[f_rf] = MultiBandResult(
            rf_hz=f_rf, extraction=be, analysis=ar, snrs_db=snrs,
            mode=mode_sep,
        )
    return results


def make_multiband_figure(results, out_path=None):
    """Build a comparison figure: one row per band, columns showing
    intensity time series, instantaneous frequency, ellipticity, and
    polarization fraction.  Returns the figure or saves to out_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = sorted(results.keys())
    n = len(bands)
    if n == 0:
        raise ValueError("no bands in multiband result")

    fig, axes = plt.subplots(n, 4, figsize=(15, 2.4 * n + 0.6),
                             squeeze=False, sharex="col")
    fig.suptitle("triloop multi-band analysis", fontsize=12)

    for i, f_rf in enumerate(bands):
        r = results[f_rf]
        ar = r.analysis
        be = r.extraction
        t = be.t
        row_label = (f"{f_rf/1e6:g} MHz\n"
                     f"zone {be.nyquist_zone}"
                     + (" inv" if be.inverted else ""))

        ax = axes[i, 0]
        ax.plot(t, ar.intensity, lw=0.8)
        ax.set_ylabel(row_label)
        if i == 0: ax.set_title("|B⊥|² (intensity)", fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[i, 1]
        ax.plot(t, ar.instant_freq, lw=0.6)
        if i == 0: ax.set_title("instantaneous freq (Hz, baseband)", fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[i, 2]
        ax.plot(t, ar.ellipticity_deg, lw=0.6)
        ax.axhline(0, color="k", alpha=0.3, lw=0.5)
        ax.set_ylim(-46, 46)
        if i == 0: ax.set_title("ellipticity (deg)", fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[i, 3]
        ax.plot(t, ar.pol_fraction, lw=0.6)
        ax.set_ylim(0, 1.05)
        if i == 0: ax.set_title("polarization fraction", fontsize=10)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return out_path
    return fig
