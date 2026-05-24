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


@dataclass
class MultiBandResult:
    rf_hz: float
    extraction: BandExtraction
    analysis: AnalysisResult
    snrs_db: np.ndarray


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
                      rf_bands=None, loops_config=None) -> dict:
    """Run analyze_z_loops on every RF band declared in the file.

    Returns a dict mapping rf_hz -> MultiBandResult.
    """
    if loops_config is None:
        loops_config = cap.get("loops_config") or default_loops_config()

    extr = extract_bands(cap, rf_bands=rf_bands, bw_hz=bw_hz,
                         decim_rate_hz=decim_rate_hz,
                         channels=list(set(loops_channels) | set(cap["channels"].keys())))

    results = {}
    for f_rf, be in extr.items():
        Z_loops = np.array([be.z[ch] for ch in loops_channels])
        # The extracted baseband is centred at DC — there's no "carrier
        # frequency within the slice" to lock to; pass the RF frequency in
        # so the AnalysisResult.f_peak field carries the band's identity.
        snrs = _per_loop_snr_from_baseband(Z_loops, be.decim_rate_hz)
        ar = analyze_z_loops(be.t, Z_loops, f_rf, snrs,
                             az_deg, el_deg, loops_config=loops_config)
        results[f_rf] = MultiBandResult(
            rf_hz=f_rf, extraction=be, analysis=ar, snrs_db=snrs,
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
