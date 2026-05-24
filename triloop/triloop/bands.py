"""triloop.bands — multi-band extraction from a bandpass-undersampled capture.

The picoacq default rate (15.625 MS/s) folds the WWV bands 2.5/5/10/15/20 MHz
to distinct baseband locations.  This module reads the file's
``capture_settings['rf_bands_hz']``, uses :mod:`picoacq.alias` to find each
band's baseband centre, and slices a complex baseband around it.

For each (band, channel) we mix the real signal to DC at the band's
baseband centre, low-pass via a brick-wall FFT mask of half-width BW/2,
decimate to a reduced IF rate, and conjugate the result if the band lives
in an even Nyquist zone (so the on-disk spectrum, post-extraction, runs
the right way for downstream analysis).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np

# picoacq lives next to triloop in the kiwiclient project tree; make the
# import resilient to either side-by-side layout or pip-installed presence.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PICOACQ_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", "picoacq"))
if _PICOACQ_SRC not in sys.path:
    sys.path.insert(0, _PICOACQ_SRC)
from picoacq.alias import alias_of  # noqa: E402


@dataclass
class BandExtraction:
    """Per-band, per-channel complex baseband slice."""
    rf_hz: float
    baseband_hz: float
    nyquist_zone: int
    inverted: bool
    bw_hz: float
    decim_rate_hz: float
    t: np.ndarray                       # decimated time axis (s)
    z: dict                             # channel-name -> complex64 array


def _extract_one_channel(b, sr_in, f_bb, bw, decim_rate, inverted):
    """Mix to DC at f_bb, low-pass to ±bw/2, decimate to ~decim_rate.
    If inverted, conjugate the result (even Nyquist zone undoes spectrum
    inversion)."""
    n = b.size
    t = np.arange(n, dtype=np.float64) / sr_in
    # Brick-wall LPF + mix via FFT.  This is the same approach as
    # extract_complex_baseband but parameterised on f_bb (not searched).
    nfft = 1 << int(np.ceil(np.log2(max(8 * n, 1024))))
    mix = np.exp(-1j * 2 * np.pi * f_bb * t)
    Z = np.fft.fft(b.astype(np.float64) * mix, n=nfft)
    fz = np.fft.fftfreq(nfft, 1.0 / sr_in)
    Z[np.abs(fz) > bw / 2] = 0.0
    z = np.fft.ifft(Z)[:n] * 2.0
    if inverted:
        z = np.conj(z)

    # Integer decimation to a rate at or above decim_rate, no further
    # filtering needed since we already brick-walled.
    decim = max(1, int(sr_in // decim_rate))
    z_d = z[::decim].astype(np.complex64)
    sr_out = sr_in / decim
    t_d = np.arange(z_d.size) / sr_out
    return t_d, z_d, sr_out


def extract_bands(cap, rf_bands=None, bw_hz=4000.0, decim_rate_hz=20_000.0,
                  channels=None):
    """Extract complex baseband slices for every (band, channel) pair.

    Parameters
    ----------
    cap : dict, the result of :func:`triloop.read_capture`.
    rf_bands : iterable of RF frequencies (Hz).  If None, read from
        ``cap['capture_settings']['rf_bands_hz']``; raises ValueError if
        absent.
    bw_hz : analysis bandwidth around each band centre (Hz).  Total kept
        is ±bw_hz/2.
    decim_rate_hz : approximate output sample rate after decimation (Hz).
    channels : iterable of channel names to extract.  Default: all.

    Returns
    -------
    dict mapping rf_hz (float) -> BandExtraction.
    """
    sr = float(cap["sample_rate"])
    if rf_bands is None:
        cs = cap.get("capture_settings", {}) or {}
        rf_bands = cs.get("rf_bands_hz")
        if not rf_bands:
            raise ValueError(
                "no rf_bands_hz in capture_settings; pass rf_bands explicitly"
            )
    rf_bands = [float(f) for f in rf_bands]
    if channels is None:
        channels = list(cap["channels"].keys())

    out = {}
    for f_rf in rf_bands:
        a = alias_of(f_rf, sr)
        # Validate that the chosen bw fits inside the Nyquist zone — if it
        # doesn't, the caller has asked for a bandwidth that overlaps an
        # adjacent band's alias and the slice will be contaminated.
        nyq_edge_dist = min(a.baseband_hz, sr / 2.0 - a.baseband_hz)
        if bw_hz / 2.0 > nyq_edge_dist:
            raise ValueError(
                f"bw_hz={bw_hz} too wide for band {f_rf/1e6:.3f} MHz "
                f"(baseband {a.baseband_hz/1e6:.3f} MHz, Nyquist edge "
                f"distance {nyq_edge_dist/1e3:.1f} kHz)"
            )
        zs = {}
        t_d = None
        sr_out = None
        for ch in channels:
            b = cap["channels"][ch]
            t_d, z_d, sr_out = _extract_one_channel(
                b, sr, a.baseband_hz, bw_hz, decim_rate_hz, a.inverted
            )
            zs[ch] = z_d
        out[f_rf] = BandExtraction(
            rf_hz=f_rf, baseband_hz=a.baseband_hz,
            nyquist_zone=a.nyquist_zone, inverted=a.inverted,
            bw_hz=bw_hz, decim_rate_hz=sr_out,
            t=t_d, z=zs,
        )
    return out
