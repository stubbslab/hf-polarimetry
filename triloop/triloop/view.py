"""triloop.view — quick-look QC plots for a capture file.

Produces a single multi-panel figure:
  - top: overall power spectrum across the full Nyquist band, with each
    rf_bands_hz entry annotated at its baseband alias position.
  - middle: per-band zoom (one column per band), spectrum un-inverted for
    even-zone bands so the displayed frequency axis runs the right way.
  - bottom-left: per-channel time-domain amplitudes (full capture).
  - bottom-right: auto-range probe table from capture_settings, if any.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from .bands import alias_of


def _channel_psd(x, sr, nfft=None):
    """Single-segment magnitude PSD (one-sided) of a real signal."""
    n = x.size
    if nfft is None:
        nfft = 1 << int(np.ceil(np.log2(min(n, 1 << 18))))
    nfft = min(nfft, n)
    xw = x[:nfft] * np.hanning(nfft)
    S = np.fft.rfft(xw)
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    p = (np.abs(S) ** 2) / (nfft * sr)
    p_db = 10.0 * np.log10(np.maximum(p, 1e-30))
    return f, p_db


def _band_zoom_spectrum(x, sr, f_bb, bw, inverted, nfft=1 << 16):
    """Spectrum of x re-centred on f_bb, returning a freq axis in kHz
    relative to band centre.  If inverted (even Nyquist zone), reverse
    the displayed axis so the user reads frequencies in the correct sense.
    """
    n = x.size
    nfft = min(nfft, n)
    xw = x[:nfft] * np.hanning(nfft)
    S = np.fft.fft(xw)
    f = np.fft.fftfreq(nfft, 1.0 / sr)
    # roll to band centre
    df = f - f_bb
    mask = np.abs(df) <= bw / 2.0
    p_db = 10.0 * np.log10(np.maximum(np.abs(S[mask]) ** 2, 1e-30))
    f_off = df[mask]
    order = np.argsort(f_off)
    f_off = f_off[order]
    p_db = p_db[order]
    if inverted:
        f_off = -f_off[::-1]
        p_db = p_db[::-1]
    return f_off / 1e3, p_db


def make_view_figure(cap, rf_bands=None, bw_hz=8000.0, out_path=None):
    """Build the QC figure.  If out_path is given, save and return the
    path; otherwise return the matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sr = float(cap["sample_rate"])
    cs = cap.get("capture_settings", {}) or {}
    if rf_bands is None:
        rf_bands = cs.get("rf_bands_hz") or []
    rf_bands = [float(f) for f in rf_bands]
    chans = list(cap["channels"].keys())

    n_bands = len(rf_bands)
    # Bottom row uses a fixed 4-column layout (time-domain spans 3, table 1)
    # so single-band captures don't collide.
    n_cols = max(n_bands, 4)
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, n_cols,
                          height_ratios=[1.4, 1.1, 1.0],
                          hspace=0.45, wspace=0.45)

    # --- row 1: full-Nyquist overview, all channels overlaid ----------
    ax0 = fig.add_subplot(gs[0, :])
    for ch in chans:
        f, p = _channel_psd(cap["channels"][ch], sr)
        ax0.plot(f / 1e6, p, lw=0.7, alpha=0.8, label=ch)
    ax0.set_xlim(0, sr / 2 / 1e6)
    ax0.set_xlabel("frequency (MHz, baseband)")
    ax0.set_ylabel("PSD (dB, arb.)")
    ax0.set_title(
        f"Capture overview — sr={sr/1e6:.4f} MS/s, "
        f"dur={cap['duration_s']:.2f} s, channels {','.join(chans)}"
    )
    ax0.legend(loc="upper right", fontsize=9, ncol=len(chans))
    ax0.grid(True, alpha=0.3)

    # Annotate each RF band at its alias location.
    ymin, ymax = ax0.get_ylim()
    for f_rf in rf_bands:
        a = alias_of(f_rf, sr)
        ax0.axvline(a.baseband_hz / 1e6, color="k",
                    alpha=0.45, lw=0.8, ls="--")
        ax0.text(a.baseband_hz / 1e6, ymax - 2,
                 f"{f_rf/1e6:g} MHz\nz{a.nyquist_zone}"
                 + (" inv" if a.inverted else ""),
                 ha="center", va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.2",
                           fc="white", ec="0.6", alpha=0.85))

    # --- row 2: per-band zoom -----------------------------------------
    if n_bands == 0:
        ax_empty = fig.add_subplot(gs[1, :])
        ax_empty.text(0.5, 0.5, "no rf_bands_hz in capture_settings",
                      ha="center", va="center", fontsize=11)
        ax_empty.set_axis_off()
    else:
        # Distribute the n_bands zoom panels across the n_cols-wide row.
        col_step = n_cols // n_bands
        for j, f_rf in enumerate(rf_bands):
            a = alias_of(f_rf, sr)
            c0 = j * col_step
            c1 = c0 + col_step if j < n_bands - 1 else n_cols
            ax = fig.add_subplot(gs[1, c0:c1])
            for ch in chans:
                fk, pk = _band_zoom_spectrum(
                    cap["channels"][ch], sr, a.baseband_hz, bw_hz, a.inverted
                )
                ax.plot(fk, pk, lw=0.8, label=ch)
            ax.set_title(f"{f_rf/1e6:g} MHz  (zone {a.nyquist_zone}"
                         + (", inverted" if a.inverted else "") + ")",
                         fontsize=10)
            ax.set_xlabel("Δf from RF (kHz)")
            if j == 0:
                ax.set_ylabel("PSD (dB)")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-bw_hz / 2e3, bw_hz / 2e3)

    # --- row 3, left: per-channel time-domain amplitude ---------------
    ax_t = fig.add_subplot(gs[2, : n_cols - 1])
    n0 = next(iter(cap["channels"].values())).size
    # Decimate for plotting so we don't push 125 M points into matplotlib.
    stride = max(1, n0 // 8000)
    t_plot = cap["time"][::stride]
    for ch in chans:
        ax_t.plot(t_plot, cap["channels"][ch][::stride],
                  lw=0.5, alpha=0.8, label=ch)
    ax_t.set_xlabel("time (s)")
    ax_t.set_ylabel("amplitude (V)")
    ax_t.set_title("Time-domain (decimated for display)")
    ax_t.grid(True, alpha=0.3)

    # --- row 3, right: probe / range table ----------------------------
    ax_tab = fig.add_subplot(gs[2, n_cols - 1])
    probe = cs.get("auto_range_probe") or {}
    pico_ranges = cs.get("picoscope_ranges_volts") or {}
    if probe or pico_ranges:
        peaks = probe.get("peaks_volts", {}) or {}
        rmss = probe.get("rms_volts", {}) or {}
        rows = []
        for ch in chans:
            r = pico_ranges.get(ch, "—")
            p = peaks.get(ch)
            r_rms = rmss.get(ch)
            rows.append([ch,
                         f"{p*1000:.1f}" if isinstance(p, (int, float)) else "—",
                         f"{r_rms*1000:.1f}" if isinstance(r_rms, (int, float)) else "—",
                         f"±{r}"])
        ax_tab.set_axis_off()
        tbl = ax_tab.table(
            cellText=rows,
            colLabels=["ch", "peak (mV)", "RMS (mV)", "Pico range"],
            loc="center", cellLoc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.3)
        ax_tab.set_title("Auto-range probe", fontsize=10)
    else:
        ax_tab.text(0.5, 0.5, "no auto-range probe metadata",
                    ha="center", va="center", fontsize=10, alpha=0.6)
        ax_tab.set_axis_off()

    fig.suptitle(
        f"triloop view — {os.path.basename(cap.get('_path', '?'))}   "
        f"start={cap.get('start_time_utc','')}",
        fontsize=11, y=0.995
    )

    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return out_path
    return fig
