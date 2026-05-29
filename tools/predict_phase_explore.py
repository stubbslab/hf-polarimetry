#!/usr/bin/env python3
"""tools/predict_phase_explore.py

Deeper diagnostic exploration of a predict_phase batch JSONL output.

Goes beyond predict_phase_batch's three default plots to answer:

  1. How does D_pred(tau) scale with median amplitude (proxy for SNR)
     within each band?  Are the highest-SNR captures actually
     predictable, or is it ionosphere-limited even at high SNR?

  2. How does D_pred correlate with the per-segment carrier-slope std?
     If most of the variance is "receiver detune is jiggling around",
     we want to flag that vs genuine ionospheric phase noise.

  3. What does the *distribution* of D_pred(20 ms) per band look like?
     Boxplots only show medians and IQRs; we want to see the
     low-end tails (the cases where adaptive correction WOULD work).

  4. Does linear extrapolation help, hurt, or do nothing relative to
     constant-phase, on a per-file basis?  (Sometimes it overfits
     noise on coherent files; we already saw that on the high-SNR
     5 MHz capture.)

Usage
-----
    python3 tools/predict_phase_explore.py \\
        /path/to/predict_phase_results.jsonl \\
        [--out-dir explore_out]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np


def load(path: str, max_slope_std_hz: float = 5000.0) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error"):
                continue
            slope_max = r.get("per_segment_slope_max_abs_Hz")
            if slope_max is not None and slope_max > max_slope_std_hz:
                continue
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", help="results JSONL from predict_phase_batch")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: alongside the JSONL)")
    ap.add_argument("--max-slope-std-hz", type=float, default=5000.0,
                    help="quality cut on per-segment slope std (Hz)")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.jsonl))
    os.makedirs(out_dir, exist_ok=True)
    rows = load(args.jsonl, args.max_slope_std_hz)
    print(f"loaded {len(rows)} usable records from {args.jsonl}")
    if not rows:
        print("nothing to plot"); return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")

    # Build numpy arrays of common columns
    band       = np.array([r["freq_khz"] / 1000.0 for r in rows])
    hour       = np.array([r["utc_hour"]         for r in rows])
    med_amp_db = np.array([20.0 * np.log10(max(r["median_amp"], 1e-9))
                            for r in rows])
    rms_C20    = np.array([r["rms_const_20ms_rad"]  for r in rows])
    rms_L20    = np.array([r["rms_linear_20ms_rad"] for r in rows])
    slope_max  = np.array([r.get("per_segment_slope_max_abs_Hz", np.nan)
                            for r in rows])
    bands = sorted(set(band.tolist()))

    # ----------------------------------------- 1.  D_pred vs SNR per band
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for k, b in enumerate(bands):
        ax = axes.flat[k] if k < 6 else None
        if ax is None: break
        m = (band == b)
        if not np.any(m): continue
        ax.scatter(med_amp_db[m], rms_L20[m], s=8,
                   c=cmap(k % 10), alpha=0.5)
        ax.set_yscale("log")
        ax.axhline(0.5, color="0.4", lw=1, ls="--")
        ax.set_title(f"{b:g} MHz  (n={int(m.sum())})", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        if k % 3 == 0:
            ax.set_ylabel("D_pred(20 ms), linear (rad)")
        if k // 3 == 1:
            ax.set_xlabel("median amplitude (dB, arb.)")
    fig.suptitle("Phase predictability vs SNR-proxy, per band (linear predictor)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = os.path.join(out_dir, "explore_dpred_vs_snr.png")
    fig.savefig(p1, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p1}")

    # ----------------------------------------- 2.  D_pred vs slope_std
    fig, ax = plt.subplots(figsize=(8, 5))
    ok = np.isfinite(slope_max)
    ax.scatter(slope_max[ok], rms_L20[ok], s=6,
               c=band[ok], cmap="viridis", alpha=0.6)
    cb = plt.colorbar(ax.collections[0], ax=ax)
    cb.set_label("RF band (MHz)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("per-segment slope max-abs (Hz, log)")
    ax.set_ylabel("D_pred(20 ms), linear (rad, log)")
    ax.axhline(0.5, color="0.4", lw=1, ls="--")
    ax.set_title("Predictability vs per-segment carrier-slope inconsistency")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "explore_dpred_vs_slope.png")
    fig.savefig(p2, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p2}")

    # ----------------------------------------- 3.  Distribution histograms
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.geomspace(0.05, 1000, 80)
    for k, b in enumerate(bands):
        m = (band == b)
        if not np.any(m): continue
        ax.hist(rms_L20[m], bins=bins, histtype="step", lw=1.6,
                color=cmap(k % 10), label=f"{b:g} MHz (n={int(m.sum())})")
    ax.axvline(0.5, color="0.4", lw=1, ls="--",
               label="0.5 rad threshold")
    ax.set_xscale("log")
    ax.set_xlabel("D_pred(20 ms), linear (rad)")
    ax.set_ylabel("number of files")
    ax.set_title("Distribution of phase predictability per band")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p3 = os.path.join(out_dir, "explore_dpred_histograms.png")
    fig.savefig(p3, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p3}")

    # ----------------------------------------- 4.  linear vs constant
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.loglog(rms_C20, rms_L20, "o", ms=2.5, alpha=0.4, c="#1f77b4")
    lim = [min(rms_C20.min(), rms_L20.min()) * 0.7,
           max(rms_C20.max(), rms_L20.max()) * 1.3]
    ax.plot(lim, lim, "k:", lw=1, label="linear = constant")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.axhline(0.5, color="0.4", lw=0.8, ls="--", alpha=0.7)
    ax.axvline(0.5, color="0.4", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel("D_pred(20 ms), constant-phase (rad)")
    ax.set_ylabel("D_pred(20 ms), linear extrap. (rad)")
    ax.set_title("Linear extrapolation: helps when point is below diagonal")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p4 = os.path.join(out_dir, "explore_linear_vs_constant.png")
    fig.savefig(p4, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p4}")

    # ----------------------------------------- 5.  Best-of-band tabulation
    print("\nBest 3 captures per band by D_pred(20 ms) linear:")
    for b in bands:
        sub = [r for r in rows if r["freq_khz"] / 1000.0 == b]
        sub.sort(key=lambda r: r["rms_linear_20ms_rad"])
        print(f"\n  {b:g} MHz:")
        for r in sub[:3]:
            print(f"    {r.get('utc_time','?'):<25}  "
                  f"const={r['rms_const_20ms_rad']:7.3f}  "
                  f"linear={r['rms_linear_20ms_rad']:7.3f} rad  "
                  f"slope_max={r.get('per_segment_slope_max_abs_Hz', float('nan')):7.1f} Hz  "
                  f"({os.path.basename(r['path'])})")

    # ----------------------------------------- 6.  Operational summary
    print("\nFraction of files achieving D_pred(20 ms) < threshold:")
    print(f"  {'band':>6}  {'n':>6}  "
          f"{'<0.3 (C)':>10}  {'<0.5 (C)':>10}  {'<1.0 (C)':>10}  "
          f"{'<0.3 (L)':>10}  {'<0.5 (L)':>10}  {'<1.0 (L)':>10}")
    for b in bands:
        m = (band == b)
        if not np.any(m): continue
        nb = int(m.sum())
        fc = lambda thr: 100 * np.mean(rms_C20[m] < thr)
        fl = lambda thr: 100 * np.mean(rms_L20[m] < thr)
        print(f"  {b:>5g}M  {nb:>6d}  "
              f"{fc(0.3):>9.1f}%  {fc(0.5):>9.1f}%  {fc(1.0):>9.1f}%  "
              f"{fl(0.3):>9.1f}%  {fl(0.5):>9.1f}%  {fl(1.0):>9.1f}%")


if __name__ == "__main__":
    main()
