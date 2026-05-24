#!/usr/bin/env python3
"""Render the same figures the notebook produces, but as standalone PNGs.
Useful for generating demo / docs plots without launching Jupyter."""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_HERE, ".mpl_cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from triloop import (read_capture, analyze, beamform_grid,
                      default_loops_config)


def render(input_file, carrier_hz, bw_hz, az_deg, el_deg, out_dir):
    cap = read_capture(input_file)
    print(f"loaded {input_file}: {len(cap['channels'])} channels, "
          f"{cap['sample_rate']:.0f} Hz, {cap['duration_s']:.2f} s")
    cfg = cap.get("loops_config") or default_loops_config()
    chans = [lp["channel"] for lp in cfg["loops"]]
    B = [cap["channels"][c] for c in chans]
    t = cap["time"]

    # 1) raw signals + spectra
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    n_show = min(int(0.020 * cap["sample_rate"]), len(t))
    colors = ["tab:blue", "tab:green", "tab:orange"]
    for b, ch, col in zip(B, chans, colors):
        axes[0].plot(t[:n_show]*1e3, b[:n_show], lw=0.5, color=col, label=ch)
    axes[0].set_xlabel("time (ms)"); axes[0].set_ylabel("loop signal")
    axes[0].set_title("Raw loop signals (first 20 ms)")
    axes[0].grid(True, alpha=0.3); axes[0].legend()
    for b, ch, col in zip(B, chans, colors):
        F = np.fft.rfft(b)
        f = np.fft.rfftfreq(len(b), 1.0/cap["sample_rate"])
        db = 20*np.log10(np.abs(F)+1e-30); db -= db.max()
        axes[1].plot(f, db, lw=0.5, color=col, label=ch)
    axes[1].axvline(carrier_hz, color="red", ls="--", lw=0.7,
                    label=f"target {carrier_hz:.0f} Hz")
    axes[1].set_xlim(0, min(cap["sample_rate"]/2, carrier_hz*4))
    axes[1].set_xlabel("frequency (Hz)"); axes[1].set_ylabel("|FFT| (dB peak-norm)")
    axes[1].set_ylim(-80, 5); axes[1].grid(True, alpha=0.3); axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_raw_and_spectra.png"), dpi=140)
    plt.close(fig)

    # 2) analyze + diagnostics
    res = analyze(t, B[0], B[1], B[2], carrier_hz, bw_hz, az_deg, el_deg,
                  loops_config=cfg)
    print(f"f_peak={res.f_peak:.4f} Hz   "
          f"pol_frac={np.median(res.pol_fraction):.3f}   "
          f"ellip={np.median(res.ellipticity_deg):+.2f}°   "
          f"PA={np.median(res.position_angle_deg):+.2f}°")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t, np.sqrt(res.intensity), lw=0.5, color="tab:blue",
                 label="|B_⊥(t)| (pol-indep)")
    axes[0].plot(t, res.amp_dominant, lw=0.5, color="tab:red", alpha=0.7,
                 label="dominant pol")
    axes[0].set_ylabel("amplitude"); axes[0].grid(True, alpha=0.3); axes[0].legend()
    axes[0].set_title("Recovered amplitude")
    axes[1].plot(t, np.rad2deg(res.instant_phase), lw=0.5, color="tab:purple")
    axes[1].set_ylabel("phase residual (deg)"); axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Phase residual after linear-detrend")
    axes[2].plot(t, res.instant_freq - res.f_peak, lw=0.5, color="tab:green")
    axes[2].set_ylabel("inst freq − f_peak (Hz)"); axes[2].grid(True, alpha=0.3)
    axes[2].set_title("Instantaneous frequency offset")
    axes[3].plot(t, res.ellipticity_deg, lw=0.5, color="tab:orange",
                 label="ellipticity (deg)")
    axes[3].plot(t, res.position_angle_deg, lw=0.5, color="tab:cyan",
                 label="position angle (deg)")
    axes[3].plot(t, 100*res.pol_fraction, lw=0.5, color="black", alpha=0.5,
                 label="pol fraction (%)")
    axes[3].set_ylabel("polarization"); axes[3].set_xlabel("time (s)")
    axes[3].grid(True, alpha=0.3); axes[3].legend()
    axes[3].set_title("Polarization observables")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_diagnostics.png"), dpi=140)
    plt.close(fig)

    # 3) beamforming sweep
    P, azg, elg = beamform_grid(res.z_loops, cfg,
                                az_grid_deg=np.arange(0, 360, 3),
                                el_grid_deg=np.arange(0, 90, 3))
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(P, aspect="auto", origin="lower",
                   extent=(azg[0], azg[-1], elg[0], elg[-1]),
                   cmap="viridis")
    fig.colorbar(im, ax=ax, label="P(az, el)")
    ax.plot(az_deg, el_deg, "rx", ms=12, mew=2, label="initial guess")
    i, j = np.unravel_index(np.argmax(P), P.shape)
    ax.plot(azg[j], elg[i], "wo", ms=10, mfc="none", mew=2,
            label=f"best ({azg[j]:.0f}°, {elg[i]:.0f}°)")
    ax.set_xlabel("azimuth (deg, N→E)"); ax.set_ylabel("elevation (deg)")
    ax.set_title("Beamforming: P(az, el) = ⟨|B_⊥|²⟩")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_beamform.png"), dpi=140)
    plt.close(fig)

    print(f"\nwrote 3 figures in {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_file")
    ap.add_argument("--carrier", type=float, required=True)
    ap.add_argument("--bw", type=float, default=2000.0)
    ap.add_argument("--az", type=float, required=True)
    ap.add_argument("--el", type=float, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.dirname(os.path.abspath(args.input_file))
    os.makedirs(out, exist_ok=True)
    render(args.input_file, args.carrier, args.bw, args.az, args.el, out)


if __name__ == "__main__":
    main()
