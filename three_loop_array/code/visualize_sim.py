#!/usr/bin/env python3
"""
visualize_sim.py — diagnostic visualization for simulated three-loop data.

Runs simulate_wwv() with user-controlled knobs (polarization, noise,
Faraday rotation rate, direction) and generates a 6-panel figure that
shows the raw loop signals, the per-loop FFT spectra, and the recovered
analysis output (intensity, instantaneous phase + frequency, polarization
state).  Useful both for sanity-checking the analysis pipeline and for
exploring how the recovered observables respond to ionospheric effects.

Usage:
    python3 visualize_sim.py            # default: linear pol, mild Faraday
    python3 visualize_sim.py --pol rcp --faraday 0  # RCP, no Faraday
    python3 visualize_sim.py --pol linear_vertical --faraday 90 --snr 15
    python3 visualize_sim.py --multipath
"""

import os, sys, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_HERE, "..", ".mpl_cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from three_loop import simulate_wwv, analyze, az_el_to_khat


def amp_mod_factory(kind):
    """Return a callable t -> envelope for amplitude modulation."""
    if kind is None or kind == "none":
        return None
    if kind == "multipath":
        # Two interfering rays: 1F2 hop and 2F2 hop with Δτ ≈ 200 µs and
        # slowly-varying relative amplitude.  At HF this looks like a
        # ~Hz-scale fade pattern when the difference path slowly varies.
        def env(t):
            return 0.7 + 0.3 * np.cos(2 * np.pi * 0.7 * t)
        return env
    if kind == "fade":
        def env(t):
            return 0.5 * (1.0 + np.cos(2 * np.pi * 0.2 * t))
        return env
    raise ValueError(f"unknown amp-modulation: {kind}")


def fft_panel(ax, t, B, label, sr):
    n = B.size
    F = np.fft.rfft(B)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    mag_db = 20 * np.log10(np.abs(F) + 1e-30)
    mag_db -= mag_db.max()
    ax.plot(f, mag_db, lw=0.5)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("magnitude (dB, peak-norm)")
    ax.set_title(f"FFT of {label}")
    ax.set_ylim(-80, 5)
    ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pol", default="linear_vertical",
                    choices=["linear_vertical", "linear_horizontal",
                             "rcp", "lcp", "elliptical"])
    ap.add_argument("--az", type=float, default=273.0,
                    help="arrival azimuth (deg from N, CW)")
    ap.add_argument("--el", type=float, default=12.0,
                    help="arrival elevation (deg above horizon)")
    ap.add_argument("--snr", type=float, default=20.0,
                    help="per-channel SNR in dB")
    ap.add_argument("--faraday", type=float, default=30.0,
                    help="Faraday rotation rate, deg/s")
    ap.add_argument("--faraday0", type=float, default=15.0,
                    help="initial Faraday angle, deg")
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--sr", type=float, default=200_000.0)
    ap.add_argument("--carrier", type=float, default=25_000.0,
                    help="IF carrier frequency in the simulation, Hz")
    ap.add_argument("--bw", type=float, default=2000.0,
                    help="analysis bandwidth, Hz")
    ap.add_argument("--multipath", action="store_true",
                    help="add a slow multipath amplitude modulation")
    ap.add_argument("--out", type=str, default=None,
                    help="output PNG path; default = figures/sim_<pol>.png")
    args = ap.parse_args()

    env = amp_mod_factory("multipath" if args.multipath else None)

    t, B1, B2, B3, truth = simulate_wwv(
        duration_s=args.duration, sample_rate=args.sr,
        f_RF=20.0e6, fs_offset=args.carrier,
        az_deg=args.az, el_deg=args.el,
        pol=args.pol,
        amp=1.0, snr_db=args.snr,
        faraday_rate_dps=args.faraday,
        faraday_phase0_deg=args.faraday0,
        amp_modulation=env,
        seed=0,
    )

    res = analyze(t, B1, B2, B3, args.carrier, args.bw, args.az, args.el)

    fig = plt.figure(figsize=(15, 11))
    gs  = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.3)

    # Row 0 — raw loop time series
    ax = fig.add_subplot(gs[0, :])
    n_show = min(int(0.02 * args.sr), B1.size)   # show the first 20 ms
    ax.plot(t[:n_show] * 1e3, B1[:n_show], lw=0.6, color="tab:blue", label="L1 (N+up)")
    ax.plot(t[:n_show] * 1e3, B2[:n_show], lw=0.6, color="tab:green", label="L2 (+120°)")
    ax.plot(t[:n_show] * 1e3, B3[:n_show], lw=0.6, color="tab:orange", label="L3 (-120°)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("loop B (a.u.)")
    ax.set_title(f"Raw loop time-series (first 20 ms) — "
                 f"pol={args.pol}, az={args.az}°, el={args.el}°, "
                 f"Faraday={args.faraday} deg/s, SNR={args.snr} dB")
    ax.legend(loc="best", fontsize=9); ax.grid(True, alpha=0.3)

    # Row 1 — three FFTs
    sr = args.sr
    for i, (B, lab) in enumerate(zip([B1, B2, B3],
                                      ["L1 (N+up)", "L2 (+120°)", "L3 (-120°)"])):
        ax = fig.add_subplot(gs[1, i])
        fft_panel(ax, t, B, lab, sr)
        ax.axvline(args.carrier, color="tab:red", lw=0.5, ls="--",
                   label=f"f0 = {args.carrier:.0f} Hz")
        ax.legend(loc="best", fontsize=8)

    # Row 2 — recovered amplitude & inst. frequency
    ax = fig.add_subplot(gs[2, :2])
    ax.plot(t, np.sqrt(res["intensity"]), lw=0.6, color="tab:blue",
            label="|B_⊥(t)| (pol-independent)")
    ax.plot(t, res["amp_dominant"], lw=0.6, color="tab:red", alpha=0.6,
            label="dominant pol amplitude")
    ax.set_xlabel("time (s)"); ax.set_ylabel("amplitude (a.u.)")
    ax.set_title("Recovered amplitude (intensity vs dominant pol)")
    ax.legend(loc="best", fontsize=9); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    if_freq = res["instant_freq"]
    ax.plot(t, if_freq - args.carrier, lw=0.6, color="tab:green")
    ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Δ inst. freq from f0 (Hz)")
    ax.set_title(f"Inst. freq − {args.carrier:.0f} Hz")
    ax.grid(True, alpha=0.3)

    # Row 3 — polarization
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(t, res["ellipticity_deg"], lw=0.6, color="tab:orange")
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.axhline(45, color="0.7", lw=0.5, alpha=0.4); ax.axhline(-45, color="0.7", lw=0.5, alpha=0.4)
    ax.set_xlabel("time (s)"); ax.set_ylabel("ellipticity (deg)")
    ax.set_title("Polarization: ellipticity (0=linear, ±45=circular)")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 1])
    ax.plot(t, res["position_angle_deg"], lw=0.6, color="tab:cyan")
    ax.set_xlabel("time (s)"); ax.set_ylabel("position angle (deg)")
    ax.set_title("Polarization: position angle of ellipse major axis")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 2])
    ax.plot(t, 100 * res["pol_fraction"], lw=0.6, color="black")
    ax.set_xlabel("time (s)"); ax.set_ylabel("polarization fraction (%)")
    ax.set_title("Polarization fraction (100% = fully polarized)")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Three-loop simulator visualization\n"
        f"truth: pol={args.pol}, az={args.az}°, el={args.el}°, Faraday rate={args.faraday}°/s, "
        f"SNR={args.snr} dB; recovered f_peak={res['f_peak']:.2f} Hz, "
        f"FFT SNR={res['snr_db']:.1f} dB",
        fontsize=11, weight="bold")

    out = args.out or os.path.join(_HERE, "..", "figures",
                                   f"sim_{args.pol}_far{int(args.faraday)}_snr{int(args.snr)}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    print(f"truth     : pol={args.pol}, az={args.az}, el={args.el}, "
          f"Faraday={args.faraday}°/s, SNR={args.snr} dB")
    print(f"recovered : f_peak={res['f_peak']:.4f} Hz "
          f"(f0={args.carrier:.4f}, error {res['f_peak']-args.carrier:+.4f} Hz)")
    print(f"            median pol_frac = {np.median(res['pol_fraction']):.3f}")
    print(f"            median ellipticity = {np.median(res['ellipticity_deg']):+.2f}°")
    print(f"            median position_angle = {np.median(res['position_angle_deg']):+.2f}°")


if __name__ == "__main__":
    main()
