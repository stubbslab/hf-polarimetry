#!/usr/bin/env python3
"""SNR-vs-azimuth at fixed elevation for the cube-vertex three-loop array.

Compares two estimators in isotropic noise:

  (a) Sum of three loop powers:   S = Σ_i ⟨B_i²⟩ ;  N = 3σ²
  (b) Coherent direction-formed:  S = ⟨|B_⊥(k̂)|²⟩ ; N = trace((I-kk^T)(N^T N)^-1) σ²
  (c) "Drop the most-aligned loop": pick the n̂_i with max |n̂_i·k̂|, discard it.
      Sum the other two:   S = Σ_{j≠i*} ⟨B_j²⟩ ;  N = 2σ²

For a polarization-averaged plane wave we take ⟨B_i²⟩ = ½(1 - (n̂_i·k̂)²).
At 35.26° elevation one loop's normal lies along k̂ when the source azimuth
aligns with that loop's azimuthal projection (0°, 120°, 240°).  In that
direction (c) drops the dead loop and is exactly equivalent to (b);
elsewhere (b) does the right thing automatically.
"""

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_HERE, ".mpl_cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from triloop.geometry import LOOP_NORMALS_DEFAULT, az_el_to_khat


def snr_curves(el_deg, az_grid_deg, sigma2=1.0):
    """Return SNRs (linear, not dB) for the three estimators."""
    N = LOOP_NORMALS_DEFAULT          # (3, 3): rows are loop normals
    NTN_inv = np.linalg.inv(N.T @ N)  # = I for our symmetric layout

    snr_sum, snr_coh, snr_drop, dot_max = [], [], [], []
    for az in az_grid_deg:
        k = az_el_to_khat(az, el_deg)
        dots = N @ k                  # (3,) inner products n̂_i · k̂
        a = dots ** 2                 # squared
        # ⟨B_i²⟩ = ½ (1 - a_i) for polarization-averaged unit-power wave
        loop_signal = 0.5 * (1 - a)

        # (a) sum of all three
        S_sum = float(np.sum(loop_signal))
        N_sum = 3.0 * sigma2
        snr_sum.append(S_sum / N_sum)

        # (c) drop the most-aligned loop
        i_drop = int(np.argmax(a))
        S_drop = float(np.sum(np.delete(loop_signal, i_drop)))
        N_drop = 2.0 * sigma2
        snr_drop.append(S_drop / N_drop)

        # (b) coherent direction-formed.  For a POLARIZATION-AVERAGED unit
        # power wave (consistent with how loop_signal above is normalized),
        # ⟨|B_⊥|²⟩ = 1, not 1/2.  Noise = trace((I - kk^T)(N^T N)^-1) · σ².
        # For our symmetric N this simplifies to 2σ².
        P = np.eye(3) - np.outer(k, k)
        S_coh = 1.0
        N_coh = float(np.trace(P @ NTN_inv)) * sigma2
        snr_coh.append(S_coh / N_coh)

        dot_max.append(float(np.sqrt(a.max())))   # cos angle to nearest normal

    return (np.array(snr_sum), np.array(snr_coh), np.array(snr_drop),
            np.array(dot_max))


def main():
    el = 35.26                                   # source elevation in degrees
    az = np.linspace(0, 360, 721)
    s_sum, s_coh, s_drop, dot_max = snr_curves(el, az)

    # Convert to dB relative to the sum-of-powers estimator
    ref = s_sum.mean()
    sum_db  = 10 * np.log10(s_sum  / ref)
    coh_db  = 10 * np.log10(s_coh  / ref)
    drop_db = 10 * np.log10(s_drop / ref)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(az, sum_db,  lw=1.5, color="tab:blue",  label="(a) sum of three loop powers")
    ax.plot(az, coh_db,  lw=1.5, color="tab:red",   label="(b) coherent direction-formed")
    ax.plot(az, drop_db, lw=1.5, color="tab:green", label="(c) drop most-aligned loop, sum others")
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    for az0, label in [(0,"L1 normal"), (120,"L2 normal"), (240,"L3 normal")]:
        ax.axvline(az0, color="0.7", lw=0.6, ls="--")
        ax.text(az0, ax.get_ylim()[1] + 0.05, label, ha="center",
                fontsize=8, color="0.4")
    ax.set_xlim(0, 360)
    ax.set_xlabel("source azimuth (deg, CW from N)")
    ax.set_ylabel("SNR relative to (a) sum-of-powers (dB)")
    ax.set_title(f"SNR vs azimuth, source elevation = {el}°, isotropic noise\n"
                 f"(cube-vertex array — loop normals at az 0°, 120°, 240°)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center")

    out = os.path.join(_HERE, "docs", "figures", "snr_vs_azimuth_el35.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)
    print("\nSNRs (linear, averaged over azimuth):")
    print(f"  (a) sum of powers           : {s_sum.mean():.4g}")
    print(f"  (b) coherent direction-form : {s_coh.mean():.4g}    "
          f"vs (a): {10*np.log10(s_coh.mean()/s_sum.mean()):+.2f} dB")
    print(f"  (c) drop most-aligned       : {s_drop.mean():.4g}    "
          f"vs (a): {10*np.log10(s_drop.mean()/s_sum.mean()):+.2f} dB")
    # peak gain at az aligned with one of the loop normals
    i_peak = int(np.argmax(s_drop))
    print(f"\npeak (c) gain over (a):       {10*np.log10(s_drop[i_peak]/s_sum[i_peak]):+.2f} dB "
          f"at az = {az[i_peak]:.1f}°")


if __name__ == "__main__":
    main()
