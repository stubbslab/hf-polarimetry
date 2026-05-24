#!/usr/bin/env python3
"""SNR over the upper hemisphere (azimuth × elevation) for the
cube-vertex three-loop array, isotropic noise.

Renders THREE views of the same data:
  Left   : 3D hemisphere mesh, SNR(az, el) of estimator (c) "drop most
           aligned loop, sum the other two", color-coded.
  Middle : flat polar (az, el) heatmap of the same surface.
  Right  : 1D azimuth slices at three elevations (15°, 35°, 60°) for
           all three estimators, for context.

The hemisphere views also overlay the locations of the three loop normals
(stars) and the SNR peaks (white circles).
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
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from triloop.geometry import LOOP_NORMALS_DEFAULT, az_el_to_khat


def estimator_snrs(az_deg, el_deg, sigma2=1.0):
    """Return (snr_sum, snr_coh, snr_drop) for one (az, el)."""
    N = LOOP_NORMALS_DEFAULT
    NTN_inv = np.linalg.inv(N.T @ N)
    k = az_el_to_khat(az_deg, el_deg)
    a = (N @ k) ** 2                          # (n_i · k)^2
    loop_signal = 0.5 * (1.0 - a)             # ⟨B_i²⟩ for unit-power wave

    # (a) sum of all three loop powers
    snr_sum = float(np.sum(loop_signal)) / (3.0 * sigma2)

    # (c) drop the most-aligned loop
    i_drop = int(np.argmax(a))
    snr_drop = float(np.sum(np.delete(loop_signal, i_drop))) / (2.0 * sigma2)

    # (b) coherent perp-projection.  Polarization-averaged signal = 1
    # (matches normalization of loop_signal above).
    P = np.eye(3) - np.outer(k, k)
    snr_coh = 1.0 / (float(np.trace(P @ NTN_inv)) * sigma2)

    return snr_sum, snr_coh, snr_drop


def build_grid(az_step=2.0, el_step=2.0):
    az = np.arange(0.0, 360.0 + az_step / 2, az_step)
    el = np.arange(0.0, 90.0 + el_step / 2, el_step)
    AZ, EL = np.meshgrid(az, el)
    SUM  = np.zeros_like(AZ)
    COH  = np.zeros_like(AZ)
    DROP = np.zeros_like(AZ)
    for i in range(EL.shape[0]):
        for j in range(AZ.shape[1]):
            s, c, d = estimator_snrs(AZ[i, j], EL[i, j])
            SUM[i, j], COH[i, j], DROP[i, j] = s, c, d
    return az, el, SUM, COH, DROP


def hemisphere_xyz(az_deg, el_deg):
    """Project (az, el) onto a unit upper hemisphere in (x, y, z).
    Convention: az clockwise from +y (so y axis = North), z up.
    Returns x, y, z (each same shape as inputs)."""
    a = np.deg2rad(az_deg); e = np.deg2rad(el_deg)
    x = np.cos(e) * np.sin(a)     # East
    y = np.cos(e) * np.cos(a)     # North
    z = np.sin(e)                 # Up
    return x, y, z


def make_plot():
    az, el, SUM, COH, DROP = build_grid()
    AZ, EL = np.meshgrid(az, el)
    X, Y, Z = hemisphere_xyz(AZ, EL)

    # We'll color the surface by SNR_drop in dB, relative to SNR_sum mean.
    ref = SUM.mean()
    DROP_dB = 10 * np.log10(DROP / ref)
    SUM_dB  = 10 * np.log10(SUM  / ref)
    COH_dB  = 10 * np.log10(COH  / ref)

    vmin, vmax = -2.0, 2.0
    cmap = cm.viridis
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    facecolors = cmap(norm(DROP_dB))

    fig = plt.figure(figsize=(16, 6.5))

    # ----------------------------- 3D hemisphere
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax3d.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1,
                      linewidth=0, antialiased=False, shade=False)
    # Loop normals as stars
    for k_idx in range(3):
        n = LOOP_NORMALS_DEFAULT[k_idx]
        ax3d.scatter([n[0]], [n[1]], [n[2]],
                     color="white", s=120, marker="*",
                     edgecolor="black", linewidth=0.7,
                     label=f"L{k_idx+1} normal" if k_idx == 0 else None)
    # Compass labels
    ax3d.text(0, 1.08, 0, "N", color="black", fontsize=11, ha="center")
    ax3d.text(1.08, 0, 0, "E", color="black", fontsize=11, ha="center")
    ax3d.text(0, -1.08, 0, "S", color="black", fontsize=11, ha="center")
    ax3d.text(-1.08, 0, 0, "W", color="black", fontsize=11, ha="center")
    ax3d.text(0, 0, 1.10, "zenith", color="black", fontsize=10, ha="center")
    ax3d.set_xlim(-1.1, 1.1); ax3d.set_ylim(-1.1, 1.1); ax3d.set_zlim(0, 1.1)
    ax3d.set_box_aspect((1, 1, 0.55))
    ax3d.view_init(elev=28, azim=-60)
    ax3d.set_xlabel("E"); ax3d.set_ylabel("N"); ax3d.set_zlabel("Up")
    ax3d.set_title('SNR over upper hemisphere\n("drop most-aligned loop")')
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])

    # ----------------------------- flat polar heatmap
    ax_polar = fig.add_subplot(1, 3, 2, projection="polar")
    # In matplotlib polar coords: theta = az (CCW from +x), r = 90 - el (so
    # the rim is the horizon, center is zenith).  We need to flip az from
    # CW-from-N to CCW-from-E:  theta = π/2 - az_rad.
    az_rad = np.deg2rad(AZ)
    theta = np.pi / 2 - az_rad
    r = 90.0 - EL                 # center is el=90, rim is el=0
    pcm = ax_polar.pcolormesh(theta, r, DROP_dB, cmap=cmap, norm=norm,
                              shading="auto")
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)              # azimuth grows CW
    ax_polar.set_rlim(0, 90)
    ax_polar.set_rticks([0, 30, 60, 90])
    ax_polar.set_yticklabels(["90°", "60°", "30°", "0°"], fontsize=8,
                             color="0.4")
    # mark the loop normals on the polar plot
    for k_idx in range(3):
        n = LOOP_NORMALS_DEFAULT[k_idx]
        # Recover its (az, el)
        el_n  = np.rad2deg(np.arctan2(n[2], np.hypot(n[0], n[1])))
        az_n  = np.rad2deg(np.arctan2(n[0], n[1])) % 360
        th_n  = np.pi / 2 - np.deg2rad(az_n)
        r_n   = 90.0 - el_n
        ax_polar.plot([th_n], [r_n], "*", color="white",
                      markeredgecolor="black", markersize=14,
                      markeredgewidth=0.8)
    ax_polar.set_title('Sky polar view: SNR(az, el)\n("drop most-aligned loop")',
                       pad=18)

    # colorbar shared by both
    cbar_ax = fig.add_axes([0.66, 0.18, 0.012, 0.65])
    fig.colorbar(pcm, cax=cbar_ax, label="SNR vs sum-of-powers (dB)")

    # ----------------------------- 1D azimuth slices
    ax1d = fig.add_subplot(1, 3, 3)
    az_line = np.linspace(0, 360, 721)
    for el_slice, color, ls in [(15, "tab:purple", "-"),
                                 (35.26, "tab:red", "-"),
                                 (60, "tab:cyan", "-")]:
        snr_sum_l, snr_coh_l, snr_drop_l = [], [], []
        for a in az_line:
            s, c, d = estimator_snrs(a, el_slice)
            snr_sum_l.append(s); snr_coh_l.append(c); snr_drop_l.append(d)
        s_arr = np.array(snr_sum_l); c_arr = np.array(snr_coh_l); d_arr = np.array(snr_drop_l)
        ax1d.plot(az_line, 10*np.log10(d_arr / s_arr.mean()),
                  color=color, lw=1.2, ls=ls,
                  label=f'el = {el_slice:.1f}°  (drop)')
    ax1d.axhline(0, color="0.5", lw=0.6, label="(a) sum of three (ref)")
    ax1d.axhline(+1.76, color="tab:red", lw=0.6, ls=":",
                 label="(b) coherent direction-formed (= +1.76 dB always)")
    for az0 in (0, 120, 240):
        ax1d.axvline(az0, color="0.7", lw=0.5, ls="--")
    ax1d.set_xlim(0, 360); ax1d.set_xlabel("azimuth (deg, CW from N)")
    ax1d.set_ylabel("SNR vs (a) sum of powers (dB)")
    ax1d.set_title("Azimuth slices at three elevations")
    ax1d.legend(loc="lower center", fontsize=8)
    ax1d.grid(True, alpha=0.3)

    fig.suptitle(
        "Three-loop array: SNR over the sky for an isotropic-noise scenario\n"
        "Color/curve = estimator (c): drop the most-aligned loop, sum the other two; "
        "Reference = (a): sum of three loop powers (= 0 dB)",
        fontsize=11, weight="bold")
    out = os.path.join(_HERE, "docs", "figures", "snr_hemisphere.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)

    # text summary
    print("\nGlobal stats:")
    print(f"  estimator (a) sum of powers: SNR = const = {SUM.mean():.4g}")
    print(f"  estimator (b) coherent      : mean SNR/SNR_sum = "
          f"{COH.mean()/SUM.mean():.3f}  ({10*np.log10(COH.mean()/SUM.mean()):+.2f} dB)")
    print(f"  estimator (c) drop one      : "
          f"  min/mean/max  = {DROP.min()/SUM.mean():.3f} / "
          f"{DROP.mean()/SUM.mean():.3f} / {DROP.max()/SUM.mean():.3f}")
    print(f"                                 ({10*np.log10(DROP.min()/SUM.mean()):+.2f} / "
          f"{10*np.log10(DROP.mean()/SUM.mean()):+.2f} / "
          f"{10*np.log10(DROP.max()/SUM.mean()):+.2f}  dB)")


if __name__ == "__main__":
    make_plot()
