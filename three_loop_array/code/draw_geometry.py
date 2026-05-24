#!/usr/bin/env python3
"""Generate the geometry illustration figures used in the report."""

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, _HERE)

import os as _os
_os.environ.setdefault("MPLCONFIGDIR",
                       _os.path.join(_HERE, "..", ".mpl_cache"))
_os.makedirs(_os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from three_loop import LOOP_NORMALS, az_el_to_khat, perp_orthonormal_basis

OUT = os.path.join(_HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)


def fig_cube_and_normals():
    fig = plt.figure(figsize=(7, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    # Draw a unit cube standing on its (-1,-1,-1)/√3 vertex with body
    # diagonal vertical.  The simplest description: define cube vertices
    # in a body-frame (axis-aligned cube), then rotate so (1,1,1)/√3 -> +z.
    cube_pts = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
                        dtype=np.float64)
    # Rotation that maps (1,1,1)/√3 -> (0,0,1)
    a = np.array([1, 1, 1.0]) / np.sqrt(3); b = np.array([0, 0, 1.0])
    v = np.cross(a, b); s = np.linalg.norm(v); c = np.dot(a, b)
    Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + Vx + Vx @ Vx * ((1 - c) / s**2 if s != 0 else 0)
    cube_pts_rot = (R @ cube_pts.T).T
    cube_pts_rot -= cube_pts_rot.min(axis=0)   # vertex on the ground

    edges = [(i, j) for i in range(8) for j in range(i+1, 8)
             if np.sum(np.abs(cube_pts[i] - cube_pts[j])) == 1]
    for i, j in edges:
        x = [cube_pts_rot[i, 0], cube_pts_rot[j, 0]]
        y = [cube_pts_rot[i, 1], cube_pts_rot[j, 1]]
        z = [cube_pts_rot[i, 2], cube_pts_rot[j, 2]]
        ax.plot(x, y, z, color="0.4", lw=1)

    # Lower vertex (origin) + body-diagonal in red
    o = cube_pts_rot[np.argmin(cube_pts_rot[:, 2])]
    top = cube_pts_rot[np.argmax(cube_pts_rot[:, 2])]
    ax.plot(*zip(o, top), color="tab:red", lw=2, label="vertical body diagonal")

    # The three loop normals as arrows from the origin
    colors = ["tab:blue", "tab:green", "tab:orange"]
    labels = ["L1 (N+up)", "L2 (+120°)", "L3 (-120°)"]
    for k in range(3):
        n = LOOP_NORMALS[k]
        ax.quiver(o[0], o[1], o[2], n[0], n[1], n[2],
                  length=0.95, color=colors[k], lw=2,
                  arrow_length_ratio=0.12, label=labels[k])

    # Compass labels
    ax.text(o[0] + 1.1, o[1], o[2], "E", fontsize=11, color="0.3")
    ax.text(o[0], o[1] + 1.1, o[2], "N", fontsize=11, color="0.3")
    ax.text(o[0], o[1], o[2] + 1.5, "Up", fontsize=11, color="0.3")

    ax.set_xlim(-0.6, 1.4); ax.set_ylim(-0.6, 1.4); ax.set_zlim(0, 1.7)
    ax.set_xlabel("E"); ax.set_ylabel("N"); ax.set_zlabel("Up")
    ax.set_title("Three-loop array geometry — cube on a vertex,\n"
                 "loop normals tilt 54.74° from zenith, 120° apart in azimuth")
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=22, azim=-60)
    fig.tight_layout()
    out = os.path.join(OUT, "geometry_cube.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_horizontal_normals():
    """Top-down view showing the 120° azimuth spacing of the three normals."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    az = np.deg2rad([0, 120, -120])
    colors = ["tab:blue", "tab:green", "tab:orange"]
    labels = ["L1: az 0° (N+up)",
              "L2: az 120° (SE+up)",
              "L3: az -120° (SW+up)"]
    R = 1.0
    for a, c, lab in zip(az, colors, labels):
        x = R * np.sin(a); y = R * np.cos(a)
        ax.plot([0, x], [0, y], color=c, lw=2)
        ax.plot(x, y, "o", color=c, ms=10)
        ax.text(1.10 * x, 1.10 * y, lab, color=c, fontsize=10,
                ha="center", va="center")
    # Compass
    ax.annotate("N", xy=(0, 1.4), xytext=(0, 1.1),
                arrowprops=dict(arrowstyle="->", color="black"),
                ha="center", fontsize=12)
    ax.annotate("E", xy=(1.4, 0), xytext=(1.1, 0),
                arrowprops=dict(arrowstyle="->", color="black"),
                ha="center", fontsize=12)
    circle = plt.Circle((0, 0), 1.0, fill=False, color="0.7", lw=0.5)
    ax.add_patch(circle)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Horizontal projection of loop normals\n"
                 "(elevation of every normal = 35.26° above horizontal)")
    fig.tight_layout()
    out = os.path.join(OUT, "normals_top_down.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_az_el_convention():
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    # Show az/el convention with a small sky map
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="0.7", lw=0.5)   # horizon
    # Cardinal labels
    for ang, lab in [(np.pi/2, "N"), (0, "E"), (-np.pi/2, "S"), (np.pi, "W")]:
        ax.text(1.15*np.cos(ang), 1.15*np.sin(ang), lab,
                ha="center", va="center", fontsize=12)
    # Show example arrival direction: az=273° (NW), el=12°
    az = 273.0; el = 12.0
    az_rad = np.deg2rad(az); el_rad = np.deg2rad(el)
    r = np.cos(el_rad)
    x = r * np.sin(az_rad); y = r * np.cos(az_rad)
    ax.plot([0, x], [0, y], color="tab:red", lw=2)
    ax.plot(x, y, "o", color="tab:red", ms=10)
    ax.text(x*1.05, y*1.05, f"az={az}°\nel={el}°",
            color="tab:red", fontsize=10, ha="left", va="bottom")
    # Annotate az direction
    arc_th = np.linspace(np.pi/2, np.pi/2 - az_rad, 60)
    ax.plot(0.4*np.cos(arc_th), 0.4*np.sin(arc_th), color="tab:blue", lw=1)
    ax.text(0.45*np.cos(arc_th[30]), 0.45*np.sin(arc_th[30]),
            "az (N→E→S)", color="tab:blue", fontsize=9)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Azimuth / elevation convention\n"
                 "az measured CW from N; el above horizontal")
    fig.tight_layout()
    out = os.path.join(OUT, "az_el_convention.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_perp_basis():
    """Diagram of the polarization basis (p̂, q̂) at a sample direction."""
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    az = 273.0; el = 12.0
    k = az_el_to_khat(az, el)
    p, q = perp_orthonormal_basis(k)

    o = np.zeros(3)
    L = 1.0
    # Lab axes (light grey)
    ax.quiver(*o, L, 0, 0, color="0.7", arrow_length_ratio=0.05)
    ax.text(L+0.05, 0, 0, "E", color="0.5")
    ax.quiver(*o, 0, L, 0, color="0.7", arrow_length_ratio=0.05)
    ax.text(0, L+0.05, 0, "N", color="0.5")
    ax.quiver(*o, 0, 0, L, color="0.7", arrow_length_ratio=0.05)
    ax.text(0, 0, L+0.05, "Up", color="0.5")

    # k, p, q
    ax.quiver(*o, *k, color="tab:red", lw=2.5, arrow_length_ratio=0.1, length=L)
    ax.text(*(k*1.05), "k̂  (toward source)", color="tab:red", fontsize=10)

    ax.quiver(*o, *p, color="tab:blue", lw=2.5, arrow_length_ratio=0.1, length=L)
    ax.text(*(p*1.10), "p̂  (horizontal pol axis)", color="tab:blue", fontsize=10)

    ax.quiver(*o, *q, color="tab:green", lw=2.5, arrow_length_ratio=0.1, length=L)
    ax.text(*(q*1.10), "q̂  (vertical pol axis)", color="tab:green", fontsize=10)

    ax.set_xlim(-1, 1.3); ax.set_ylim(-1, 1.3); ax.set_zlim(-0.2, 1.3)
    ax.set_xlabel("E"); ax.set_ylabel("N"); ax.set_zlabel("Up")
    ax.set_title(f"Polarization basis (p̂, q̂) for arrival at az={az}°, el={el}°")
    ax.view_init(elev=22, azim=-60)
    fig.tight_layout()
    out = os.path.join(OUT, "perp_basis.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_cube_and_normals()
    fig_horizontal_normals()
    fig_az_el_convention()
    fig_perp_basis()
