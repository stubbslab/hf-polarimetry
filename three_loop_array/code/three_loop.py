#!/usr/bin/env python3
"""
three_loop.py — Three orthogonal magnetic-loop array analysis.

Geometry:
    Three identical mag-loops sit on three faces of a cube whose body
    diagonal is vertical and whose lower vertex is on the ground.  The
    loop NORMALS therefore each tilt 54.7356° (= arccos(1/√3)) from the
    zenith and are spaced 120° apart in azimuth.  Loop L1 has its normal
    pointing N + up; L2 and L3 follow at +120° and -120° in azimuth.

Coordinates:
    Lab frame is right-handed (East, North, Up).  Azimuth φ is measured
    clockwise from North (so 0° = N, 90° = E).  Elevation θ is measured
    upward from the local horizontal.

Provides:
    LOOP_NORMALS           geometry matrix N (rows = loop normals in lab frame)
    Ninv                   pre-computed N^-1 (recover lab B from loop B's)
    az_el_to_khat()        sky direction unit vector (toward source)
    perp_projector()       (I - k k^T) for B_perp = P · B_lab
    extract_complex_baseband(t, b, f0, BW)
                           narrow-band complex baseband at carrier f0
    analyze(t, B1, B2, B3, f0, BW, az0, el0)
                           full analysis -> dict with amp, phase, freq,
                           polarization, intensity in arrival plane.

A `__main__` section demonstrates the pipeline on a simulated WWV
20 MHz signal arriving from Fort Collins, CO, at the right azimuth /
elevation for a Cambridge, MA receiver.
"""

import os, sys
import numpy as np

# ----------------------- geometry --------------------------------------------

C54 = 1.0 / np.sqrt(3.0)            # cos(54.7356°)
S54 = np.sqrt(2.0 / 3.0)            # sin(54.7356°)


def loop_normals():
    """Return 3x3 matrix N whose rows are unit vectors of the three loop
    normals in the (E, N, Up) lab frame."""
    # Azimuth (deg, clockwise from N) of each normal's horizontal projection.
    # L1 -> N (0°), L2 -> +120° (toward S-SE in compass terms, 120° CW from N),
    # L3 -> -120° (toward S-SW, equivalently 240° CW from N).
    az = np.deg2rad([0.0, 120.0, -120.0])
    rows = []
    for a in az:
        rows.append([S54 * np.sin(a),     # East component
                     S54 * np.cos(a),     # North component
                     C54])                # Up component
    return np.asarray(rows, dtype=np.float64)


LOOP_NORMALS = loop_normals()
Ninv = np.linalg.inv(LOOP_NORMALS)


# ----------------------- sky / projector -------------------------------------

def az_el_to_khat(az_deg, el_deg):
    """Unit vector pointing TOWARD the source, in (E, N, Up).
    Convention: azimuth measured clockwise from North."""
    a = np.deg2rad(az_deg); e = np.deg2rad(el_deg)
    return np.array([np.cos(e) * np.sin(a),
                     np.cos(e) * np.cos(a),
                     np.sin(e)], dtype=np.float64)


def perp_projector(khat):
    """3x3 projector P = I - khat khat^T that drops the k-parallel
    component of any 3-vector."""
    k = np.asarray(khat, dtype=np.float64).reshape(3)
    return np.eye(3) - np.outer(k, k)


def perp_orthonormal_basis(khat):
    """Return two orthonormal vectors p̂, q̂ that span the plane ⊥ khat.
    p̂ is the unit vector lying in the (E,N) horizontal plane and ⊥ khat
    (so p̂ traces the "horizontal polarization axis" of the wave).
    q̂ = khat × p̂ then points generally upward — the "vertical" axis."""
    k = np.asarray(khat, dtype=np.float64).reshape(3)
    z = np.array([0.0, 0.0, 1.0])
    # If khat is straight up, fall back to East as the reference.
    if abs(np.dot(k, z)) > 0.999:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = z
    p = np.cross(k, ref)
    p /= np.linalg.norm(p)
    q = np.cross(k, p)
    q /= np.linalg.norm(q)
    return p, q


# ----------------------- complex baseband extraction -------------------------

def extract_complex_baseband(t, b, f0, BW):
    """Find the spectral peak nearest to f0 in real signal b(t), and
    return the complex baseband z(t) at the native sample rate after a
    rectangular ±BW/2 window around the detected peak.

    Returns
    -------
    z : complex ndarray, same length as b
    f_peak : float, detected peak frequency (Hz)
    snr_db : float, FFT-domain peak/noise-floor SNR in dB
    """
    b = np.asarray(b, dtype=np.float64)
    n = b.size
    sr = 1.0 / np.median(np.diff(t))
    # FFT over zero-padded length for finer-than-bin frequency resolution.
    nfft = 1 << int(np.ceil(np.log2(8 * n)))
    F = np.fft.fft(b, n=nfft)
    f = np.fft.fftfreq(nfft, 1.0 / sr)
    mag = np.abs(F)

    # Search ±BW around f0 in the positive-frequency half (real signal -> two-sided)
    mask = (f >= f0 - BW) & (f <= f0 + BW)
    if not np.any(mask):
        raise ValueError("No FFT bins fall within the search window.")
    idxs = np.flatnonzero(mask)
    k = idxs[np.argmax(mag[idxs])]
    f_peak = float(f[k])

    # SNR estimate: peak / median |F| outside ±2·BW
    far = ((np.abs(f - f_peak) > 2.0 * BW) & (np.abs(f) < sr / 2.0))
    if np.any(far):
        noise = float(np.median(mag[far]))
        snr_db = 20.0 * np.log10(mag[k] / max(noise, 1e-30))
    else:
        snr_db = float("nan")

    # Complex baseband by mixing then ±BW/2 brick-wall low-pass via FFT.
    tt = t - t[0]
    mix = np.exp(-1j * 2 * np.pi * f_peak * tt)
    Z = np.fft.fft(b * mix, n=nfft)
    fz = np.fft.fftfreq(nfft, 1.0 / sr)
    Z[np.abs(fz) > BW / 2] = 0.0
    z = np.fft.ifft(Z)[:n] * 2.0   # ×2 to recover the analytic signal amplitude
    return z.astype(np.complex128), f_peak, snr_db


# ----------------------- full analysis ---------------------------------------

def analyze(t, B1, B2, B3, f0, BW, az_deg, el_deg):
    """Run the full pipeline.

    Parameters
    ----------
    t  : ndarray, time stamps (s)
    B1, B2, B3 : ndarray, real-valued loop signals (V or A.U.)
    f0 : float, target carrier frequency (Hz)
    BW : float, search/extract bandwidth (Hz, ±BW/2 retained around peak)
    az_deg, el_deg : initial direction estimate (azimuth from N, elevation
                     above horizon).  Used both for projecting onto the
                     plane perpendicular to k̂ and for computing the
                     polarization basis (p̂, q̂).

    Returns
    -------
    out : dict with keys
        f_peak       : detected carrier frequency (Hz)
        snr_db       : FFT-peak SNR (dB)
        amp          : complex narrow-band envelope, shape (3, N)
        intensity    : |B_perp|^2(t), polarization-independent
        instant_freq : instantaneous frequency of dominant pol component (Hz)
        instant_phase: instantaneous phase (rad), unwrapped & detrended
        stokes       : (I, Q, U, V) tuple of time series
        pol_fraction : √(Q²+U²+V²) / I, time series
        ellipticity_deg : ½·arctan(V / √(Q²+U²)), in deg, time series
        position_angle_deg : ½·arctan2(U, Q), in deg, time series
        khat, p_hat, q_hat : the geometry vectors used
    """
    t = np.asarray(t, dtype=np.float64)
    if not (B1.shape == B2.shape == B3.shape == t.shape):
        raise ValueError("B1, B2, B3, t must all be the same length.")

    # 1) per-loop complex baseband at the dominant peak inside (f0 ± BW)
    zs = []
    for b in (B1, B2, B3):
        z, f_peak, snr_db = extract_complex_baseband(t, b, f0, BW)
        zs.append(z)
    z1, z2, z3 = zs
    Z_loops = np.array([z1, z2, z3])              # shape (3, N)

    # 2) recover lab-frame complex B vector at each sample
    Z_lab = Ninv @ Z_loops                         # shape (3, N)

    # 3) project onto the plane perpendicular to the assumed k̂
    khat = az_el_to_khat(az_deg, el_deg)
    P    = perp_projector(khat)
    Z_perp = P @ Z_lab                             # shape (3, N)

    # Polarization-independent intensity
    intensity = np.real(np.sum(Z_perp * np.conj(Z_perp), axis=0))

    # 4) split Z_perp onto p̂, q̂ to get the two complex polarization
    # components.
    p_hat, q_hat = perp_orthonormal_basis(khat)
    A_p = p_hat @ Z_perp                           # complex, length N
    A_q = q_hat @ Z_perp                           # complex, length N

    # 5) Stokes parameters
    I = np.abs(A_p)**2 + np.abs(A_q)**2
    Q = np.abs(A_p)**2 - np.abs(A_q)**2
    U = 2.0 * np.real(np.conj(A_p) * A_q)
    V = 2.0 * np.imag(np.conj(A_p) * A_q)
    pol_frac = np.sqrt(Q**2 + U**2 + V**2) / np.maximum(I, 1e-30)
    ellipticity = 0.5 * np.arctan2(V, np.sqrt(Q**2 + U**2))
    posn_angle  = 0.5 * np.arctan2(U, Q)

    # 6) phase / instantaneous frequency on the dominant component
    # Dominant polarization axis = unit eigenvector of the 2x2 covariance
    # over a long baseline (just the time mean of A_p*, A_q* outer products).
    C = np.array([[np.mean(np.abs(A_p)**2),                np.mean(np.conj(A_p) * A_q)],
                  [np.mean(A_p * np.conj(A_q)),            np.mean(np.abs(A_q)**2)]])
    w, V_eig = np.linalg.eigh(C)
    # eigenvector with the LARGEST eigenvalue -> dominant pol direction
    u_dom = V_eig[:, -1]
    z_dom = u_dom[0] * A_p + u_dom[1] * A_q          # complex projection
    inst_phase = np.unwrap(np.angle(z_dom))
    # remove linear drift (residual carrier-freq mismatch)
    slope, intercept = np.polyfit(t, inst_phase, 1)
    inst_phase -= slope * t + intercept
    inst_freq = (slope / (2 * np.pi)) + f_peak  # add back the carrier
    inst_freq_t = np.gradient(np.unwrap(np.angle(z_dom)), t) / (2 * np.pi) + f_peak

    return dict(
        f_peak=f_peak, snr_db=snr_db,
        z_loops=Z_loops, z_lab=Z_lab, z_perp=Z_perp,
        khat=khat, p_hat=p_hat, q_hat=q_hat,
        A_p=A_p, A_q=A_q,
        intensity=intensity,
        amp_dominant=np.abs(z_dom),
        instant_phase=inst_phase,
        instant_freq=inst_freq_t,
        instant_freq_mean=inst_freq,
        stokes=(I, Q, U, V),
        pol_fraction=pol_frac,
        ellipticity_deg=np.rad2deg(ellipticity),
        position_angle_deg=np.rad2deg(posn_angle),
    )


# ----------------------- simulator -------------------------------------------

def simulate_wwv(duration_s=2.0, sample_rate=200_000.0,
                 f_RF=20.0e6, fs_offset=0.0,
                 az_deg=255.0, el_deg=12.0,
                 pol="linear_vertical",
                 amp=1.0, snr_db=30.0,
                 faraday_rate_dps=0.0, faraday_phase0_deg=0.0,
                 amp_modulation=None,
                 seed=0):
    """Generate a synthetic three-loop dataset.

    Parameters
    ----------
    duration_s : recording duration
    sample_rate : Hz
    f_RF : the carrier frequency injected into the synthetic loops.  In a
           real KiwiSDR-style measurement we'd be at baseband already; for
           this self-contained sim we generate a band-limited waveform near
           DC representing the IF beat at the real RF.  Equivalently, we
           generate a complex baseband at offset (f_RF - f_LO) and take
           Re[..] for each loop -- but for clarity we just place a real
           tone at fs_offset and let the analysis pipeline find it.
    fs_offset : where to place the carrier within the recording's spectrum.
                If 0, place it at sample_rate/8 by default.
    az_deg, el_deg : direction the wave is coming FROM, in (az from N, el).
                     But our khat convention is "toward source" so we feed
                     the same numbers in.  See the caller.
    pol : 'linear_vertical', 'linear_horizontal', 'rcp', 'lcp', 'elliptical'
    amp : peak amplitude of B_perp (a.u.)
    snr_db : per-channel additive Gaussian noise SNR (dB)
    faraday_rate_dps : Faraday-rotation rate in degrees per second.  The
        polarization ellipse rotates in the (p̂, q̂) plane at this rate.
        For linear pol this produces classical Faraday fading on a single
        linear-pol receiver; for circular pol it just adds a common phase
        (no amplitude effect).
    faraday_phase0_deg : initial Faraday rotation angle at t=0, deg.
    amp_modulation : optional callable f(t) returning a multiplicative
        amplitude envelope to apply to the wave (e.g. np.ones_like, or a
        slowly-varying fade). If None, amplitude is constant.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate

    f_carrier = sample_rate / 8 if fs_offset == 0 else fs_offset

    # Direction from N (az) and elevation
    khat = az_el_to_khat(az_deg, el_deg)
    p_hat, q_hat = perp_orthonormal_basis(khat)

    # Build the (initial) complex polarization vector in the (p̂, q̂) basis
    if pol == "linear_vertical":
        A_p0, A_q0 = 0.0, amp
    elif pol == "linear_horizontal":
        A_p0, A_q0 = amp, 0.0
    elif pol == "rcp":
        A_p0, A_q0 = amp / np.sqrt(2), -1j * amp / np.sqrt(2)
    elif pol == "lcp":
        A_p0, A_q0 = amp / np.sqrt(2), +1j * amp / np.sqrt(2)
    elif pol == "elliptical":
        A_p0, A_q0 = 0.7 * amp, 0.3j * amp
    else:
        raise ValueError(f"unknown polarization: {pol}")

    # Faraday rotation: rotate (A_p, A_q) by Ω(t) at every instant.
    Omega = np.deg2rad(faraday_phase0_deg) + np.deg2rad(faraday_rate_dps) * t
    A_p_t = A_p0 * np.cos(Omega) - A_q0 * np.sin(Omega)
    A_q_t = A_p0 * np.sin(Omega) + A_q0 * np.cos(Omega)

    # Optional slow amplitude modulation (multipath envelope)
    if amp_modulation is not None:
        env = np.asarray(amp_modulation(t), dtype=np.float64)
        A_p_t = A_p_t * env
        A_q_t = A_q_t * env

    # Carrier
    omega = 2 * np.pi * f_carrier
    carrier = np.exp(1j * omega * t)

    # Lab-frame complex B at each instant (3-vector x N samples)
    Z_lab = (A_p_t[None, :] * p_hat[:, None] +
             A_q_t[None, :] * q_hat[:, None]) * carrier[None, :]

    # Project onto each loop normal, take Re[...] to get a real B-field
    # signal in each loop.
    B_loops = np.real(LOOP_NORMALS @ Z_lab)        # (3, N)

    # Add white Gaussian noise calibrated to the requested per-channel SNR
    sig_rms = np.sqrt(np.mean(B_loops**2, axis=1, keepdims=True))
    snr_lin = 10**(snr_db / 20.0)
    n_rms = sig_rms / snr_lin
    B_loops += n_rms * rng.standard_normal(B_loops.shape)

    return t, B_loops[0], B_loops[1], B_loops[2], dict(
        f_carrier=f_carrier, az_deg=az_deg, el_deg=el_deg, pol=pol,
        amp=amp, snr_db=snr_db,
        faraday_rate_dps=faraday_rate_dps,
        faraday_phase0_deg=faraday_phase0_deg,
    )


# ----------------------- plotting helper -------------------------------------

def plot_results(t, result, title="three-loop analysis", outpath=None):
    import matplotlib
    if outpath is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Amplitude
    ax = axes[0]
    ax.plot(t, np.sqrt(result["intensity"]), lw=0.7, color="tab:blue",
            label="|B_⊥(t)| (polarization-independent)")
    ax.plot(t, result["amp_dominant"], lw=0.7, color="tab:red", alpha=0.7,
            label="dominant pol amplitude")
    ax.set_ylabel("amplitude (a.u.)")
    ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)

    # Phase
    ax = axes[1]
    ax.plot(t, np.rad2deg(result["instant_phase"]), lw=0.7, color="tab:purple")
    ax.set_ylabel("phase residual (deg)")
    ax.grid(True, alpha=0.3)

    # Instantaneous frequency
    ax = axes[2]
    ax.plot(t, result["instant_freq"], lw=0.7, color="tab:green")
    ax.axhline(result["instant_freq_mean"], color="black", lw=0.5, ls="--",
               alpha=0.6, label=f"mean = {result['instant_freq_mean']:.2f} Hz")
    ax.set_ylabel("inst. freq (Hz)")
    ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)

    # Polarization state
    ax = axes[3]
    ax.plot(t, result["ellipticity_deg"], lw=0.7, color="tab:orange",
            label="ellipticity (deg)")
    ax.plot(t, result["position_angle_deg"], lw=0.7, color="tab:cyan",
            label="position angle (deg)")
    ax.plot(t, 100 * result["pol_fraction"], lw=0.7, color="black", alpha=0.5,
            label="pol fraction (%)")
    ax.set_ylabel("polarization (deg / %)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if outpath is not None:
        fig.savefig(outpath, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {outpath}")
    else:
        plt.show()


# ----------------------- demo / validation -----------------------------------

def main():
    # Cambridge MA at (42.37°N, 71.11°W); Fort Collins CO at (40.59°N, 105.08°W).
    # Great-circle bearing FROM Cambridge TO Fort Collins ≈ 273° (W to slightly N).
    # Coming from Fort Collins, the wave arrives at Cambridge from approximately
    # the WEST.  Equivalently khat (toward source) points west-ish.  For the
    # arrival angle from N (clockwise convention) the source-direction azimuth
    # is ~273°.  With a single-hop F2 reflection at ~300 km height:
    #     elevation = arctan(300 / (2840/2)) ≈ 12°
    AZ = 273.0
    EL = 12.0

    print("Synthesizing WWV-like 20 MHz signal arriving from "
          f"az={AZ}°, el={EL}° with linear_vertical polarization.")
    t, B1, B2, B3, truth = simulate_wwv(
        duration_s=2.0, sample_rate=200_000.0,
        f_RF=20.0e6, fs_offset=0.0,                  # carrier at sample_rate/8
        az_deg=AZ, el_deg=EL,
        pol="linear_vertical",
        amp=1.0, snr_db=40.0,
        seed=42,
    )
    print(f"  truth:  f_carrier = {truth['f_carrier']:.2f} Hz "
          f"(IF representation of {truth['f_carrier']:.0f} Hz tone)")

    # Use a search bandwidth wide enough to catch a slightly-off carrier
    # but narrow enough to reject noise.
    BW = 2000.0
    print(f"Analyzing with f0 = {truth['f_carrier']} Hz, BW = ±{BW/2:.0f} Hz, "
          f"initial guess az={AZ}°, el={EL}°.")
    res = analyze(t, B1, B2, B3, truth['f_carrier'], BW, AZ, EL)

    print(f"  found f_peak = {res['f_peak']:.4f} Hz "
          f"(error = {res['f_peak'] - truth['f_carrier']:+.4f} Hz)")
    print(f"  FFT SNR ≈ {res['snr_db']:.1f} dB")
    print(f"  median pol fraction = {np.median(res['pol_fraction']):.4f}")
    print(f"  median ellipticity  = {np.median(res['ellipticity_deg']):+.2f}° "
          "(expect 0° for linear, ±45° for circular)")
    print(f"  median position angle = "
          f"{np.median(res['position_angle_deg']):+.2f}° (relative to p̂)")

    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "figures", "validation_20MHz.png")
    plot_results(t, res, title=f"Validation — WWV-like 20 MHz, az={AZ}° el={EL}°",
                 outpath=out_png)


if __name__ == "__main__":
    main()
