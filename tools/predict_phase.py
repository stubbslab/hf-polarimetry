#!/usr/bin/env python3
"""tools/predict_phase.py

Phase-prediction analysis on KiwiSDR IQ recordings of WWV.

Given an IQ .wav of a WWV carrier (Kiwi format: 2-channel int16 at 20 kHz
sample rate, complex baseband at the receiver's detuned offset), this
script:

  1. extracts the complex baseband at the dominant beat tone (~2 kHz);
  2. unwraps the phase track and removes the residual carrier slope;
  3. finds 'high-SNR windows' where amplitude is stable and the phase
     tracker is solid;
  4. runs three predictors over a sweep of lookahead times tau:
        - constant-phase                 (zero-th order, do-nothing baseline)
        - linear extrapolation           (first-order; nails out fixed Doppler)
        - 2-state Kalman filter          (recursive; adaptive to drifting Doppler)
  5. plots prediction-error vs tau on a log-log scale, alongside the
     descriptive structure function D(tau) for context.

Output: <input>_predict.png and <input>_predict.json next to the input
file.

Usage
-----
    python3 tools/predict_phase.py path/to/wwv_iq.wav
    python3 tools/predict_phase.py path/to/wwv_iq.wav \\
            --window-len 1.0 --tau-max 0.1 --kalman-q 1e2

The script tries to be parameter-light: the defaults work for any 90-s
WWV IQ recording at 20 kHz sample rate.

Notes on conventions
--------------------
- The phase-prediction structure function D_pred(tau) is defined as
      D_pred(tau) = < [phi(t+tau) - phi_hat(t+tau | t)]^2 >
  averaged over all valid prediction epochs t in the high-SNR windows.
  This is the metric an adaptive-correction system would care about.
- D_pred(tau) <= D(tau) by construction (a predictor that knew nothing
  except phi(t) reduces to constant-phase, which gives D(tau) exactly).
- For an ergodic stationary process with known second-order statistics,
  the optimal predictor's D_pred(tau) sets a hard lower bound on what
  any closed-loop correction can achieve.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------- I/O

def load_kiwi_iq_wav(path: str) -> Tuple[np.ndarray, float]:
    """Load a Kiwi IQ .wav file (2-ch int16, real+imag) and return
    (z, sr) where z is complex64 baseband and sr is the file sample
    rate in Hz."""
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 2 or w.getsampwidth() != 2:
            raise ValueError(
                f"expected 2-ch int16 WAV; got "
                f"{w.getnchannels()}-ch {8*w.getsampwidth()}-bit")
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    d = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, 2)
    z = (d[:, 0] + 1j * d[:, 1]).astype(np.complex64)
    return z, float(sr)


# ----------------------------------------------------- carrier extraction

def find_carrier_freq(z: np.ndarray, sr: float,
                      drop_first_s: float = 1.0,
                      search_lo: float = 1500.0,
                      search_hi: float = 2500.0) -> float:
    """FFT-based search for the dominant tone in |f| ∈ [lo, hi].
    Allows the carrier to appear at either positive OR negative f
    (Kiwi detuning convention varies)."""
    head = int(drop_first_s * sr)
    z = z[head:]
    n_fft = 1 << 18
    n_fft = min(n_fft, z.size)
    Z = np.fft.fft(z[:n_fft])
    f = np.fft.fftfreq(n_fft, 1.0 / sr)
    mag = np.abs(Z)
    mask = (np.abs(f) >= search_lo) & (np.abs(f) <= search_hi)
    if not np.any(mask):
        raise RuntimeError(
            f"no FFT bins in search window |f|∈[{search_lo},{search_hi}] Hz")
    idxs = np.flatnonzero(mask)
    return float(f[idxs[np.argmax(mag[idxs])]])


def mix_to_dc(z: np.ndarray, sr: float, f_carrier: float) -> np.ndarray:
    """Mix down to DC at f_carrier; do not low-pass (caller can decimate)."""
    t = np.arange(z.size) / sr
    return (z * np.exp(-1j * 2 * np.pi * f_carrier * t)).astype(np.complex64)


def amplitude_phase(z_dc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Magnitude (linear) and unwrapped phase (radians) of complex z_dc."""
    A = np.abs(z_dc).astype(np.float32)
    phi = np.unwrap(np.angle(z_dc)).astype(np.float64)
    return A, phi


def detrend_linear(t: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, float]:
    """Remove the residual carrier-frequency mismatch (a linear-in-time
    phase ramp).  Returns (phi_detrended, slope_rad_per_s)."""
    slope, intercept = np.polyfit(t, phi, 1)
    return phi - (slope * t + intercept), float(slope)


# -------------------------------------------------------- segment finder

@dataclass
class Segment:
    i_start: int
    i_end: int          # exclusive
    median_A: float
    rms_A: float
    duration_s: float


def find_high_snr_segments(A: np.ndarray, sr: float,
                           min_duration_s: float = 1.0,
                           amp_threshold_quantile: float = 0.5,
                           amp_relative_var_max: float = 0.5,
                           smooth_s: float = 0.1
                           ) -> List[Segment]:
    """Identify contiguous time intervals where the *smoothed*
    amplitude is above the ``amp_threshold_quantile`` of the file's
    smoothed-amplitude distribution.

    The amplitude is first low-pass filtered with a moving-average
    window of length ``smooth_s`` so that audio-rate AM (the WWV time
    code, BCD subcarriers, and short fades from scintillation) doesn't
    fragment otherwise-good stretches into thousands of micro-runs.

    Each surviving run is further checked: its RMS/median amplitude
    must be below ``amp_relative_var_max`` (default 0.5).
    """
    A = np.asarray(A, dtype=np.float64)
    n_smooth = max(1, int(round(smooth_s * sr)))
    if n_smooth > 1:
        kernel = np.ones(n_smooth) / n_smooth
        A_smooth = np.convolve(A, kernel, mode="same")
    else:
        A_smooth = A
    thr = float(np.quantile(A_smooth, amp_threshold_quantile))
    above = A_smooth > thr
    diff = np.diff(above.astype(np.int8))
    starts = np.flatnonzero(diff == 1) + 1
    ends   = np.flatnonzero(diff == -1) + 1
    if above[0]:
        starts = np.concatenate([[0], starts])
    if above[-1]:
        ends = np.concatenate([ends, [A.size]])
    segs = []
    min_n = int(min_duration_s * sr)
    for i_s, i_e in zip(starts, ends):
        if i_e - i_s < min_n:
            continue
        A_seg = A_smooth[i_s:i_e]
        med = float(np.median(A_seg))
        rms = float(np.std(A_seg))
        if med > 0 and rms / med > amp_relative_var_max:
            continue
        segs.append(Segment(i_start=int(i_s), i_end=int(i_e),
                            median_A=med, rms_A=rms,
                            duration_s=(i_e - i_s) / sr))
    return segs


# ----------------------------------------------------------- predictors

def predict_constant(phi: np.ndarray, sr: float, taus: np.ndarray,
                     fit_window_s: float = 0.0) -> np.ndarray:
    """No prediction: phi_hat(t+tau) = phi(t).  Returns a (n_t, n_tau)
    array of squared prediction errors averaged over t.  Actually we
    return the mean squared error per tau (i.e. D_pred(tau))."""
    n = phi.size
    out = np.empty(taus.size, dtype=np.float64)
    for k, tau in enumerate(taus):
        m = int(round(tau * sr))
        if m <= 0 or m >= n:
            out[k] = np.nan; continue
        diff = phi[m:] - phi[:-m]
        out[k] = float(np.mean(diff * diff))
    return out


def predict_linear(phi: np.ndarray, sr: float, taus: np.ndarray,
                   fit_window_s: float = 0.05) -> np.ndarray:
    """Linear extrapolation: at each epoch t0, fit a line to
    phi[t0-fit_window_s : t0] and extrapolate to t0+tau.

    Implemented in vectorised form via a moving slope/intercept
    computation.  fit_window_s = 50 ms by default — long enough to
    reduce noise contribution but short enough to track drifting
    Doppler.
    """
    n = phi.size
    n_fit = int(round(fit_window_s * sr))
    if n_fit < 8:
        raise ValueError("fit window too short")
    # Vectorised running OLS:  slope(t0) = sum(x*y - <x><y>) / sum(x^2 - <x>^2)
    # where x is the relative time within the window.
    t = np.arange(n_fit, dtype=np.float64) / sr
    x_mean = float(t.mean())
    x_dev = t - x_mean
    Sxx = float((x_dev * x_dev).sum())

    # Use scipy uniform_filter for the mean over the trailing window;
    # numpy convolution is equivalent and avoids the dependency.
    kernel = np.ones(n_fit) / n_fit
    phi_mean = np.convolve(phi, kernel, mode="valid")    # length n - n_fit + 1
    # Cross-product sum: convolve phi with x_dev (reversed because of
    # convolution definition) over each window.
    cross = np.convolve(phi, x_dev[::-1], mode="valid")
    slope = cross / Sxx
    # Intercept at end-of-window (t = t[-1] = (n_fit-1)/sr from the
    # window's start).  For convenience we want phi_hat at t0 = window
    # end:  phi_at_t0 = phi_mean + slope * (t[-1] - x_mean)
    t_end = t[-1]
    phi_at_t0 = phi_mean + slope * (t_end - x_mean)
    # At sample index i in `slope`, the corresponding t0 in the original
    # phi array is i + n_fit - 1.
    base_idx = np.arange(slope.size) + (n_fit - 1)

    out = np.empty(taus.size, dtype=np.float64)
    for k, tau in enumerate(taus):
        m = int(round(tau * sr))
        if m <= 0:
            out[k] = np.nan; continue
        # We need phi[base_idx + m] for m within bounds
        valid = base_idx + m < n
        if not np.any(valid):
            out[k] = np.nan; continue
        bi = base_idx[valid]
        phi_true = phi[bi + m]
        # Predicted phase at t0 + tau: phi_at_t0 + slope * tau
        phi_pred = phi_at_t0[valid] + slope[valid] * tau
        diff = phi_true - phi_pred
        out[k] = float(np.mean(diff * diff))
    return out


def predict_kalman(phi: np.ndarray, sr: float, taus: np.ndarray,
                   q_omega: float = 1e3,
                   r_phi: Optional[float] = None) -> np.ndarray:
    """2-state Kalman filter on (phi, omega).  State propagation:
        phi_{k+1} = phi_k + omega_k * dt + noise
        omega_{k+1} = omega_k + process noise
    Measurement: z = phi (the unwrapped, detrended phase track).

    Returns D_pred(tau) just like the other predictors.

    q_omega controls the assumed process-noise variance on omega
    (in (rad/s)^2 / s).  Larger -> faster Kalman tracks.

    r_phi is the measurement-noise variance on phi, in rad^2.  If None,
    estimated from short-lag phase differences as
    sigma_phi^2 ≈ 0.5 * E[(phi[i+1] - phi[i])^2] * sr.  (The factor 1/2
    accounts for the differencing variance doubling.)
    """
    n = phi.size
    dt = 1.0 / sr
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = np.array([[q_omega * dt**3 / 3, q_omega * dt**2 / 2],
                  [q_omega * dt**2 / 2, q_omega * dt]])
    H = np.array([[1.0, 0.0]])
    if r_phi is None:
        # Crude noise estimate
        d1 = np.diff(phi)
        r_phi = max(0.5 * float(np.median(d1 * d1)), 1e-12)

    # State and covariance arrays at every step (for picking off epochs)
    x_post = np.zeros((n, 2))
    P_post = np.zeros((n, 2, 2))
    x = np.array([phi[0], 0.0])
    P = np.array([[r_phi, 0.0], [0.0, q_omega]])
    x_post[0] = x
    P_post[0] = P
    for k in range(1, n):
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q
        # Update
        y = phi[k] - (H @ x)[0]
        S = (H @ P @ H.T)[0, 0] + r_phi
        K = (P @ H.T).flatten() / S
        x = x + K * y
        P = P - np.outer(K, H @ P)
        x_post[k] = x
        P_post[k] = P

    # Now: at each epoch t0, predict ahead by tau using the F^m operator
    out = np.empty(taus.size, dtype=np.float64)
    for k_tau, tau in enumerate(taus):
        m = int(round(tau * sr))
        if m <= 0 or m >= n:
            out[k_tau] = np.nan; continue
        # Closed-form propagation: phi(t+tau) ≈ phi(t) + omega(t) * tau
        phi_pred = x_post[:n - m, 0] + x_post[:n - m, 1] * tau
        phi_true = phi[m:m + (n - m)]
        diff = phi_true - phi_pred
        out[k_tau] = float(np.mean(diff * diff))
    return out


def descriptive_structure_function(phi: np.ndarray, sr: float,
                                   taus: np.ndarray) -> np.ndarray:
    """The classical phase structure function D(tau) = <[phi(t+tau)-phi(t)]^2>."""
    n = phi.size
    out = np.empty(taus.size, dtype=np.float64)
    for k, tau in enumerate(taus):
        m = int(round(tau * sr))
        if m <= 0 or m >= n:
            out[k] = np.nan; continue
        diff = phi[m:] - phi[:-m]
        out[k] = float(np.mean(diff * diff))
    return out


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Kiwi IQ .wav file")
    p.add_argument("--out-png", default=None,
                   help="output PNG path (default: <input>_predict.png)")
    p.add_argument("--out-json", default=None,
                   help="output JSON summary path "
                        "(default: <input>_predict.json)")
    p.add_argument("--carrier-search-lo", type=float, default=1500.0,
                   help="Hz; FFT search window lower bound for the beat tone")
    p.add_argument("--carrier-search-hi", type=float, default=2500.0)
    p.add_argument("--min-segment-s", type=float, default=1.0,
                   help="minimum duration of a high-SNR window (s)")
    p.add_argument("--amp-quantile", type=float, default=0.5,
                   help="amplitude threshold quantile (0..1) on the "
                        "*smoothed* amplitude")
    p.add_argument("--amp-rel-var-max", type=float, default=0.5,
                   help="reject windows whose smoothed amplitude "
                        "RMS/median > this")
    p.add_argument("--amp-smooth-s", type=float, default=0.1,
                   help="moving-average window for amplitude "
                        "smoothing (s) before thresholding")
    p.add_argument("--tau-min", type=float, default=1e-3,
                   help="lookahead tau lower bound (s)")
    p.add_argument("--tau-max", type=float, default=0.2,
                   help="lookahead tau upper bound (s)")
    p.add_argument("--n-tau", type=int, default=40,
                   help="number of tau samples (logspace)")
    p.add_argument("--linear-fit-window-s", type=float, default=0.05,
                   help="fitting window for the linear predictor (s)")
    p.add_argument("--kalman-q-omega", type=float, default=1e3,
                   help="Kalman process-noise on omega ((rad/s)^2 / s)")
    args = p.parse_args()

    print(f"Loading {args.input}")
    z, sr = load_kiwi_iq_wav(args.input)
    print(f"  {z.size:,} complex samples at {sr:.0f} Hz "
          f"({z.size/sr:.1f} s)")

    f_c = find_carrier_freq(z, sr,
                            search_lo=args.carrier_search_lo,
                            search_hi=args.carrier_search_hi)
    print(f"  carrier beat frequency: {f_c:.3f} Hz")

    z_dc = mix_to_dc(z, sr, f_c)
    A, phi = amplitude_phase(z_dc)

    # NB: do NOT globally detrend.  Deep fades introduce huge unwrap
    # errors that contaminate any global linear fit, and the prediction
    # algorithms care about *local* phase behaviour anyway.  We instead
    # detrend per-segment (below) so each high-SNR window has its own
    # zero-mean, residual-slope-removed phase track.
    print("Finding high-SNR segments…")
    segs = find_high_snr_segments(
        A, sr, min_duration_s=args.min_segment_s,
        amp_threshold_quantile=args.amp_quantile,
        amp_relative_var_max=args.amp_rel_var_max,
        smooth_s=args.amp_smooth_s)
    if not segs:
        print("  NO segments matched the SNR criteria; relax the thresholds.",
              file=sys.stderr)
        sys.exit(1)
    print(f"  found {len(segs)} segments, total "
          f"{sum(s.duration_s for s in segs):.1f} s of usable data")
    for k, s in enumerate(segs[:8]):
        print(f"    segment {k:2d}: t = {s.i_start/sr:6.2f}-{s.i_end/sr:6.2f} s "
              f"(dur {s.duration_s:5.2f} s), median A = {s.median_A:.2g}, "
              f"rel var = {s.rms_A/s.median_A:.3f}")

    # Define lookahead grid
    taus = np.geomspace(args.tau_min, args.tau_max, args.n_tau)

    # Run the predictors per segment, then average D_pred(tau) across
    # segments weighted by segment length
    n_tau = taus.size
    Dpred_const  = np.zeros(n_tau)
    Dpred_linear = np.zeros(n_tau)
    Dpred_kalman = np.zeros(n_tau)
    Ddesc        = np.zeros(n_tau)
    weight       = np.zeros(n_tau)

    per_segment_slopes_hz = []
    for s in segs:
        # Re-unwrap from raw IQ within the segment (avoids unwrap errors
        # accumulated from deep fades elsewhere in the file).
        z_dc_seg = z_dc[s.i_start:s.i_end]
        phi_s = np.unwrap(np.angle(z_dc_seg)).astype(np.float64)
        # Record the per-segment linear slope (residual Doppler/offset)
        # for diagnostic purposes — but do NOT subtract it from phi_s.
        # The predictors operate on the raw track that an operational
        # closed-loop system would see; subtracting the slope before
        # comparison would unfairly help the constant-phase predictor.
        t_seg = np.arange(phi_s.size) / sr
        slope_rad_per_s, _ = np.polyfit(t_seg, phi_s, 1)
        per_segment_slopes_hz.append(float(slope_rad_per_s) / (2 * np.pi))
        n_s = phi_s.size
        dD_const  = predict_constant(phi_s, sr, taus)
        dD_lin    = predict_linear(phi_s, sr, taus,
                                    fit_window_s=args.linear_fit_window_s)
        dD_kal    = predict_kalman(phi_s, sr, taus,
                                    q_omega=args.kalman_q_omega)
        dD_desc   = descriptive_structure_function(phi_s, sr, taus)
        m = int(round(args.tau_max * sr))
        n_eff = max(n_s - m, 0)
        for arr, dst in [(dD_const, Dpred_const),
                         (dD_lin,   Dpred_linear),
                         (dD_kal,   Dpred_kalman),
                         (dD_desc,  Ddesc)]:
            ok = np.isfinite(arr)
            dst[ok] += arr[ok] * n_eff
        weight += n_eff * np.isfinite(dD_const)

    weight = np.where(weight > 0, weight, np.nan)
    Dpred_const  /= weight
    Dpred_linear /= weight
    Dpred_kalman /= weight
    Ddesc        /= weight
    rms_const  = np.sqrt(Dpred_const)
    rms_linear = np.sqrt(Dpred_linear)
    rms_kalman = np.sqrt(Dpred_kalman)
    rms_desc   = np.sqrt(Ddesc)

    if per_segment_slopes_hz:
        slopes = np.array(per_segment_slopes_hz)
        print(f"\nPer-segment phase slopes (residual carrier in Hz):")
        print(f"  mean {slopes.mean():+.4f} ± {slopes.std():.4f} Hz "
              f"(median {np.median(slopes):+.4f}, range "
              f"{slopes.min():+.4f} … {slopes.max():+.4f})")
    print("\nRMS prediction error (rad) at selected lookaheads:")
    print(f"  {'tau (ms)':>8}  {'const':>8}  {'linear':>8}  {'kalman':>8}  "
          f"{'D(tau)':>8}")
    for tau in [0.001, 0.005, 0.020, 0.050, 0.100, 0.200]:
        if tau > taus[-1]: continue
        i = int(np.argmin(np.abs(taus - tau)))
        print(f"  {1000*taus[i]:8.2f}  {rms_const[i]:8.4f}  "
              f"{rms_linear[i]:8.4f}  {rms_kalman[i]:8.4f}  "
              f"{rms_desc[i]:8.4f}")

    # ---------- output paths
    base, _ = os.path.splitext(args.input)
    out_png = args.out_png or (base + "_predict.png")
    out_json = args.out_json or (base + "_predict.json")

    # ---------- plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.loglog(1e3 * taus, rms_desc,   "k:", lw=1.5, label="D(τ) (descriptive)")
    ax.loglog(1e3 * taus, rms_const,  "C0",  lw=1.5,
              label="constant-phase (do nothing)")
    ax.loglog(1e3 * taus, rms_linear, "C1",  lw=1.5,
              label=f"linear extrap. (fit {1e3*args.linear_fit_window_s:.0f} ms)")
    ax.loglog(1e3 * taus, rms_kalman, "C2",  lw=1.5,
              label=f"Kalman (q_omega={args.kalman_q_omega:.0g})")
    ax.set_xlabel("look-ahead τ (ms)")
    ax.set_ylabel("RMS prediction error (rad)")
    ax.set_title(f"Phase predictability — {os.path.basename(args.input)}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    # Reference line: 0.5 rad threshold ("good" for adaptive correction)
    ax.axhline(0.5, color="0.6", lw=0.8, ls="--")
    ax.text(taus[-1]*1e3, 0.5, " 0.5 rad goal",
            va="center", ha="right", color="0.4", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out_png}")

    # ---------- json
    summary = {
        "input": os.path.abspath(args.input),
        "sample_rate_Hz": sr,
        "carrier_beat_Hz": f_c,
        "per_segment_slopes_Hz": per_segment_slopes_hz,
        "n_segments": len(segs),
        "total_duration_s": float(sum(s.duration_s for s in segs)),
        "linear_fit_window_s": args.linear_fit_window_s,
        "kalman_q_omega": args.kalman_q_omega,
        "amp_quantile": args.amp_quantile,
        "amp_rel_var_max": args.amp_rel_var_max,
        "taus_s": taus.tolist(),
        "rms_pred_constant_rad": rms_const.tolist(),
        "rms_pred_linear_rad":   rms_linear.tolist(),
        "rms_pred_kalman_rad":   rms_kalman.tolist(),
        "rms_descriptive_rad":   rms_desc.tolist(),
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
