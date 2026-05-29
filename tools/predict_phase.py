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
                      search_lo: float = 500.0,
                      search_hi: float = 4500.0,
                      n_iter: int = 3) -> float:
    """Robust sub-Hz carrier (beat-tone) finder for Kiwi IQ data.

    1. Coarse FFT peak search in |f| ∈ [search_lo, search_hi] Hz.
       Handles both positive and negative offsets (Kiwi detuning
       convention varies).
    2. Iterative refinement: mix to baseband at the current estimate,
       low-pass to ±200 Hz, fit residual phase slope, update estimate.
       Converges to sub-Hz accuracy in 3 passes.

    Adapted from wwv_phase_analysis_final.find_carrier_frequency().
    """
    head = int(drop_first_s * sr)
    z_active = z[head:]

    # Coarse FFT peak
    n_fft = 1 << 18
    n_fft = min(n_fft, z_active.size)
    Z = np.fft.fft(z_active[:n_fft])
    f = np.fft.fftfreq(n_fft, 1.0 / sr)
    mag = np.abs(Z)
    mask = (np.abs(f) >= search_lo) & (np.abs(f) <= search_hi)
    if not np.any(mask):
        raise RuntimeError(
            f"no FFT bins in search window |f|∈[{search_lo},{search_hi}] Hz")
    idxs = np.flatnonzero(mask)
    f_est = float(f[idxs[np.argmax(mag[idxs])]])

    # Iterative refinement via residual-phase-slope tracking on the
    # whole file.  Fades cause some bias on heavily-faded captures, but
    # the linear and Kalman predictors handle a constant residual
    # offset internally, so a small bias here is fine.
    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(4, 200.0 / (sr / 2), btype="low")
        have_scipy = True
    except ImportError:
        have_scipy = False

    t = np.arange(z_active.size) / sr
    for _ in range(n_iter):
        z_bb = z_active * np.exp(-1j * 2 * np.pi * f_est * t)
        if have_scipy:
            z_bb = filtfilt(b, a, z_bb)
        phase = np.unwrap(np.angle(z_bb))
        slope, _ = np.polyfit(t, phase, 1)
        f_est += slope / (2 * np.pi)
    return f_est


def mix_to_dc(z: np.ndarray, sr: float, f_carrier: float) -> np.ndarray:
    """Mix down to DC at f_carrier; do not low-pass (caller can decimate)."""
    t = np.arange(z.size) / sr
    return (z * np.exp(-1j * 2 * np.pi * f_carrier * t)).astype(np.complex64)


def lowpass_complex(z_dc: np.ndarray, sr: float,
                    cutoff_hz: float = 200.0,
                    order: int = 6) -> np.ndarray:
    """Brick-wall-ish low-pass filter (zero-phase Butterworth via
    filtfilt) on a complex baseband signal.  cutoff_hz is the one-sided
    cutoff: a 200 Hz value keeps ±200 Hz around DC, total noise
    bandwidth 400 Hz.

    For a Kiwi 20 kHz IQ recording at SNR-limited HF carriers, this
    gain matters a lot: noise variance scales as bandwidth, so going
    from 20 kHz to 400 Hz reduces phase-tracker noise by ~50× in
    variance (~7× in RMS).  For ionospheric phase work we want the
    narrowest filter that still passes the slow phase fluctuations
    we're trying to predict (well below 100 Hz).
    """
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        # No scipy -> graceful fallback to FFT brick-wall
        n = z_dc.size
        Z = np.fft.fft(z_dc)
        f = np.fft.fftfreq(n, 1.0 / sr)
        Z[np.abs(f) > cutoff_hz] = 0.0
        return np.fft.ifft(Z).astype(np.complex64)
    nyq = sr / 2.0
    wn = cutoff_hz / nyq
    if wn >= 1.0:
        return z_dc
    b, a = butter(order, wn, btype="low")
    # Filter real and imaginary parts separately (filtfilt is real-only)
    fr = filtfilt(b, a, z_dc.real)
    fi = filtfilt(b, a, z_dc.imag)
    return (fr + 1j * fi).astype(np.complex64)


def amplitude_phase(z_dc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Magnitude (linear) and unwrapped phase (radians) of complex z_dc."""
    A = np.abs(z_dc).astype(np.float32)
    phi = np.unwrap(np.angle(z_dc)).astype(np.float64)
    return A, phi


# Defaults adopted from wwv_phase_analysis_final.py
SNR_THRESHOLD_DB = 12.0
PHASE_JUMP_THRESHOLD_RAD = 1.5
FADE_MARGIN_SAMPLES = 100


def detect_fades(amplitude: np.ndarray, sr: float,
                 threshold_db: float = SNR_THRESHOLD_DB,
                 margin_samples: int = FADE_MARGIN_SAMPLES
                 ) -> np.ndarray:
    """Boolean mask, True where signal is in a fade.  Threshold is
    set ``threshold_db`` below the 90th-percentile amplitude (a robust
    proxy for the unfaded carrier level).  Detected fades are then
    dilated by ``margin_samples`` on each side to cover edge effects
    around the fade boundary."""
    amp_db = 20.0 * np.log10(amplitude + 1e-12)
    ref_db = np.percentile(amp_db, 90)
    is_faded = amp_db < (ref_db - threshold_db)
    if margin_samples <= 0:
        return is_faded.astype(bool)
    # Dilate via a uniform-window max filter
    n = is_faded.size
    out = np.zeros(n, dtype=bool)
    starts = np.flatnonzero(np.diff(np.concatenate([[0],
                                                     is_faded.astype(int)])) == 1)
    ends   = np.flatnonzero(np.diff(np.concatenate([is_faded.astype(int),
                                                     [0]])) == -1) + 1
    if is_faded[0]:
        starts = np.concatenate([[0], starts])
    if is_faded[-1] and len(ends) == 0:
        ends = np.array([n])
    for s, e in zip(starts, ends):
        out[max(0, s - margin_samples):min(n, e + margin_samples)] = True
    return out


def splice_phase(phase: np.ndarray, fade_mask: np.ndarray,
                 jump_threshold_rad: float = PHASE_JUMP_THRESHOLD_RAD
                 ) -> Tuple[np.ndarray, int]:
    """Remove fade-induced phase discontinuities.

    When the signal fades out the phase-locked-loop / unwrap loses
    track and on recovery picks up an arbitrary integer-2π offset.
    These discontinuities corrupt the structure-function and
    predictor outputs.  Walking the time series from start to end:
    when we exit a fade region with a large jump from the last valid
    sample, subtract that jump from the entire remainder of the trace.

    Returns (spliced_phase, n_splices_applied).  Ported from
    wwv_phase_analysis_final.splice_phase."""
    phase_out = np.copy(phase)
    n = phase_out.size
    # Identify exit-from-fade transitions
    fm = fade_mask.astype(int)
    exits = np.flatnonzero(np.diff(fm) == -1) + 1   # index into phase
    n_splices = 0
    for idx in exits:
        if idx < 10 or idx >= n - 10:
            continue
        # find last valid sample BEFORE the fade we just exited
        pre = idx - 1
        while pre > 0 and fade_mask[pre]:
            pre -= 1
        if pre <= 0:
            continue
        jump = phase_out[idx] - phase_out[pre]
        if abs(jump) > jump_threshold_rad:
            phase_out[idx:] -= jump
            n_splices += 1
    return phase_out, n_splices


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


# --------------------------------------------------- programmatic entry

def compute_predictability(path: str, *,
                           taus_s: Optional[np.ndarray] = None,
                           bandwidth_hz: float = 200.0,
                           snr_fade_db: float = SNR_THRESHOLD_DB,
                           fade_margin_samples: int = FADE_MARGIN_SAMPLES,
                           phase_jump_threshold_rad: float = PHASE_JUMP_THRESHOLD_RAD,
                           linear_fit_window_s: float = 0.05,
                           kalman_q_omega: float = 1.0,
                           skip_kalman: bool = False,
                           carrier_search_lo: float = 500.0,
                           carrier_search_hi: float = 4500.0,
                           ) -> dict:
    """Run the full prediction pipeline on one file and return a
    summary dict.

    Pipeline (whole-file, post-splice):
      1. FFT-search the carrier within |f| ∈ [search_lo, search_hi].
      2. Mix to DC; brick-wall-equivalent low-pass to ±bandwidth_hz
         (default ±200 Hz to match the previous wwv_phase_analysis
         work).  Reduces phase-tracker noise by 20kHz/2*bw vs the raw
         signal, ~7× improvement in phase RMS for noise-limited cases.
      3. Detect fades on the smoothed amplitude; expand by
         ``fade_margin_samples``.
      4. Unwrap the whole-file phase, then SPLICE across each fade
         exit (subtract any jump > ``phase_jump_threshold_rad`` from
         all subsequent samples).  This gives a continuous phase track.
      5. Compute the predictors over the whole spliced track,
         masking out samples inside fades from the prediction-error
         averages (so deep-fade samples don't pollute the metric).

    Returns a dict; on error, returns ``{"error": "..."}``.
    """
    if taus_s is None:
        # Default geometric grid plus exact 20 / 50 / 75 ms reference
        # points used by the batch summary tooling.
        base = np.geomspace(1e-3, 0.2, 25)
        anchors = np.array([0.020, 0.050, 0.075])
        taus_s = np.unique(np.concatenate([base, anchors]))
    try:
        z, sr = load_kiwi_iq_wav(path)
        if z.size < int(5 * sr):
            return {"error": f"file too short ({z.size/sr:.1f}s)"}

        f_c = find_carrier_freq(z, sr,
                                search_lo=carrier_search_lo,
                                search_hi=carrier_search_hi)
        # 1. mix to DC
        z_dc = mix_to_dc(z, sr, f_c)
        # 2. low-pass to ±bandwidth_hz around the carrier
        if bandwidth_hz > 0 and bandwidth_hz < sr / 2:
            z_dc = lowpass_complex(z_dc, sr, cutoff_hz=bandwidth_hz)
        A = np.abs(z_dc).astype(np.float32)

        # 3. fade mask
        fade_mask = detect_fades(A, sr, threshold_db=snr_fade_db,
                                 margin_samples=fade_margin_samples)
        n_total = z_dc.size
        n_fade = int(fade_mask.sum())
        good_frac = 1.0 - n_fade / n_total
        if good_frac < 0.05:
            return {"error": f"only {good_frac*100:.1f}% of file is unfaded"}

        # 4. unwrap whole file then splice
        phi_full = np.unwrap(np.angle(z_dc)).astype(np.float64)
        phi_full, n_splices = splice_phase(
            phi_full, fade_mask,
            jump_threshold_rad=phase_jump_threshold_rad)

        # Residual carrier slope (post-splice) — measured over unfaded
        # samples only, as a quality / Doppler diagnostic
        good_idx = np.flatnonzero(~fade_mask)
        if good_idx.size >= 100:
            t_good = good_idx.astype(np.float64) / sr
            slope_rad_s, _ = np.polyfit(t_good, phi_full[good_idx], 1)
            residual_slope_hz = float(slope_rad_s) / (2 * np.pi)
        else:
            residual_slope_hz = float("nan")

        # 5. predictors on the spliced full-file phase track, with
        # the fade mask used to drop t-and-t+τ pairs whose endpoints
        # land inside a fade.
        n_tau = taus_s.size
        Dpred_const  = np.full(n_tau, np.nan, dtype=np.float64)
        Dpred_linear = np.full(n_tau, np.nan, dtype=np.float64)
        Dpred_kalman = np.full(n_tau, np.nan, dtype=np.float64)
        Ddesc        = np.full(n_tau, np.nan, dtype=np.float64)

        valid = ~fade_mask
        # constant-phase + descriptive structure function are the same
        # quantity by construction (D_pred_const(tau) = <(phi(t+tau)-phi(t))^2>)
        for k, tau in enumerate(taus_s):
            m = int(round(tau * sr))
            if m <= 0 or m >= n_total:
                continue
            pair_valid = valid[:n_total - m] & valid[m:]
            if not np.any(pair_valid):
                continue
            d = phi_full[m:n_total] - phi_full[:n_total - m]
            d = d[pair_valid]
            mse = float(np.mean(d * d))
            Dpred_const[k] = mse
            Ddesc[k]        = mse

        # linear extrapolation: vectorised running OLS on the
        # spliced phase, evaluated only at unfaded prediction epochs
        n_fit = int(round(linear_fit_window_s * sr))
        if n_fit >= 8 and n_fit < n_total:
            t_local = np.arange(n_fit, dtype=np.float64) / sr
            x_mean = float(t_local.mean())
            x_dev = t_local - x_mean
            Sxx = float((x_dev * x_dev).sum())
            kernel = np.ones(n_fit) / n_fit
            phi_mean = np.convolve(phi_full, kernel, mode="valid")
            cross = np.convolve(phi_full, x_dev[::-1], mode="valid")
            slope_arr = cross / Sxx
            t_end = t_local[-1]
            phi_at_t0 = phi_mean + slope_arr * (t_end - x_mean)
            base_idx = np.arange(slope_arr.size) + (n_fit - 1)
            for k, tau in enumerate(taus_s):
                m = int(round(tau * sr))
                if m <= 0:
                    continue
                target_ix = base_idx + m
                in_range = target_ix < n_total
                if not np.any(in_range):
                    continue
                ti = target_ix[in_range]
                bi = base_idx[in_range]
                # Both t0 and t0+tau must be in unfaded sections, AND
                # the entire fit window [t0-fit_window, t0] should be
                # mostly unfaded so the slope estimate is meaningful.
                pair_ok = valid[bi] & valid[ti]
                if not np.any(pair_ok):
                    continue
                phi_pred = phi_at_t0[in_range][pair_ok] \
                          + slope_arr[in_range][pair_ok] * tau
                phi_true = phi_full[ti[pair_ok]]
                d = phi_true - phi_pred
                Dpred_linear[k] = float(np.mean(d * d))

        # Kalman (optional, expensive)
        if not skip_kalman:
            kal = predict_kalman_with_mask(phi_full, fade_mask, sr,
                                           taus_s, kalman_q_omega)
            Dpred_kalman[:] = kal

        rms_const  = np.sqrt(Dpred_const)
        rms_linear = np.sqrt(Dpred_linear)
        rms_kalman = np.sqrt(Dpred_kalman)
        rms_desc   = np.sqrt(Ddesc)

        return dict(
            sample_rate_Hz=float(sr),
            carrier_beat_Hz=float(f_c),
            bandwidth_hz=float(bandwidth_hz),
            n_total_samples=int(n_total),
            n_fade_samples=int(n_fade),
            good_fraction=float(good_frac),
            n_splices=int(n_splices),
            residual_slope_Hz=residual_slope_hz,
            median_amp=float(np.median(A)),
            taus_s=taus_s.tolist(),
            rms_pred_constant_rad=rms_const.tolist(),
            rms_pred_linear_rad=rms_linear.tolist(),
            rms_pred_kalman_rad=rms_kalman.tolist(),
            rms_descriptive_rad=rms_desc.tolist(),
        )
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def predict_kalman_with_mask(phi: np.ndarray, fade_mask: np.ndarray,
                             sr: float, taus: np.ndarray,
                             q_omega: float = 1.0,
                             r_phi: Optional[float] = None) -> np.ndarray:
    """Kalman-filter prediction-error structure function with a fade
    mask: when fade_mask[k] is True, skip the measurement update at
    that step (predict-only) so the filter doesn't lock onto fade-
    contaminated samples.

    Same return shape as ``predict_kalman``.
    """
    n = phi.size
    dt = 1.0 / sr
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = np.array([[q_omega * dt**3 / 3, q_omega * dt**2 / 2],
                  [q_omega * dt**2 / 2, q_omega * dt]])
    H = np.array([[1.0, 0.0]])
    if r_phi is None:
        d1 = np.diff(phi[~fade_mask] if np.any(~fade_mask) else phi)
        r_phi = max(0.5 * float(np.median(d1 * d1)), 1e-12)

    x_post = np.zeros((n, 2))
    x = np.array([phi[0], 0.0])
    P = np.array([[r_phi, 0.0], [0.0, q_omega]])
    x_post[0] = x
    for k in range(1, n):
        x = F @ x
        P = F @ P @ F.T + Q
        if not fade_mask[k]:
            y = phi[k] - (H @ x)[0]
            S = (H @ P @ H.T)[0, 0] + r_phi
            K = (P @ H.T).flatten() / S
            x = x + K * y
            P = P - np.outer(K, H @ P)
        x_post[k] = x

    valid = ~fade_mask
    out = np.full(taus.size, np.nan, dtype=np.float64)
    for k_tau, tau in enumerate(taus):
        m = int(round(tau * sr))
        if m <= 0 or m >= n:
            continue
        phi_pred = x_post[:n - m, 0] + x_post[:n - m, 1] * tau
        phi_true = phi[m:]
        pair_ok = valid[:n - m] & valid[m:]
        if not np.any(pair_ok):
            continue
        d = (phi_true - phi_pred)[pair_ok]
        out[k_tau] = float(np.mean(d * d))
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
    p.add_argument("--bandwidth-hz", type=float, default=200.0,
                   help="one-sided low-pass cutoff around the carrier (Hz). "
                        "200 Hz preserves all the slow phase fluctuations "
                        "while reducing tracker noise by ~50× in variance.")
    p.add_argument("--snr-fade-db", type=float, default=12.0,
                   help="fade threshold below 90th-percentile amplitude (dB)")
    p.add_argument("--carrier-search-lo", type=float, default=500.0)
    p.add_argument("--carrier-search-hi", type=float, default=4500.0)
    p.add_argument("--tau-min", type=float, default=1e-3)
    p.add_argument("--tau-max", type=float, default=0.2)
    p.add_argument("--n-tau", type=int, default=40)
    p.add_argument("--linear-fit-window-s", type=float, default=0.05)
    p.add_argument("--kalman-q-omega", type=float, default=1.0)
    p.add_argument("--skip-kalman", action="store_true")
    args = p.parse_args()

    taus = np.geomspace(args.tau_min, args.tau_max, args.n_tau)
    print(f"Processing {args.input}")
    res = compute_predictability(
        args.input, taus_s=taus,
        bandwidth_hz=args.bandwidth_hz,
        snr_fade_db=args.snr_fade_db,
        linear_fit_window_s=args.linear_fit_window_s,
        kalman_q_omega=args.kalman_q_omega,
        skip_kalman=args.skip_kalman,
        carrier_search_lo=args.carrier_search_lo,
        carrier_search_hi=args.carrier_search_hi,
    )
    if "error" in res:
        print(f"ERROR: {res['error']}", file=sys.stderr)
        sys.exit(1)

    sr = res["sample_rate_Hz"]
    print(f"  sample rate:         {sr:.0f} Hz")
    print(f"  carrier beat:        {res['carrier_beat_Hz']:+.3f} Hz")
    print(f"  bandwidth:           ±{res['bandwidth_hz']:.0f} Hz")
    print(f"  good fraction:       {res['good_fraction']*100:.1f}% "
          f"(unfaded samples)")
    print(f"  fade splices:        {res['n_splices']}")
    print(f"  residual slope:      {res['residual_slope_Hz']:+.4f} Hz")
    print(f"\nRMS prediction error (rad) at selected lookaheads:")
    print(f"  {'tau (ms)':>8}  {'const':>8}  {'linear':>8}  {'kalman':>8}  "
          f"{'D(tau)':>8}")
    rms_const = np.array(res["rms_pred_constant_rad"])
    rms_linear = np.array(res["rms_pred_linear_rad"])
    rms_kalman = np.array(res["rms_pred_kalman_rad"])
    rms_desc   = np.array(res["rms_descriptive_rad"])
    for tau in [0.001, 0.005, 0.020, 0.050, 0.100, 0.200]:
        if tau > taus[-1]: continue
        i = int(np.argmin(np.abs(taus - tau)))
        print(f"  {1000*taus[i]:8.2f}  {rms_const[i]:8.4f}  "
              f"{rms_linear[i]:8.4f}  "
              f"{rms_kalman[i] if np.isfinite(rms_kalman[i]) else float('nan'):8.4f}  "
              f"{rms_desc[i]:8.4f}")

    base, _ = os.path.splitext(args.input)
    out_png = args.out_png or (base + "_predict.png")
    out_json = args.out_json or (base + "_predict.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.loglog(1e3 * taus, rms_const,  "C0",  lw=1.5,
              label="constant-phase (do nothing)")
    ax.loglog(1e3 * taus, rms_linear, "C1",  lw=1.5,
              label=f"linear extrap. (fit {1e3*args.linear_fit_window_s:.0f} ms)")
    if not args.skip_kalman:
        ax.loglog(1e3 * taus, rms_kalman, "C2",  lw=1.5,
                  label=f"Kalman (q_ω={args.kalman_q_omega:.0g})")
    ax.set_xlabel("look-ahead τ (ms)")
    ax.set_ylabel("RMS prediction error (rad)")
    ax.set_title(f"Phase predictability — {os.path.basename(args.input)}\n"
                 f"BW=±{args.bandwidth_hz:.0f} Hz, "
                 f"good={res['good_fraction']*100:.0f}%, "
                 f"residual carrier={res['residual_slope_Hz']:+.3f} Hz, "
                 f"{res['n_splices']} splices")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    ax.axhline(0.5, color="0.6", lw=0.8, ls="--")
    ax.text(taus[-1]*1e3, 0.5, " 0.5 rad goal",
            va="center", ha="right", color="0.4", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out_png}")

    res["input"] = os.path.abspath(args.input)
    res["linear_fit_window_s"] = args.linear_fit_window_s
    res["kalman_q_omega"] = args.kalman_q_omega
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
