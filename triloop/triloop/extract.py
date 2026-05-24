"""triloop.extract — narrow-band complex baseband extraction."""

import numpy as np


def extract_complex_baseband(t, b, f0, BW, search_BW=None):
    """Find the spectral peak nearest to f0 in the real signal b(t),
    mix it down to DC, and rectangular-window low-pass to ±BW/2.

    Parameters
    ----------
    t  : ndarray, time stamps in seconds (assumed uniformly sampled)
    b  : ndarray, real-valued sampled signal
    f0 : float, target carrier frequency in Hz
    BW : float, total analysis bandwidth in Hz (kept window = ±BW/2)
    search_BW : float, optional. Half-width (Hz) of the search region
                around f0 to find the dominant peak.  Defaults to BW.

    Returns
    -------
    z : complex ndarray, length len(b).  Complex baseband at full rate.
    f_peak : float, the detected peak frequency (Hz)
    snr_db : float, FFT-domain peak/noise-floor SNR in dB
    """
    b = np.asarray(b, dtype=np.float64)
    n = b.size
    sr = 1.0 / np.median(np.diff(t))
    if search_BW is None:
        search_BW = BW

    nfft = 1 << int(np.ceil(np.log2(8 * n)))
    F = np.fft.fft(b, n=nfft)
    f = np.fft.fftfreq(nfft, 1.0 / sr)
    mag = np.abs(F)

    mask = (f >= f0 - search_BW) & (f <= f0 + search_BW)
    if not np.any(mask):
        raise ValueError(f"No FFT bins found in search window "
                         f"[{f0 - search_BW}, {f0 + search_BW}] Hz")
    idxs = np.flatnonzero(mask)
    k = idxs[np.argmax(mag[idxs])]
    f_peak = float(f[k])

    # SNR estimate
    far = ((np.abs(f - f_peak) > 2.0 * BW) & (np.abs(f) < sr / 2.0))
    if np.any(far):
        noise = float(np.median(mag[far]))
        snr_db = 20.0 * np.log10(mag[k] / max(noise, 1e-30))
    else:
        snr_db = float("nan")

    # Mix to DC and brick-wall low-pass via FFT
    tt = t - t[0]
    mix = np.exp(-1j * 2 * np.pi * f_peak * tt)
    Z = np.fft.fft(b * mix, n=nfft)
    fz = np.fft.fftfreq(nfft, 1.0 / sr)
    Z[np.abs(fz) > BW / 2] = 0.0
    z = np.fft.ifft(Z)[:n] * 2.0     # ×2 to recover analytic-signal amplitude
    return z.astype(np.complex128), f_peak, snr_db


def extract_three_loops(t, B1, B2, B3, f0, BW, phase_offsets_deg=(0, 0, 0)):
    """Convenience: extract baseband for all 3 loops, applying per-loop
    phase calibration offsets.  Returns a (3, N) complex array, the
    detected peak frequency (mean across loops, since real receivers
    will share an LO), and per-loop SNR.
    """
    zs = []
    f_peaks = []
    snrs = []
    for i, b in enumerate((B1, B2, B3)):
        z, fp, snr = extract_complex_baseband(t, b, f0, BW)
        # Apply user-supplied per-loop phase offset (calibration)
        phi_off = np.deg2rad(phase_offsets_deg[i])
        z *= np.exp(-1j * phi_off)
        zs.append(z); f_peaks.append(fp); snrs.append(snr)
    Z = np.array(zs)
    return Z, float(np.mean(f_peaks)), np.array(snrs)
