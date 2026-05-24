"""picoacq.alias — map between RF and baseband under bandpass undersampling.

When the PicoScope samples at fs, an RF signal at frequency f_rf folds to
a baseband frequency

    f_bb = | f_rf - round(f_rf / fs) * fs |

The Nyquist zone is z = floor(2 f_rf / fs) + 1.  Odd zones preserve the
spectrum; even zones invert it.  This module exposes a tiny helper used
by analysis tools to disambiguate the WWV bands captured at the picoacq
default 15.625 MS/s rate.

At fs = 15.625 MS/s the WWV bands fold as:

    RF (MHz)   zone   baseband (MHz)   inverted?
       2.5      1         2.500           no
       5.0      1         5.000           no
      10.0      2         5.625           yes
      15.0      2         0.625           yes
      20.0      3         4.375           no
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Alias:
    rf_hz: float
    baseband_hz: float
    nyquist_zone: int
    inverted: bool


def alias_of(rf_hz: float, sample_rate_hz: float) -> Alias:
    """Compute the baseband alias of an RF tone given the sample rate."""
    fs = float(sample_rate_hz)
    f = float(rf_hz)
    n = round(f / fs)
    f_bb = abs(f - n * fs)
    zone = int(2 * f // fs) + 1
    inverted = (zone % 2 == 0)
    return Alias(rf_hz=f, baseband_hz=f_bb,
                 nyquist_zone=zone, inverted=inverted)


def aliases_for_bands(rf_bands_hz, sample_rate_hz):
    """Map a list of RF frequencies to their baseband Aliases."""
    return [alias_of(f, sample_rate_hz) for f in rf_bands_hz]


def rf_for_baseband(f_bb_hz: float, rf_bands_hz, sample_rate_hz: float,
                    tol_hz: float = 1000.0):
    """Given an observed baseband peak, return the RF band whose alias
    is closest (within tol_hz), or None.  Useful when a peak detector
    finds a tone in the digitized stream and you want the original band.
    """
    best = None
    best_err = float("inf")
    for f_rf in rf_bands_hz:
        a = alias_of(f_rf, sample_rate_hz)
        err = abs(a.baseband_hz - f_bb_hz)
        if err < best_err:
            best_err = err
            best = a
    if best is not None and best_err <= tol_hz:
        return best
    return None
