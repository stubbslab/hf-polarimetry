"""picoacq.recorder — orchestrates a capture: open scope, configure
channels, run a block, and write the result to a triloop-format HDF5
file.

Two-step capture:
  1. (auto-range, optional) probe each channel briefly at the largest
     voltage range to estimate signal RMS and peak.  Pick the smallest
     available range whose full-scale ≥ K · peak (default K=3) for
     the science capture.
  2. capture for the requested duration with the chosen ranges.

Samples are stored as float32 volts in the HDF5 file (gain = 1.0,
unit-aware).  Per-channel original Pico range setting and the probe
RMS / peak are recorded in capture_settings for traceability.

If the PicoScope SDK isn't available (or --simulate is used), falls
back to picoacq.simulator.
"""

import os, json
from datetime import datetime, timezone

import numpy as np

# Re-use triloop's HDF5 writer so the format stays consistent.
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "..", "triloop")))
from triloop.io_hdf5 import write_capture
from triloop.config import default_loops_config

from .simulator import simulate_capture


# Ordered list of (range_volts, range_idx) ascending by full-scale.  Same
# values as picoacq.ps5444a._RANGE_VOLTS but ordered for the auto-range
# search.
_RANGE_LADDER = [
    (0.010,  0), (0.020,  1), (0.050,  2), (0.100,  3), (0.200,  4),
    (0.500,  5), (1.000,  6), (2.000,  7), (5.000,  8), (10.000, 9),
    (20.000, 10),
]


def _pick_range(peak_v, headroom=3.0):
    """Pick the smallest available Pico range whose full scale is at
    least `headroom * peak_v`.  Returns the range in volts."""
    target = headroom * peak_v
    for r_v, _ in _RANGE_LADDER:
        if r_v >= target:
            return r_v
    return _RANGE_LADDER[-1][0]   # fall back to largest range


def auto_range(channels, sample_rate, probe_duration_s=0.1, coupling="DC",
               headroom=3.0):
    """Probe each channel at maximum range, measure peak/RMS, and pick
    a per-channel voltage range with `headroom` margin above peak.

    Parameters
    ----------
    channels : list of channel-letter strings (e.g. ['A', 'B', 'C', 'D'])
    sample_rate : Hz, requested
    probe_duration_s : seconds (short, default 0.1)
    coupling : "DC" or "AC"
    headroom : multiplier above measured peak; the smallest available
        range whose full-scale ≥ headroom * peak is chosen.  Default 3.

    Returns
    -------
    dict with keys:
        ranges_volts  : {channel -> chosen range volts}
        peaks_volts   : {channel -> measured peak |sample| in volts}
        rms_volts     : {channel -> measured RMS in volts}
        actual_rate   : float Hz, the actual probe rate used
        n_samples     : int
    """
    from .ps5444a import (open_5444, configure_channel, get_timebase,
                          acquire_block, _RANGE_VOLTS)
    n_samples = int(probe_duration_s * sample_rate)
    if n_samples < 1024:
        n_samples = 1024
    probe_ranges = {ch: 20.0 for ch in channels}   # widest range
    with open_5444(n_channels=len(channels)) as (ps, handle):
        timebase_idx, actual_rate = get_timebase(ps, handle, sample_rate)
        per_ch_range_idx = {}
        for ch in channels:
            r_idx = configure_channel(ps, handle, ch,
                                      range_volts=probe_ranges[ch],
                                      coupling=coupling)
            per_ch_range_idx[ch] = r_idx
        raw = acquire_block(ps, handle, channels, n_samples, timebase_idx,
                            probe_ranges)

    peaks = {}; rmss = {}; ranges = {}
    for ch in channels:
        r_v = _RANGE_VOLTS[per_ch_range_idx[ch]]
        gain = r_v / 32767.0
        sig_v = raw[ch].astype(np.float32) * gain
        peak = float(np.max(np.abs(sig_v)))
        rms  = float(np.sqrt(np.mean(sig_v ** 2)))
        peaks[ch] = peak
        rmss[ch]  = rms
        ranges[ch] = _pick_range(peak, headroom=headroom)

    return dict(ranges_volts=ranges, peaks_volts=peaks, rms_volts=rmss,
                actual_rate=actual_rate, n_samples=n_samples,
                headroom=headroom)


def _try_real_capture(duration_s, sample_rate, channels, ranges_volts,
                      coupling="DC"):
    """Attempt a real PicoScope acquisition.  Returns the same dict
    shape as simulate_capture, or raises RuntimeError if hardware/SDK
    is unavailable."""
    from .ps5444a import (open_5444, configure_channel, get_timebase,
                          acquire_block, _RANGE_VOLTS)
    n_samples = int(duration_s * sample_rate)
    with open_5444(n_channels=len(channels)) as (ps, handle):
        timebase_idx, actual_rate = get_timebase(ps, handle, sample_rate)
        per_ch_range_idx = {}
        for ch in channels:
            r_idx = configure_channel(ps, handle, ch,
                                      range_volts=ranges_volts.get(ch, 2.0),
                                      coupling=coupling)
            per_ch_range_idx[ch] = r_idx
        raw = acquire_block(ps, handle, channels, n_samples, timebase_idx,
                            ranges_volts)
        # Convert int16 -> volts (float32) and store volts directly.
        # gain_v_per_count is set to 1.0 so the value at write-time is
        # already volts; this keeps the HDF5 reader unit-aware without
        # special-casing.
        ch_dict = {}
        ch_gains = {}; ch_offsets = {}
        for ch in channels:
            r_v = _RANGE_VOLTS[per_ch_range_idx[ch]]
            scale = r_v / 32767.0
            ch_dict[ch] = (raw[ch].astype(np.float32) * scale).astype(np.float32)
            ch_gains[ch] = 1.0       # values are already in volts
            ch_offsets[ch] = 0.0
        return dict(
            t=np.arange(n_samples) / actual_rate,
            channels=ch_dict,
            sample_rate=actual_rate,
            scope_model="PicoScope 5444D",
            scope_serial="(read from scope handle)",
            truth=None,
            channel_gains=ch_gains, channel_offsets=ch_offsets,
            picoscope_ranges_volts={ch: _RANGE_VOLTS[per_ch_range_idx[ch]]
                                    for ch in channels},
        )


# PicoScope 5444D onboard memory: 512 MS total, split across enabled
# channels.  In 4-channel mode each channel gets 128 MS; in ≤2-channel mode
# each gets 256 MS.  The block-mode capture must fit in this per-channel
# allotment.  Values from the PicoScope 5000a series datasheet.
_BUFFER_SAMPLES_PER_CHANNEL = {1: 512_000_000, 2: 256_000_000,
                               3: 128_000_000, 4: 128_000_000}


def _check_buffer(duration_s, sample_rate, n_channels):
    """Raise ValueError if the requested capture exceeds onboard memory."""
    # Look up per-channel sample budget; default to the 4-channel value
    # (most conservative) if n_channels is out of range.
    per_ch = _BUFFER_SAMPLES_PER_CHANNEL.get(n_channels, 128_000_000)
    requested = int(duration_s * sample_rate)
    if requested > per_ch:
        max_dur = per_ch / sample_rate
        raise ValueError(
            f"requested {duration_s:.2f} s @ {sample_rate:.0f} Hz with "
            f"{n_channels} channels needs {requested:,} samples/channel, "
            f"but the 5444D onboard buffer holds only {per_ch:,} per channel "
            f"in this configuration (max {max_dur:.2f} s)."
        )


def capture(output_file, duration_s=8.0, sample_rate=15_625_000.0,
            channels=("A", "B", "C", "D"), ranges_volts=None, coupling="DC",
            loops_config=None, simulate=False,
            sim_kwargs=None,
            auto_range_enabled=True, headroom=3.0,
            probe_duration_s=0.1, rf_bands=None, verbose=True):
    """High-level capture function.  Tries real hardware first; if that
    fails (or simulate=True), uses the simulator.

    Writes a triloop-format HDF5 file at `output_file`.

    Parameters
    ----------
    output_file : path
    duration_s : capture length, seconds
    sample_rate : Hz (target; actual depends on Pico timebase)
    channels : list of channel letters to enable
    ranges_volts : optional pre-set per-channel voltage ranges.  If
        None and auto_range_enabled is True, the function probes each
        channel and chooses ranges automatically.
    auto_range_enabled : if True (default), run a brief probe capture
        to determine signal levels, then pick ranges with `headroom`
        margin above peak.  Skipped if `ranges_volts` is provided
        explicitly.
    headroom : multiplier above measured peak (default 3.0).
    probe_duration_s : duration of the probe capture (default 0.1 s).
    rf_bands : optional list of expected RF frequencies (Hz), recorded
        in capture_settings so downstream tools can map aliased baseband
        peaks back to the correct RF band.
    verbose : if True, print probe results to stdout.
    """
    if sim_kwargs is None:
        sim_kwargs = {}
    if loops_config is None:
        loops_config = default_loops_config()

    if not simulate:
        _check_buffer(duration_s, sample_rate, len(channels))

    used_simulator = False
    auto_range_info = None

    if simulate:
        cap = simulate_capture(duration_s=duration_s, sample_rate=sample_rate,
                               **sim_kwargs)
        # Default to ±2 V for the simulator path
        if ranges_volts is None:
            ranges_volts = {ch: 2.0 for ch in channels}
        used_simulator = True
    else:
        try:
            # Step 1: auto-range probe (unless ranges supplied)
            if ranges_volts is None and auto_range_enabled:
                if verbose:
                    print(f"[picoacq] auto-range probe ({probe_duration_s*1000:.0f} ms "
                          f"at {sample_rate:.0f} Hz, headroom={headroom})")
                auto_range_info = auto_range(
                    list(channels), sample_rate,
                    probe_duration_s=probe_duration_s,
                    coupling=coupling, headroom=headroom,
                )
                ranges_volts = auto_range_info["ranges_volts"]
                if verbose:
                    print("[picoacq] probe results:")
                    for ch in channels:
                        print(f"           {ch}: peak={auto_range_info['peaks_volts'][ch]*1000:.2f} mV  "
                              f"RMS={auto_range_info['rms_volts'][ch]*1000:.2f} mV  "
                              f"-> range ±{ranges_volts[ch]} V")
            elif ranges_volts is None:
                ranges_volts = {ch: 2.0 for ch in channels}

            # Step 2: real capture at chosen ranges
            cap = _try_real_capture(duration_s, sample_rate,
                                    list(channels), ranges_volts, coupling)
        except Exception as e:
            print(f"[picoacq] real capture failed: {e}\n"
                  f"[picoacq] falling back to simulator")
            cap = simulate_capture(duration_s=duration_s, sample_rate=sample_rate,
                                   **sim_kwargs)
            if ranges_volts is None:
                ranges_volts = {ch: 2.0 for ch in channels}
            used_simulator = True

    settings = {
        "requested_duration_s": duration_s,
        "requested_sample_rate": sample_rate,
        "channels": list(channels),
        "ranges_volts": ranges_volts,
        "coupling": coupling,
        "used_simulator": used_simulator,
        "auto_range_enabled": auto_range_enabled and not used_simulator,
        "headroom": headroom,
        "truth": cap.get("truth"),
    }
    if auto_range_info is not None:
        settings["auto_range_probe"] = {
            "peaks_volts": auto_range_info["peaks_volts"],
            "rms_volts": auto_range_info["rms_volts"],
            "headroom": auto_range_info["headroom"],
            "probe_duration_s": probe_duration_s,
        }
    if "picoscope_ranges_volts" in cap:
        settings["picoscope_ranges_volts"] = cap["picoscope_ranges_volts"]
    if rf_bands:
        settings["rf_bands_hz"] = list(rf_bands)

    write_capture(
        output_file,
        channels=cap["channels"],
        sample_rate=cap["sample_rate"],
        start_time_utc=datetime.now(timezone.utc).isoformat(),
        scope_model=cap["scope_model"],
        scope_serial=cap["scope_serial"],
        capture_settings=settings,
        loops_config=loops_config,
        channel_gains=cap.get("channel_gains"),
        channel_offsets=cap.get("channel_offsets"),
    )
    return output_file
