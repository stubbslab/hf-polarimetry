"""picoacq.ps5444a — thin wrapper around the PicoSDK 5000a driver
(used by both 5444D and the rest of the 5000-series).

The PicoSDK Python wrappers expose the raw C library functions; we
package them into a small, idiomatic context-manager interface.

This module IMPORTS picosdk at use time (not at module load), so the
rest of picoacq still imports cleanly on machines without the SDK
installed.  The simulator path doesn't need it.
"""

import ctypes
import time
from contextlib import contextmanager

import numpy as np


# Voltage range LUT (in volts) for the 5000a channel ranges.
# Index = ps5000a enum value of PS5000A_RANGE.  Values per Pico docs.
_RANGE_VOLTS = {
    0: 0.010,   # PS5000A_10MV
    1: 0.020,
    2: 0.050,
    3: 0.100,
    4: 0.200,
    5: 0.500,
    6: 1.000,
    7: 2.000,
    8: 5.000,
    9: 10.000,
    10: 20.000,
}


def _ensure_pico_dyld_path():
    """On macOS, the PicoSDK dylibs are installed inside a framework
    bundle at /Library/Frameworks/PicoSDK.framework/Libraries/<lib>/
    rather than in /usr/local/lib.  The picosdk Python wrapper looks
    for them via DYLD_LIBRARY_PATH (macOS) or LD_LIBRARY_PATH (Linux).
    Inject the framework path automatically if it exists and isn't
    already on the search path."""
    import sys, glob, os
    fw_root = "/Library/Frameworks/PicoSDK.framework/Libraries"
    if not os.path.isdir(fw_root):
        return
    # Each library lives in its own subdir.  Add all of them to the path.
    sub_dirs = sorted(glob.glob(os.path.join(fw_root, "lib*")))
    if not sub_dirs:
        return
    addition = os.pathsep.join(sub_dirs)
    for env_var in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        existing = os.environ.get(env_var, "")
        if addition not in existing.split(os.pathsep):
            os.environ[env_var] = (addition + os.pathsep + existing).rstrip(
                os.pathsep
            )


def _import_picosdk():
    _ensure_pico_dyld_path()
    try:
        from picosdk.ps5000a import ps5000a as ps  # noqa
        from picosdk.functions import assert_pico_ok
        return ps, assert_pico_ok
    except Exception as e:
        raise RuntimeError(
            "picosdk not installed or PicoSDK system library not found.  "
            "See https://www.picotech.com/downloads for the PicoSDK installer, "
            "then `pip install picosdk`.  On macOS, the SDK installs as a "
            "framework at /Library/Frameworks/PicoSDK.framework/."
        ) from e


@contextmanager
def open_5444(n_channels=4):
    """Open the 5000-series driver, yield (ps_module, handle).
    Closes the handle on exit.

    Resolution selection: the 5444D supports 16 bits at 2 channels max,
    14 bits at 4 channels.  We pick the highest resolution compatible
    with the requested channel count.  See PicoScope 5000a series
    documentation, ``ps5000aSetDeviceResolution``.
    """
    ps, assert_pico_ok = _import_picosdk()
    if n_channels <= 2:
        resolution_name = "PS5000A_DR_16BIT"
    else:
        resolution_name = "PS5000A_DR_14BIT"
    res_enum = ps.PS5000A_DEVICE_RESOLUTION[resolution_name]

    handle = ctypes.c_int16()
    status = ps.ps5000aOpenUnit(ctypes.byref(handle), None, res_enum)
    if status != 0:
        # Sometimes the unit opens at default resolution, then we set it.
        status0 = ps.ps5000aOpenUnit(ctypes.byref(handle), None,
                                     ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_8BIT"])
        assert_pico_ok(status0)
        status = ps.ps5000aSetDeviceResolution(handle, res_enum)
        assert_pico_ok(status)
    try:
        yield ps, handle
    finally:
        ps.ps5000aCloseUnit(handle)


def configure_channel(ps, handle, channel_letter, range_volts=2.0,
                      coupling="DC"):
    """Enable channel and set its voltage range."""
    enum_ch = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{channel_letter}"]
    enum_co = ps.PS5000A_COUPLING[f"PS5000A_{coupling}"]
    # Find the closest PS5000A_RANGE >= the requested volts
    enum_rng = None
    for k, v in _RANGE_VOLTS.items():
        if v >= range_volts:
            enum_rng = k; break
    if enum_rng is None:
        raise ValueError(f"requested range {range_volts}V exceeds 20V cap")
    # ps5000aSetChannel: handle, channel, enabled, coupling, range, analogueOffset
    status = ps.ps5000aSetChannel(handle,
                                  ctypes.c_int32(enum_ch),
                                  ctypes.c_int16(1),       # enabled = True
                                  ctypes.c_int32(enum_co),
                                  ctypes.c_int32(enum_rng),
                                  ctypes.c_float(0.0))
    return enum_rng


def get_timebase(ps, handle, target_rate_sps):
    """Return (timebase_idx, actual_sample_rate) for the closest available
    sample rate at or below target_rate_sps."""
    # PicoScope 5000a uses an integer timebase index.  In 16-bit mode the
    # base rate is 62.5 MHz, so timebase n -> 62.5e6 / 2^(n-1) when n >= 1.
    # We conservatively pick a low integer and let the driver tell us the
    # actual rate via ps5000aGetTimebase2.
    # Pico SDK ps5000aGetTimebase2 signature:
    #   PICO_STATUS ps5000aGetTimebase2(handle, timebase, noSamples,
    #                                    *timeIntervalNs, *maxSamples,
    #                                    segmentIndex)
    #
    # Strategy: walk timebase indices and find the smallest tb whose
    # actual sample rate is <= target_rate_sps * 1.05 (i.e., the
    # closest available rate at or below the requested value).
    # Some early indices are invalid for 4-channel mode; just skip those
    # (status PICO_INVALID_TIMEBASE = 14).
    best = None
    for tb in range(0, 100_000):
        timeIntervalNs = ctypes.c_float()
        maxSamples     = ctypes.c_int32()
        ret = ps.ps5000aGetTimebase2(handle,
                                     ctypes.c_uint32(tb),
                                     ctypes.c_int32(1024),
                                     ctypes.byref(timeIntervalNs),
                                     ctypes.byref(maxSamples),
                                     ctypes.c_uint32(0))   # segmentIndex
        if ret != 0:
            # Skip invalid timebases (early ones often invalid in 4-ch mode)
            if best is not None and tb > best[0] + 5:
                # We've found a working one and now hit invalids again ⇒ done
                break
            continue
        if timeIntervalNs.value <= 0:
            continue
        actual_rate = 1.0 / (timeIntervalNs.value * 1e-9)
        if actual_rate <= target_rate_sps * 1.05:
            return tb, actual_rate
        # Otherwise rate is too fast; try next tb (slower)
        best = (tb, actual_rate)   # remember in case nothing slower exists
    if best is not None:
        # Nothing below target; return the slowest we found
        return best
    raise RuntimeError(f"could not find a timebase for {target_rate_sps} Hz")


def acquire_block(ps, handle, channels, n_samples, timebase_idx,
                  ranges_volts):
    """Acquire a block of `n_samples` samples on each enabled channel.
    Returns a dict {channel_letter -> ndarray (int16)}.
    """
    # Allocate buffers per channel.  All ctypes integer args must be wrapped
    # in their proper type — the picosdk wrapper's argtypes are not
    # explicitly declared, so byref()/raw-int arguments must be carefully
    # cast or Python's ctypes complains with the sort of TypeError you saw.
    buffers = {}
    for ch in channels:
        buf = (ctypes.c_int16 * n_samples)()
        buffers[ch] = buf
        enum_ch = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{ch}"]
        ps.ps5000aSetDataBuffer(handle, ctypes.c_int32(enum_ch),
                                ctypes.byref(buf),
                                ctypes.c_int32(n_samples),
                                ctypes.c_uint32(0),       # segmentIndex
                                ctypes.c_int32(0))        # PS5000A_RATIO_MODE_NONE

    # Run the block
    timeIndisposedMs = ctypes.c_int32()
    # ps5000aRunBlock: handle, noOfPreTriggerSamples, noOfPostTriggerSamples,
    #                  timebase, *timeIndisposedMs, segmentIndex,
    #                  lpReady (callback or NULL), pParameter (NULL)
    ps.ps5000aRunBlock(handle,
                       ctypes.c_int32(0),                  # pre
                       ctypes.c_int32(n_samples),          # post
                       ctypes.c_uint32(timebase_idx),
                       ctypes.byref(timeIndisposedMs),
                       ctypes.c_uint32(0),                 # segmentIndex
                       None, None)
    # Wait for completion
    ready = ctypes.c_int16(0)
    while not ready.value:
        ps.ps5000aIsReady(handle, ctypes.byref(ready))
        time.sleep(0.005)

    # Pull values: handle, startIndex, *noOfSamples, downSampleRatio,
    #              downSampleRatioMode, segmentIndex, *overflow
    nSamples = ctypes.c_uint32(n_samples)
    overflow = ctypes.c_int16()
    ps.ps5000aGetValues(handle,
                        ctypes.c_uint32(0),                # startIndex
                        ctypes.byref(nSamples),
                        ctypes.c_uint32(1),                # downSampleRatio
                        ctypes.c_int32(0),                 # downSampleRatioMode
                        ctypes.c_uint32(0),                # segmentIndex
                        ctypes.byref(overflow))

    out = {}
    for ch in channels:
        out[ch] = np.frombuffer(buffers[ch], dtype=np.int16,
                                count=nSamples.value).copy()
    return out
