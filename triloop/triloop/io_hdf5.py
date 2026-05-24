"""triloop.io_hdf5 — read/write the triloop HDF5 capture format.

File format (v0.1):
  /                        HDF5 root, attribute @format_version = "0.1"
  /metadata                group
      @sample_rate         (float64) Hz
      @start_time_utc      (str, ISO-8601)
      @duration_s          (float64)
      @scope_model         (str, e.g. "PicoScope 5444D")
      @scope_serial        (str)
      @capture_settings    (str, JSON)
      @loops_config        (str, JSON)
  /channels
      /A   (int16 or float32, shape (N,))
      /B
      /C
      /D   (optional reference channel)
      Each channel dataset has attributes:
          @gain_v_per_count (float64)  (set to 1.0 for float32 datasets)
          @offset_v         (float64)
  /time   (float64, shape (N,)) -- optional; if absent, regenerate from
                                   sample_rate and duration_s.
"""

import json
import os
from datetime import datetime, timezone

import h5py
import numpy as np


FORMAT_VERSION = "0.1"


def write_capture(path, channels, sample_rate, *,
                  start_time_utc=None, scope_model="", scope_serial="",
                  capture_settings=None, loops_config=None,
                  channel_gains=None, channel_offsets=None,
                  store_time_array=False, compress=True):
    """Write a triloop HDF5 capture file.

    Parameters
    ----------
    path : str, output file path
    channels : dict mapping channel name -> 1D ndarray of samples.
               All arrays must have the same length.
    sample_rate : float, samples per second
    start_time_utc : datetime or str.  Defaults to now (UTC).
    scope_model, scope_serial : strings for metadata.
    capture_settings : dict, optional, serialized as JSON in metadata.
    loops_config : dict, optional, serialized as JSON in metadata.
    channel_gains, channel_offsets : dicts mapping channel name -> float.
                                     Used when storing int16; ignored
                                     for float32.
    store_time_array : bool.  If True, also store a /time dataset.
                              Otherwise time is regenerated from rate.
    compress : bool, gzip level 4 if True.
    """
    if start_time_utc is None:
        start_time_utc = datetime.now(timezone.utc).isoformat()
    elif isinstance(start_time_utc, datetime):
        start_time_utc = start_time_utc.isoformat()
    if capture_settings is None:
        capture_settings = {}
    if loops_config is None:
        loops_config = {}
    if channel_gains is None:
        channel_gains = {}
    if channel_offsets is None:
        channel_offsets = {}

    # Validate same-length channels
    lens = {len(v) for v in channels.values()}
    if len(lens) != 1:
        raise ValueError(f"All channels must be same length; got {lens}")
    n = lens.pop()

    duration_s = n / float(sample_rate)
    comp = "gzip" if compress else None
    comp_opts = 4 if compress else None

    with h5py.File(path, "w") as h:
        h.attrs["format_version"] = FORMAT_VERSION
        meta = h.create_group("metadata")
        meta.attrs["sample_rate"] = float(sample_rate)
        meta.attrs["start_time_utc"] = start_time_utc
        meta.attrs["duration_s"] = float(duration_s)
        meta.attrs["scope_model"] = scope_model
        meta.attrs["scope_serial"] = scope_serial
        meta.attrs["capture_settings"] = json.dumps(capture_settings)
        meta.attrs["loops_config"] = json.dumps(loops_config)

        ch_grp = h.create_group("channels")
        for name, arr in channels.items():
            ds = ch_grp.create_dataset(name, data=np.asarray(arr),
                                       compression=comp,
                                       compression_opts=comp_opts)
            ds.attrs["gain_v_per_count"] = float(channel_gains.get(name, 1.0))
            ds.attrs["offset_v"] = float(channel_offsets.get(name, 0.0))

        if store_time_array:
            t = np.arange(n) / sample_rate
            h.create_dataset("time", data=t, compression=comp,
                             compression_opts=comp_opts)


def read_capture(path):
    """Read a triloop HDF5 capture file.

    Returns
    -------
    dict with keys:
      sample_rate (float, Hz)
      duration_s (float)
      start_time_utc (str)
      scope_model (str), scope_serial (str)
      capture_settings (dict)
      loops_config (dict)
      channels (dict, channel-name -> ndarray of physical units (V))
      channel_raw (dict, same but raw int16/float32 if needed)
      time (ndarray, seconds)
      format_version (str)
    """
    with h5py.File(path, "r") as h:
        format_version = h.attrs.get("format_version", "unknown")
        meta = h["metadata"]
        sr  = float(meta.attrs["sample_rate"])
        dur = float(meta.attrs["duration_s"])
        start = str(meta.attrs.get("start_time_utc", ""))
        scope_model  = str(meta.attrs.get("scope_model", ""))
        scope_serial = str(meta.attrs.get("scope_serial", ""))
        capture_settings = json.loads(meta.attrs.get("capture_settings", "{}"))
        loops_config     = json.loads(meta.attrs.get("loops_config", "{}"))

        channels = {}
        channel_raw = {}
        for name in h["channels"].keys():
            ds = h["channels"][name]
            raw = ds[:]
            gain  = float(ds.attrs.get("gain_v_per_count", 1.0))
            offs  = float(ds.attrs.get("offset_v", 0.0))
            phys  = raw.astype(np.float64) * gain + offs
            channels[name] = phys
            channel_raw[name] = raw

        if "time" in h:
            t = h["time"][:]
        else:
            n = next(iter(channels.values())).size
            t = np.arange(n) / sr

    return dict(
        sample_rate=sr, duration_s=dur,
        start_time_utc=start,
        scope_model=scope_model, scope_serial=scope_serial,
        capture_settings=capture_settings,
        loops_config=loops_config,
        channels=channels, channel_raw=channel_raw,
        time=t, format_version=format_version,
    )
