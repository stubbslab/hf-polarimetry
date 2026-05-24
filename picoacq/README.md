# picoacq

PicoScope 5444D data acquisition for the [triloop](../triloop/) analysis
toolchain.  Writes the same HDF5 capture format that triloop reads.

## Install

```bash
cd picoacq
pip install -e .                    # software only
# or, with hardware support:
pip install -e .[hardware]          # also installs picosdk
```

Hardware mode also requires the PicoSDK system installer
(https://www.picotech.com/downloads).

## Quick start

```bash
# Default: 8-second 4-channel capture at 15.625 MS/s, auto-ranged.
# This rate is chosen so all 5 WWV bands (2.5/5/10/15/20 MHz) appear at
# distinct baseband alias positions in the same recording.
picoacq capture                     # writes capture_<UTC>.h5 to cwd

# Real-hardware capture with explicit options
picoacq capture -o capture.h5 --duration 8 --channels A,B,C,D

# Simulator (no hardware needed)
picoacq capture -o capture.h5 --simulate \
    --sim-pol rcp --sim-faraday 30 --sim-snr 25

# Continuous monitor (8 s every 10 minutes)
picoacq monitor --interval 600 --duration 8 --prefix wwv
```

If real-hardware capture fails (driver not installed, scope not
connected), `picoacq` automatically falls back to the simulator with a
warning.

## What's new in v0.1

- **Default rate is 15.625 MS/s** so all WWV bands are captured
  simultaneously through bandpass undersampling — see the user manual
  for the alias map.
- **`--rf-bands`** flag stores the list of expected RF frequencies
  (default: `2.5e6,5e6,10e6,15e6,20e6`) into the file's
  `capture_settings` so downstream tools can map detected baseband
  peaks back to RF.
- **Auto-ranging is on by default**: a brief 100 ms probe per
  capture chooses the smallest Pico voltage range with a 3× headroom
  margin above the measured peak.
- **Onboard buffer guard**: `recorder.capture()` raises a clear
  `ValueError` if the requested duration would exceed the 5444D's
  onboard memory (128 MS/ch in 4-channel mode → ≤ 8.19 s at 15.625 MS/s).
- **`picoacq.alias`** module: `alias_of(rf, fs)` and
  `rf_for_baseband(...)` for converting between RF and baseband under
  bandpass undersampling.

## File format

`picoacq` writes the **triloop HDF5 capture format** (v0.1).  See
`triloop/triloop/io_hdf5.py` for the schema; both packages share the
same reader/writer code.

## Documentation

- `docs/picoacq_user_manual.pdf` — hardware setup, software install,
  CLI reference, alias map for the 15.625 MS/s default, and
  troubleshooting.
- `../triloop/docs/triloop_user_manual.pdf` — analysis side, including
  the new `view`, `analyze-multi`, and `browse` subcommands.
- `../triloop/docs/triloop_analysis_ref.pdf` — module-by-module
  developer reference for both packages.
