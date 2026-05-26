# hf-polarimetry

Coherent multi-channel HF radio acquisition and polarimetric analysis for ionospheric work, built around a PicoScope 5444D digitizer driving a three-loop magnetic-field array.

The project ships three Python packages plus a small example dataset:

| Package | What it does |
|---|---|
| **`picoacq`** | Drives a PicoScope 5444D over USB, captures coherent 4-channel data into a self-describing HDF5 file. Default rate is 15.625 MS/s — chosen so all five WWV bands (2.5 / 5 / 10 / 15 / 20 MHz) appear at distinct baseband alias positions in a single recording. |
| **`triloop`** | Reads those HDF5 files; extracts narrow-band complex baseband at one or many RF bands; recovers the lab-frame magnetic-field vector via the inverse of the array geometry; computes Stokes parameters, ellipticity, position angle, polarization fraction; and ships interactive HTML browsing tools. |
| **`three_loop_array`** | The math + simulator: cube-vertex array geometry, perpendicular-plane decomposition, and a synthetic three-loop signal generator used by both packages. |

Plus an `examples/` directory containing a 0.25 s synthetic 5-band capture and a pre-rendered interactive HTML browser, so you can see what the analysis tools produce without buying any hardware.

## Quick look

```bash
git clone https://github.com/stubbslab/hf-polarimetry.git
cd hf-polarimetry
pip install -e './triloop[all]' -e ./picoacq

# Static QC plot
triloop view examples/example_5band_wwv.h5

# Per-band scientific summary
triloop analyze-multi examples/example_5band_wwv.h5 --az 9 --el 35

# Self-contained interactive HTML browser
triloop browse examples/example_5band_wwv.h5
```

For a full walk-through (including the optional PicoSDK install needed to run real hardware), see [`INSTALL.md`](INSTALL.md).

## End-to-end workflow

The toolchain has three stages: **capture → analyze → visualize**. Below is the standard recipe for each. All commands assume both packages are installed (see [`INSTALL.md`](INSTALL.md)) and you're at the repo root.

### 1. Data collection

Real hardware (PicoScope 5444D plugged in via USB-3, PicoSDK installed):

```bash
# One-shot 8-second capture of all five WWV bands at 15.625 MS/s,
# 4 channels, auto-ranged.  Writes capture_<UTC-timestamp>.h5 in the
# current working directory.
picoacq capture

# Specify output path, duration, channels explicitly:
picoacq capture -o my_capture.h5 --duration 4 --channels A,B,C,D

# Continuous unattended monitoring: 8 seconds out of every 10 minutes,
# stored under data/ with prefix "wwv".  Stop with Ctrl-C.
picoacq monitor -d data/ --interval 600 --duration 8 --prefix wwv
```

Without hardware, use the simulator — same file format, runs anywhere:

```bash
picoacq capture --simulate -o my_sim.h5 \
    --sim-pol rcp --sim-faraday 30 --sim-snr 25
```

The capture file's `metadata` group records sample rate, the auto-range probe results, the chosen Pico voltage range per channel, and the list of expected RF bands (`rf_bands_hz`) so the analysis side can locate every band's baseband alias automatically. `triloop summary <file.h5>` prints a one-screen overview of any capture.

### 2. Analysis

Three subcommands cover the common cases:

```bash
# Multi-band: run the polarimetry pipeline on every RF band recorded
# in the file (reads rf_bands_hz from capture_settings).  Writes a
# 5x4 comparison PNG and a per-band JSON summary.
triloop analyze-multi my_capture.h5 --az 9 --el 35

# With TX/RX coordinates: also project each band onto the predicted
# Appleton-Hartree O/X modes at the descending exit point of the ray.
# The JSON summary then carries a "magnetoionic" field per band with
# theta(k,B), |B|, predicted mode ellipticities, and the matched-filter
# amplitudes |a_O|, |a_X| and their differential phase.
triloop analyze-multi my_capture.h5 --az 9 --el 35 \
    --tx "40.6796,-105.0411" --rx "32.7803,-105.8200"

# Single-band: target a specific carrier frequency.  Useful when you
# only care about one tone or have a non-standard capture.
triloop analyze my_capture.h5 --carrier 25000 --bw 2000 --az 273 --el 12

# Direction finding: null-eigenvector + 2-D null sweep + parabolic
# refinement to recover (az, el) from the data itself.  Optional
# residual-map PNG.
triloop locate my_capture.h5 --carrier 25000 --bw 2000 --az0 270 --el0 15 \
    --out-png residual_map.png

# Batch mode: run analyze on every .h5 in a directory in parallel,
# producing one JSON line per file (loads cleanly into pandas).
triloop batch data/ --carrier 25000 --bw 2000 --az 9 --el 35 --workers 4
```

Output JSON contains f_peak, per-loop SNR, median polarization fraction / ellipticity / position angle, intensity statistics, and the recovered direction (for `locate`).

### 3. Visualization

```bash
# Static QC plot: full-Nyquist PSD with each RF band annotated at its
# baseband alias position, per-band zoom panels (axis flipped for
# inverted Nyquist zones), time-domain traces, and the auto-range
# probe table.  Writes <file>_view.png.
triloop view my_capture.h5

# Self-contained interactive HTML browser (Plotly): tabbed layout
# with full-Nyquist PSD, time series, and one tab per RF band
# containing a PSD zoom, intensity time history, intensity CDF with
# P10/P1 fade-depth markers, and an animated polarization ellipse
# in the (Re A_p, Re A_q) plane with a frame slider.  Writes a
# single .html file you can email or scp.
triloop browse my_capture.h5

# Jupyter notebook with the file path pre-loaded:
triloop notebook my_capture.h5
```

The interactive browser is a single HTML file with no server dependency — every panel pans, zooms, and exposes hover-readouts; the polarization-ellipse animation has a Play button and frame slider so Faraday rotation and time-varying ellipticity become directly visible. A pre-rendered version of the bundled example is at [`examples/example_5band_wwv_browse.html`](examples/example_5band_wwv_browse.html); to view it interactively, either clone the repo and open the file locally, or visit the GitHub Pages URL once the repo's Pages site is enabled (Settings → Pages → Source: `main` branch / root, then `https://stubbslab.github.io/hf-polarimetry/examples/example_5band_wwv_browse.html`).

## What's in the repo

```
hf-polarimetry/
├── README.md                            (this file)
├── INSTALL.md                           setup guide, including PicoSDK
├── LICENSE                              MIT
├── picoacq/                             acquisition package
│   ├── README.md
│   ├── pyproject.toml
│   ├── picoacq/                         source
│   │   ├── cli.py                       `picoacq capture / monitor`
│   │   ├── recorder.py                  capture orchestrator + auto-range
│   │   ├── ps5444a.py                   PicoSDK wrapper (ctypes)
│   │   ├── simulator.py                 no-hardware fallback
│   │   └── alias.py                     bandpass-undersampling math
│   └── docs/
│       └── picoacq_user_manual.pdf      hardware setup + CLI reference
├── triloop/                             analysis package
│   ├── README.md
│   ├── pyproject.toml
│   ├── triloop/                         source
│   │   ├── cli.py                       `triloop view / analyze-multi /
│   │   │                                 browse / analyze / locate / ...`
│   │   ├── io_hdf5.py                   capture file reader/writer
│   │   ├── geometry.py                  N-matrix, perp projector, basis
│   │   ├── extract.py                   single-band complex baseband
│   │   ├── bands.py                     multi-band extraction
│   │   ├── analyze.py                   polarimetry pipeline
│   │   ├── multiband.py                 per-band runner + comparison plot
│   │   ├── stokes.py                    Stokes parameter computation
│   │   ├── direction.py                 null-eigenvector direction finding
│   │   ├── beamform.py                  P(az, el) sweep
│   │   ├── view.py                      static QC PNG generator
│   │   ├── browse.py                    interactive HTML browser (Plotly)
│   │   └── config.py                    loops_config schema + default
│   ├── notebooks/
│   │   └── triloop_analysis.ipynb
│   ├── tests/
│   └── docs/
│       ├── triloop_user_manual.pdf      user-facing manual
│       └── triloop_analysis_ref.pdf     module-by-module dev reference
├── three_loop_array/                    geometry + simulator package
│   ├── code/
│   │   ├── three_loop.py
│   │   ├── draw_geometry.py
│   │   └── visualize_sim.py
│   └── three_loop_array_report.pdf      technical report on array design
└── examples/
    ├── example_5band_wwv.h5             58 MB synthetic capture
    ├── example_5band_wwv_browse.html    pre-rendered interactive browser
    └── make_example_capture.py          regenerate the example
```

## What's *not* in the repo

This is a fresh release tree. It deliberately omits raw recordings, intermediate plots, scratch notebooks, and unrelated tools that lived alongside the development repo. The MIT license covers only the code committed here.

## Documentation

- [`INSTALL.md`](INSTALL.md) — install both packages and the PicoSDK system library.
- [`picoacq/docs/picoacq_user_manual.pdf`](picoacq/docs/picoacq_user_manual.pdf) — hardware reference, CLI, undersampling alias map, troubleshooting.
- [`triloop/docs/triloop_user_manual.pdf`](triloop/docs/triloop_user_manual.pdf) — analysis-side user manual, including all `view` / `analyze-multi` / `browse` subcommands.
- [`triloop/docs/triloop_analysis_ref.pdf`](triloop/docs/triloop_analysis_ref.pdf) — module-by-module developer reference covering both packages.
- [`three_loop_array/three_loop_array_report.pdf`](three_loop_array/three_loop_array_report.pdf) — technical report on the cube-vertex array geometry and SNR analysis.

## Citing / contact

If this code is useful to you in a publication, please cite the technical report (`three_loop_array_report.pdf`) and link to this repository.

Issues and pull requests welcome on GitHub.

## License

MIT — see [`LICENSE`](LICENSE).
