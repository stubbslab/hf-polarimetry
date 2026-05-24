# triloop

Three-loop magnetic-field array analysis for HF polarimetry.

`triloop` reads an HDF5 capture file containing time-aligned signals
from three orthogonal magnetic-loop antennas, extracts the narrow-band
complex baseband at one or more target carriers (or all WWV bands at
once for multi-band undersampled captures), recovers the lab-frame
magnetic field vector, and computes polarization-independent intensity,
Stokes parameters, ellipticity, position angle, and an instantaneous
amplitude / phase / frequency trace on the dominant polarization
component.

## Install

```bash
cd triloop
pip install -e .                 # editable install
pip install -e .[notebook]       # also installs jupyter
pip install -e .[browser]        # also installs plotly (for `triloop browse`)
```

Pulls in: numpy, scipy, h5py, matplotlib, click.  Plotly is required
only for the interactive HTML browser.

## Quick start: CLI

```bash
# Generate a synthetic 4-channel capture (no hardware needed)
picoacq capture --simulate -o cap.h5

# Print metadata
triloop summary cap.h5

# Static QC plot (full-Nyquist PSD + per-band zooms + probe table)
triloop view cap.h5

# Per-band scientific summary across every RF band recorded in the file
triloop analyze-multi cap.h5 --az 9 --el 35

# Interactive HTML browser: pan/zoom + intensity CDF + animated
# polarization ellipse, all in a single self-contained .html file
triloop browse cap.h5

# Single-band analysis (when you only care about one carrier)
triloop analyze cap.h5 --carrier 25000 --bw 2000 --az 273 --el 12

# Single-band direction finding via null-eigenvector + null sweep
triloop locate cap.h5 --carrier 25000 --bw 2000 --az0 270 --el0 15

# Open the Jupyter analysis notebook with the file pre-loaded
triloop notebook cap.h5
```

## Quick start: Python

```python
from triloop import read_capture, analyze, analyze_all_bands

cap = read_capture("cap.h5")

# Single-band:
B1, B2, B3 = (cap["channels"][c] for c in ("A", "B", "C"))
res = analyze(cap["time"], B1, B2, B3,
              f0=25_000.0, BW=2_000.0,
              az_deg=273.0, el_deg=12.0)
print(res.f_peak, res.median_pol_fraction)

# Multi-band: read rf_bands_hz from capture_settings, run polarimetry
# on every band, return a dict[rf_hz -> MultiBandResult].
results = analyze_all_bands(cap, az_deg=9.0, el_deg=35.0)
for f_rf, r in sorted(results.items()):
    a = r.analysis
    print(f"{f_rf/1e6:5.2f} MHz  pol_frac={a.pol_fraction.mean():.2f}  "
          f"ellip={a.ellipticity_deg.mean():+.1f}°")
```

## What's new in v0.1

- **`triloop view`** — single-PNG QC plot.
- **`triloop analyze-multi`** — runs the polarimetry pipeline on every
  RF band recorded in the capture (reads `capture_settings.rf_bands_hz`).
- **`triloop browse`** — interactive single-file HTML browser
  (Plotly): tabbed layout with full-Nyquist PSD, time series, and a
  per-band tab containing PSD zoom, intensity time history, intensity
  CDF with P10/P1 fade-depth markers, and an animated polarization
  ellipse with a frame slider.
- **`triloop.bands.extract_bands()`** — multi-band baseband extractor
  that uses `picoacq.alias` to find the right baseband location for
  each RF band and conjugates inverted-zone slices.
- **`triloop.analyze_z_loops()`** — inner pipeline factored out so
  pre-extracted complex baseband can be analyzed without re-mixing.

## File format

HDF5 with a small fixed schema; see `triloop/io_hdf5.py` for the spec.
Both `triloop` and `picoacq` read/write this format.  When the file is
a multi-band capture, `capture_settings.rf_bands_hz` lists the RF
frequencies of interest so analysis tools can locate each band's
baseband alias.

## Loop geometry

Default geometry is the cube-vertex configuration: three loops on three
faces of a cube whose body diagonal is vertical.  Loop normals tilt
35.26° above horizontal and are spaced 120° apart in azimuth.  L1's
normal points N+up by convention.

To override: edit `loops_config` in the notebook (cell 2), or pass
`loops_config=` to `analyze()`.  See `triloop/config.py` for the
schema.

## Documentation

- `docs/triloop_user_manual.pdf` — full user manual.
- `docs/triloop_analysis_ref.pdf` — module-by-module developer
  reference covering both `triloop` and `picoacq`.
- `../picoacq/docs/picoacq_user_manual.pdf` — capture-side hardware
  manual.
