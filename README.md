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
