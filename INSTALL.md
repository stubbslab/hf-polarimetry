# Installation

There are three layers to install:

1. **Python packages** (`triloop` and `picoacq`) — required for everything.
2. **PicoSDK system library** — required only if you want to drive a real PicoScope. Skip it if you only need the simulator and analysis tools.
3. **Optional extras** — Jupyter for the analysis notebook, Plotly for the interactive browser.

You can stop after step 1 and use the toolchain end-to-end against the bundled `examples/example_5band_wwv.h5` capture. The PicoSDK install is only needed when you want to record live data.

---

## 1. Python packages

```bash
git clone https://github.com/stubbslab/hf-polarimetry.git
cd hf-polarimetry

# Install both packages, editable so you can hack on them
pip install -e ./triloop
pip install -e ./picoacq

# Optional: also install the interactive HTML browser dependency
pip install -e './triloop[browser]'

# Optional: also install Jupyter for the analysis notebook
pip install -e './triloop[notebook]'

# Or grab everything optional:
pip install -e './triloop[all]'
```

Recommended Python: 3.10 or later. Required dependencies (numpy, scipy, h5py, matplotlib, click) are pulled in automatically.

### Verify

```bash
triloop --help
picoacq --help
triloop summary examples/example_5band_wwv.h5
```

The first two should print help. The third should report sample rate (15.625 MS/s), duration (0.25 s), and four channels (A/B/C/D). If those work, the analysis side is fully installed.

### Run the analysis tools on the bundled example

```bash
# Static QC plot (writes examples/example_5band_wwv_view.png)
triloop view examples/example_5band_wwv.h5

# Per-band scientific summary (writes ..._multiband.png and ..._multiband.json)
triloop analyze-multi examples/example_5band_wwv.h5 --az 9 --el 35

# Interactive HTML browser (open the resulting .html file in any browser)
triloop browse examples/example_5band_wwv.h5
```

The browser output is also pre-rendered as `examples/example_5band_wwv_browse.html` — open that directly to see what the tool produces without installing Plotly.

---

## 2. PicoSDK system library (optional, hardware only)

`picoacq` drives a PicoScope 5444D over USB. The Python wrapper (`picosdk`) needs a system-level shared library installed via Pico Technology's installer.

### A. Install PicoScope 7 (the GUI app)

Useful for first-light hardware checks before any code runs.

1. Visit <https://www.picotech.com/downloads>.
2. Download **PicoScope 7** for your operating system (NOT PicoSDK — that comes next).
3. **macOS:** mount the `.dmg`, drag `PicoScope.app` into `/Applications`. The first launch will ask macOS to authorize the user-space USB plugin; allow it.
4. **Linux:** install the `.deb` or `.rpm` from Pico's repo.
5. **Windows:** run the `.exe` installer.
6. Plug a 5444D into a USB-3 port. Launch PicoScope 7. Confirm you see the unit in the device list and can take a few traces. Quit before continuing — the GUI app and `picoacq` cannot share the device handle.

### B. Install PicoSDK (the C library)

This installs the `libps5000a.{dylib,so,dll}` shared library that `picosdk` calls into.

1. Same downloads page <https://www.picotech.com/downloads>.
2. Download **PicoSDK** (separate from PicoScope 7).
3. Run the installer for your OS:
   - **macOS:** `.pkg` installer puts the framework at `/Library/Frameworks/PicoSDK.framework/Libraries/`. `picoacq` injects this path into `DYLD_LIBRARY_PATH` automatically; no further setup needed.
   - **Linux:** the `.deb` puts shared libs in `/opt/picoscope/lib/`. Add this to `LD_LIBRARY_PATH` if your distribution's loader doesn't pick it up automatically.
   - **Windows:** `.exe` installer drops DLLs into a directory on the system PATH.

Verify the library is reachable:

```bash
# macOS
sudo find / -name "libps5000a*" 2>/dev/null

# Linux
ldconfig -p | grep ps5000a
```

### C. Install the `picosdk` Python wrapper

```bash
python3 -m pip install picosdk
```

The `python3 -m pip install` form is recommended over plain `pip install` because some systems (especially macOS with a python.org install alongside Apple's Python) have multiple Python installations whose `pip` resolves to a different one than the `python3` you actually use.

Verify:

```bash
python3 -c "from picosdk.ps5000a import ps5000a; print('SDK OK')"
```

### D. First hardware capture

```bash
# Plug in the 5444D, quit PicoScope 7 if running, then:
picoacq capture --duration 0.5 --channels A,B,C,D
```

This should print an auto-range probe table (peak/RMS in mV per channel and the chosen Pico voltage range), then write a `capture_<UTC>.h5` file in the current directory. If you instead see `picosdk not installed` or `PicoSDK ... not found`, recheck the steps above — the library install is by far the most common stumbling point.

If the SDK or hardware is unavailable, `picoacq` prints a one-line warning and silently falls back to the simulator. In that mode you still get a valid HDF5 file with synthetic data; the `capture_settings.used_simulator` field will be `true`.

---

## 3. Optional extras

### Plotly (for `triloop browse`)

```bash
pip install plotly
# or
pip install -e './triloop[browser]'
```

If Plotly isn't installed, `triloop view` and `triloop analyze-multi` still work fine; only `browse` requires it.

### Jupyter (for the analysis notebook)

```bash
pip install jupyterlab ipywidgets
# or
pip install -e './triloop[notebook]'
```

Then:

```bash
triloop notebook examples/example_5band_wwv.h5
```

opens the bundled analysis notebook with the file path pre-loaded.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `picosdk not installed or PicoSDK system library not found` | One of the two PicoSDK install steps was skipped | Re-run §2.B and §2.C; check `find / -name 'libps5000a*'` |
| `PICO_NOT_FOUND` from any `picoacq` call | PicoScope 7 (or another picoacq) is holding the handle | Quit other consumers, replug USB |
| `PICO_INVALID_NUMBER_CHANNELS_FOR_RESOLUTION` | Hand-set 16-bit mode with 4 channels | Use the default — `picoacq` auto-selects 14-bit for 4 channels |
| `requested duration exceeds onboard buffer` from `picoacq capture` | Block-mode read won't fit in the 5444D's 128 MS/ch memory | Lower `--duration`, or drop to 2 channels |
| `triloop browse` errors on `import plotly` | Plotly extra not installed | `pip install plotly` |
| HTML browser pages render blank | Adobe Reader has known PDF rendering quirks; for HTML, try a different browser | Open in Safari/Firefox/Chrome instead |

For deeper hardware-side troubleshooting, see `picoacq/docs/picoacq_user_manual.pdf`.
For analysis-side internals, see `triloop/docs/triloop_user_manual.pdf` and `triloop/docs/triloop_analysis_ref.pdf`.
