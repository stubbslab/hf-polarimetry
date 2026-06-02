# tools/

Auxiliary utilities that don't belong inside the `picoacq` or `triloop`
Python packages but are useful in the experimental workflow.

## predict_phase_batch.py

Batch driver that runs `predict_phase.compute_predictability` across an
entire directory of KiwiSDR IQ recordings, parses each filename for
date/time/freq/port, and produces:

- `predict_phase_results.jsonl` — one record per file
- `predict_phase_dpred20ms_vs_freq.png` — boxplot of `D_pred(20 ms)` per band
- `predict_phase_dpred20ms_vs_hour.png` — median `D_pred(20 ms)` vs UTC hour
- `predict_phase_correctable_fraction.png` — % of files achieving the
  0.5 rad threshold per band

```bash
python3 tools/predict_phase_batch.py /path/to/wav_dir \
    --pattern '*wwv*.wav' \
    --workers 4 \
    --out-dir /tmp/predict_batch_out
```

The Kalman predictor is skipped by default (slow); pass `--with-kalman`
to enable it.  On constrained systems where `ProcessPoolExecutor` can't
allocate semaphores, the driver falls back to a serial loop
automatically.  Quality cut: files with per-segment slope std exceeding
`--max-slope-std-hz` (default 5000 Hz) are excluded as having
phase-unwrap failures rather than meaningful prediction signals.

## predict_phase.py

Phase-prediction analysis on KiwiSDR IQ recordings of WWV (or any
single-tone HF carrier in 2-channel int16 IQ format).

```bash
python3 tools/predict_phase.py path/to/wwv_iq.wav
```

Compares three predictors of the future phase track:
- **constant-phase** (do-nothing baseline; this *is* the descriptive
  $D(\tau)$ structure function),
- **linear extrapolation** (least-squares fit on a moving 50 ms window,
  extrapolated; nails out a constant Doppler shift),
- **2-state Kalman filter** on $(\varphi, \omega)$ with tunable
  process noise.

For each predictor it computes the prediction-error structure function
$D_{\rm pred}(\tau) = \langle [\varphi(t+\tau) - \hat\varphi(t+\tau\,|\,t)]^2\rangle$
averaged across all high-SNR segments in the file, then plots
$\sqrt{D_{\rm pred}(\tau)}$ for each predictor on the same axes.  An
adaptive HF correction system's lower bound on closed-loop residual
phase error is given by the optimum predictor.

Output:  ``<input>_predict.png`` and ``<input>_predict.json`` next to
the input file.

Useful for comparing predictability across bands, between fades, or as
the input model for a future closed-loop correction experiment.  Tune
``--kalman-q-omega`` (default 1e3, but values in the range 1–100 often
work better on long-coherent captures) and ``--linear-fit-window-s``
(default 50 ms) to find the optimum for your data.

## bodnar_gui.py

Tkinter GUI for the **Leo Bodnar LBE-1425 Dual GPSDO** (the recommended
GPS-locked frequency synthesizer for HF phase-jitter experiments;
<https://www.leobodnar.com>).

```bash
python3 tools/bodnar_gui.py
```

Features:

- Two-output frequency control with 1 Hz precision (this firmware
  revision; see protocol notes below).
- WWV band presets (2.5 / 5 / 10 / 15 / 20 / 25 MHz) on each output.
- CHU and ISM 13.56 MHz presets.
- Sub-Hz fine-adjust nudge buttons (display only on this firmware --
  underlying writes are integer-Hz).
- GPS / PLL / antenna lock status from the HID status feature report.
- Save/load configuration to JSON.

### Backends

| Backend | When to use | Install |
|---|---|---|
| Simulator | No hardware connected; learn the GUI | (none) |
| HID       | LBE-1425 (and likely LBE-1421) frequency control + lock booleans | `pip install hid hidapi` + `brew install hidapi` |
| Serial    | NMEA stream from the unit's CDC interface — UTC time, lat/lon, sat count, HDOP. Available on LBE-1425, LBE-1421, LBE-1423. | `pip install pyserial` |

The HID and CDC interfaces are independent USB endpoints and can both
be open simultaneously; the GUI uses both, and `bodnar_cli.py` opens
both on each `--status` read.

### Wire-protocol verification status

The HID protocol used by `HIDBackend` has been **empirically verified**
against an actual LBE-1425 (factory firmware, 2026 timestamp).  See
[BODNAR_LBE1425_PROTOCOL.md](BODNAR_LBE1425_PROTOCOL.md) for:

- USB VID / PID
- Feature-report layouts (read-back at byte 1 status / bytes 6-9 freq1 /
  bytes 14-17 freq2; write payload at bytes 6-9, integer u32 only)
- Set-frequency opcodes (0x05/0x06 for output 1 temp/persist;
  0x09/0x0A for output 2)
- Status bitmask decoding (GPS / PLL / antenna / output enable / PPS)
- Verification log with before/after byte dumps for every flag we
  decoded
- What's NOT yet decoded (constellation enable, satellite count, etc.)

If you have a different Bodnar SKU or a substantially different firmware
revision, re-verify before trusting science results.  The protocol doc
includes a step-by-step procedure for diffing a new feature against the
verified baseline.

### Hardware exclusivity

The LBE-1425 exposes a HID interface (control / lock state) and a
separate USB CDC serial interface (NMEA stream).  Within each
interface, only one process can hold it open at a time.  Specifically:

- The **official Bodnar app** and our **HIDBackend** both want the HID
  endpoint exclusively.  Quit one before launching the other.
- The **CDC NMEA endpoint** also serves only one reader at a time, so
  if the official app is reading it, our `bodnar_cli.py --status` won't
  see GPS data (it'll print `(could not open CDC NMEA port)` and skip).

This is normal USB-class behavior, not a bug.

## bodnar_cli.py — command-line interface

Command-line front-end to the same `HIDBackend` and `SerialBackend`
used by the GUI.  Useful for scripts, automation, and quick bench
tests.  All three of:

- HID frequency control (write to flash or RAM-only)
- HID status read (lock booleans, frequencies, FLL/low-power flags)
- CDC NMEA read (UTC time, lat/lon, satellite counts, HDOP)

are folded into one tool.

### Synopsis

```bash
python3 tools/bodnar_cli.py [--out1 FREQ] [--out2 FREQ]
                            [--temp] [--quiet]
                            [--status] [--no-gps] [--json]
                            [--raw-status]
```

When invoked with no flags, `bodnar_cli.py` prints a full status
report (HID + GPS).  When invoked with `--out1` and/or `--out2`, it
sets the requested frequencies, then prints status to confirm.

### Examples

#### Read full status (the default action)

```bash
python3 tools/bodnar_cli.py
```

Sample output:

```
Bodnar LBE-1425  status:
  locks:        GPS, PLL, ANT
  raw status:   0x7f
  output 1:     3.000000 MHz  [enabled, normal]
  output 2:     2.500000 MHz  [enabled, normal]
  FLL mode:     off (PLL mode -- recommended)

GPS / NMEA  status (read from /dev/cu.usbmodem...):
  fix:          3D fix
  UTC:          2026-06-02  15:06:53.00 UTC
  position:     +42.397813°N  -71.375297°E
  altitude:     34.6 m
  sats used:    12
  sats in view: GP=12  GL=10
  SNR:          GL med=33 dB,  GP med=28 dB
  HDOP:         0.66
```

#### Read only HID status (skip CDC, faster)

```bash
python3 tools/bodnar_cli.py --no-gps
```

#### Machine-readable JSON

```bash
python3 tools/bodnar_cli.py --status --json
```

Returns one or two JSON objects on stdout (one per status block).

#### Set output 1 to a WWV preset

```bash
python3 tools/bodnar_cli.py --out1 wwv10
python3 tools/bodnar_cli.py --out1 10MHz       # equivalent
python3 tools/bodnar_cli.py --out1 10000000    # equivalent (bare = Hz)
```

#### Set both outputs in one call (writes are persistent by default)

```bash
python3 tools/bodnar_cli.py --out1 10MHz --out2 24.000kHz
```

#### Set output 1 temporarily (does NOT write to flash)

```bash
python3 tools/bodnar_cli.py --out1 5MHz --temp
```

The unit reverts to its persistent value (last `--persist` write) on
the next power cycle.  Use `--temp` for development and testing to
avoid flash erase cycles.

#### Quiet writes for shell scripts

```bash
python3 tools/bodnar_cli.py --out1 10MHz --quiet || echo "Bodnar write failed"
```

Exit codes (suitable for shell tests):
- `0` : success
- `1` : device not found / connection error
- `2` : invalid argument (unparseable frequency, etc.)
- `3` : HID write or read failure

#### Raw HID status dump (protocol debugging only)

```bash
python3 tools/bodnar_cli.py --raw-status
```

Prints the full 60-byte HID status response in hex.  Useful only when
verifying the protocol against a new unit or new firmware revision.
The byte map is documented in `BODNAR_LBE1425_PROTOCOL.md`.

### Frequency parsing

The `--out1` / `--out2` arguments accept any of:

| Format | Example | Hz value |
|---|---|---|
| Plain Hz | `10000000` | 10,000,000 |
| Hz with unit | `10000000 Hz` | 10,000,000 |
| kHz | `24.123kHz` or `24123Hz` | 24,123 |
| MHz | `10MHz` or `10.5MHz` | 10,000,000 / 10,500,000 |
| Scientific | `1.234e6` | 1,234,000 |
| WWV preset | `wwv2.5`, `wwv5`, `wwv10`, `wwv15`, `wwv20`, `wwv25` | 2.5/5/10/15/20/25 MHz |
| CHU preset | `chu3.33`, `chu7.85`, `chu14.67` | 3.33/7.85/14.67 MHz |
| ISM preset | `ism` | 13.560 MHz |

Note: this firmware revision accepts only **integer Hz** writes.  Any
fractional Hz in your input is silently truncated by `int(hz)`.
Practical precision is therefore 1 Hz; the underlying GPS-disciplined
oscillator is much more stable than that.

### Persistence

Writes default to **persistent** mode (flash-backed; the LBE-1425's
permanent set-frequency opcodes 0x06 / 0x0A).  The unit will reload
the same frequencies on next power-up.

Use `--temp` to use the temporary set-frequency opcodes (0x05 / 0x09)
which take effect immediately but do not touch flash.  Recommended
for any iterative or experimental work.

### Common gotchas

- **macOS dylib load error**: install hidapi via `brew install hidapi`.
  The `pip install hid hidapi` Python wrappers don't bundle the native
  library on macOS.
- **"could not open port" or "device not found"**: another tool
  (typically the official Bodnar app) is holding the device.  Quit it.
- **GPS shows `(could not open CDC NMEA port)`**: same reason as
  above, but for the CDC endpoint.  Either quit the conflicting app
  or run `bodnar_cli.py --no-gps` to skip the CDC read.
- **Persistent writes don't seem to stick across power cycles**:
  confirm the unit is actually power-cycled (USB unplug + 10 s wait +
  reconnect).  A `bodnar_cli.py --status` read straight after power-up
  will tell you what's actually in flash.
