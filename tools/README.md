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
| HID       | LBE-1425 (and likely LBE-1421) frequency control | `pip install hid hidapi` + `brew install hidapi` |
| Serial    | NMEA status read on units with a CDC interface (LBE-1421/1423; NOT the LBE-1425) | `pip install pyserial` |

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

The LBE-1425 exposes a single HID interface and only one process can
hold it open at a time.  The official Bodnar Windows / macOS app and
our tooling **cannot run simultaneously**; quit one before launching
the other.  This is normal HID-class device behavior, not a bug.

## bodnar_cli.py

Command-line wrapper around the same `HIDBackend` used by the GUI.
Useful for scripting and automation.

```bash
python3 tools/bodnar_cli.py                          # status read
python3 tools/bodnar_cli.py --out1 10MHz             # set output 1
python3 tools/bodnar_cli.py --out1 10MHz --out2 24kHz   # set both
python3 tools/bodnar_cli.py --out1 5MHz --temp       # RAM-only write
python3 tools/bodnar_cli.py --raw-status             # 64-byte hex dump
                                                      # (for protocol
                                                      # debugging)
python3 tools/bodnar_cli.py --status --json          # machine-readable
```

Frequency parsing accepts: `Hz`, `kHz`, `MHz` suffixes; bare numbers
treated as Hz; preset names like `wwv5`, `wwv10`, `chu7.85`, `ism`.

Default writes are **persistent** (written to the unit's flash).  Use
`--temp` for testing — the temporary opcodes do not touch flash and
revert on power cycle.

Exit codes:
- 0 : success
- 1 : device not found / connection error
- 2 : invalid arguments
- 3 : HID write or read failure
