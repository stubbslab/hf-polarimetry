# tools/

Auxiliary utilities that don't belong inside the `picoacq` or `triloop`
Python packages but are useful in the experimental workflow.

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

Tkinter GUI for the **Leo Bodnar Dual GPSDO** (the recommended GPS-locked
frequency synthesizer for HF phase-jitter experiments;
<https://www.leobodnar.com>).

```bash
python3 tools/bodnar_gui.py
```

Features:

- Two-output frequency control with sub-Hz precision.
- WWV band presets (2.5 / 5 / 10 / 15 / 20 / 25 MHz) on each output.
- CHU and ISM 13.56 MHz presets.
- Sub-Hz fine-adjust nudge buttons (± 1, 0.1, 0.01 Hz).
- GPS lock status + satellite count polling.
- Save/load configuration to JSON.

### Backends

Three transports are supported and selectable in the GUI:

| Backend | When to use | Install |
|---|---|---|
| Simulator | No hardware connected; learn the GUI | (none) |
| HID       | Current Bodnar firmware (USB HID) | `pip install hidapi` |
| Serial    | Older firmware / clones (USB-CDC TTY) | `pip install pyserial` |

### Important caveats (must read before science use)

The **HID backend's USB Vendor/Product IDs** and **packet layout** are
**placeholders**.  Before relying on this for science work:

1. Plug the Bodnar in.  Find the real VID/PID:
   ```bash
   # macOS
   system_profiler SPUSBDataType | grep -A 5 -i bodnar
   # Linux
   lsusb -v 2>/dev/null | grep -B 1 -A 5 -i bodnar
   ```
2. Download the Bodnar's user manual from <https://www.leobodnar.com>
   and confirm the HID feature-report register layout for setting
   the Si5351 fractional divider.
3. Update `HIDBackend.VID`, `HIDBackend.PID`, and the byte encoding
   in `HIDBackend.set_frequency_hz` accordingly.

Same applies to the **Serial backend's** ASCII command syntax — the
``F<output> <hz>\r\n`` form is illustrative only.  Confirm against the
manual or by capturing traffic from the vendor's Windows utility with
USBPcap before relying on it.

The GUI's plumbing (event loop, layout, presets, log, save/load) is
ready to use; only the wire-protocol details need verification once a
unit is in hand.  The Simulator backend is fully working and lets you
exercise the UI without hardware.
