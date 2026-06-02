# Leo Bodnar LBE-1425 GPSDO — USB HID protocol notes

These are working notes for the USB HID interface of the **Leo Bodnar
LBE-1425 Mini Precision GPS Reference Clock (Dual)** as implemented in
this project's `bodnar_gui.py` / `bodnar_cli.py`.

The protocol is **not officially documented by the vendor**.  We
reverse-engineered enough of it for our needs by:

  1. Cross-referencing two open-source projects:
     - [`bvernoux/lbe-142x`](https://github.com/bvernoux/lbe-142x)
       (C, GPL; covers LBE-1420/1421/1423/Mini)
     - [`jjcarrier/gpsdo`](https://github.com/jjcarrier/gpsdo)
       (C#/.NET; explicit two-output support)
  2. **Empirically toggling settings** in the official Bodnar app and
     diffing the resulting status feature-report bytes.  See the
     [Verification log](#verification-log) below for the actual
     before/after byte dumps.

The two open-source references **disagree** on byte offsets and on
whether the frequency payload is integer-only or Q32.32.  Our LBE-1425
running its current factory firmware revision uses the **integer-only**
encoding at byte offset 6.  Different firmware revisions of the same
chassis may use different layouts; if you have a different unit,
re-run the verification procedure below.

---

## USB identifiers

| Model | VID | PID |
|---|---|---|
| **LBE-1425 (Dual)** | **0x1DD2** | **0x2269** |
| LBE-1421 (Dual)      | 0x1DD2 | 0x2444 |
| LBE-1420 (single)    | 0x1DD2 | 0x2443 |

The unit enumerates as a **HID-class** device.  On the LBE-1425 we
have NOT seen a separate CDC serial interface — all status and control
goes over the single HID endpoint.

**Exclusivity:** only one process can hold the HID device open at a
time.  The official Bodnar app and our tool **cannot run simultaneously**;
quit one before launching the other.

---

## HID feature reports

All commands and queries are sent as 64-byte HID **feature reports**.

```
byte 0     : report ID                  (always 0x00 in our usage)
byte 1     : opcode                     (defines what kind of report)
bytes 2..N : payload                    (opcode-dependent layout)
bytes N+1..63 : padding (zero on writes)
```

### Set-frequency opcodes (writes)

| Opcode | Meaning                                | Persistence |
|---|---|---|
| **0x05** | Set output 1 frequency, **temporary** | RAM only — reverts on power cycle |
| **0x06** | Set output 1 frequency, **persistent** | written to flash |
| **0x09** | Set output 2 frequency, **temporary** | RAM only |
| **0x0A** | Set output 2 frequency, **persistent** | written to flash |

#### Frequency payload (LBE-1425 firmware revision verified empirically)

```
buf[0]    = 0x00              (report ID)
buf[1]    = opcode             (one of the four above)
buf[2..5] = 0                  (reserved)
buf[6..9] = hz_int             (little-endian uint32, integer Hz)
buf[10..63] = 0                (padding)
```

**Sub-Hz precision is NOT available** with this firmware via these
opcodes.  Frequencies are written as plain little-endian u32 values
in Hz.

> **Note:** the C# `jjcarrier/gpsdo` reference describes a Q32.32
> encoding at offset 2, with 4 fractional bytes and 4 integer bytes.
> Our LBE-1425 does NOT use that layout — empirically the firmware
> reads the integer u32 starting at byte 6.  If you naively follow
> the C# reference your frequency comes out off by a factor of 4096
> (12-bit shift).

### Status read

Issuing a **GET feature report** with **report ID 0x4B** returns a
60-byte response.  Parse as follows.

| Byte | Meaning |
|---|---|
| 0 | echo of the report ID (`0x01`-ish on our unit; not parsed) |
| **1** | **Status bitmask** (see below) |
| 2..5 | reserved (zero) |
| **6..9** | **Output 1 frequency**, little-endian uint32, Hz |
| 10..13 | reserved (zero) |
| **14..17** | **Output 2 frequency**, little-endian uint32, Hz |
| **18** | **FLL enabled** (1 = on, 0 = off; PLL mode when 0) |
| **19** | **Output 1 low-power** (1 = low, 0 = normal) |
| **20** | **Output 2 low-power** (1 = low, 0 = normal) |
| 21..23 | constant `0x67 0x02 0x05` on our unit; **identity unknown** but stable across all toggles we tried |
| 24..59 | filled with `0xff` (uninitialized buffer tail) |

#### Status bitmask (byte 1)

| Bit | Mask | Meaning |
|---|---|---|
| 0 | `0x01` | GPS_LOCKED — GNSS module has a 3-D fix |
| 1 | `0x02` | PLL_LOCKED — Si5351 disciplining loop is locked |
| 2 | `0x04` | ANT_OK — antenna current is in spec (not shorted, not open) |
| 3 | `0x08` | OUT1_LED — front-panel output-1 indicator state |
| 4 | `0x10` | OUT2_LED — front-panel output-2 indicator state |
| 5 | `0x20` | OUT1_EN — output 1 enabled |
| 6 | `0x40` | OUT2_EN — output 2 enabled |
| 7 | `0x80` | PPS_EN — 1 PPS output enabled |

A healthy unit reads `0x7f` (everything but PPS) or `0xff` (everything,
if PPS is also enabled).

---

## Things the protocol does NOT expose (yet)

We have **not** reverse-engineered:

- **GNSS constellation enable bits** (GPS / GLONASS / Galileo / BeiDou /
  QZSS / SBAS).  These almost certainly live in a separate feature
  report that we haven't identified.  **Set constellations through the
  official Bodnar app**; settings persist across power cycles and our
  tool does not disturb them.
- **Satellite count** / **DOP** / **lat-lon-alt** — not in the 0x4B
  report.  The unit might expose these via NMEA on a CDC interface
  on other models, but we haven't seen a CDC interface on the LBE-1425.
- **DAC value / oscillator steering value** — not exposed.
- **Antenna bias-current measurement** — only the boolean ANT_OK bit.
- **Firmware version** / **serial number** — available via HID
  descriptor (vendor / product / serial-number strings) but not in
  the 0x4B report.
- **Save-to-flash / factory-defaults** — no opcode discovered.  The
  per-output `0x06` and `0x0A` opcodes already write to flash, so
  there's no separate "commit" required.
- **Antenna power on/off**.
- **PLL/FLL mode toggle** — `bvernoux/lbe-142x` documents
  `LBE_142X_SET_PLL = 0x0B` with payload `{0x01}` for FLL, `{0x00}` for
  PLL.  Not confirmed on our unit; the C reference is for the
  LBE-1421, which is closely related but not identical.

---

## Verification log

The following was captured against an actual LBE-1425, factory firmware
(2026 timestamp), with frequencies preset by the official Bodnar app
to **3 MHz on output 1** and **2.5 MHz on output 2**.

### Baseline: out1 = 3 MHz normal, out2 = 2.5 MHz normal, FLL off

```
byte:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
data: 01 7f 00 00 00 00 c0 c6 2d 00 00 00 00 00 a0 25
byte: 16 17 18 19 20 21 22 23 24
data: 26 00 00 00 00 67 02 05 00
```

Decode:
- byte 1 = `0x7f`: GPS + PLL + ANT + LED1 + LED2 + EN1 + EN2 (no PPS)
- bytes 6..9 = `c0 c6 2d 00` LE = 0x002dc6c0 = **3,000,000 Hz** ✓
- bytes 14..17 = `a0 25 26 00` LE = 0x002625a0 = **2,500,000 Hz** ✓
- bytes 18, 19, 20 = 0: FLL off, both outputs normal-power
- bytes 21..23 = `67 02 05`: unknown constant

### Toggle: out1 -> low-power

```
byte: 16 17 18 19 20 21 22 23
data: 26 00 00 01 00 67 02 05
                 ^^
             byte 19: 0 -> 1
```

→ **Byte 19 = output-1 low-power flag.**

### Toggle: out2 -> low-power (out1 returned to normal)

```
byte: 16 17 18 19 20 21 22 23
data: 26 00 00 00 01 67 02 05
                    ^^
                byte 20: 0 -> 1
```

→ **Byte 20 = output-2 low-power flag.**

### Toggle: FLL enabled (both outputs back to normal)

```
byte: 16 17 18 19 20 21 22 23
data: 26 00 01 00 00 67 02 05
              ^^
          byte 18: 0 -> 1
```

→ **Byte 18 = FLL flag.**

The triplet `67 02 05` at bytes 21..23 did not change in any of the
four cases, suggesting it's a hardware/firmware identity constant and
not configuration state.

### Frequency-write verification

After a `--temp` write of 5 MHz to output 1, the read-back showed
**5.000000 MHz** exactly.  Power-cycling the unit reverted output 1
to its persistent 3 MHz value, confirming temp-vs-persist semantics.

---

## Practical / safety notes

- **Default to `--temp` writes** during development and testing.
  Persistent writes consume flash erase cycles and a misdirected
  opcode can scramble configuration.
- **Power-cycle the unit** after any unexpected behavior; this clears
  RAM-resident temp state and reloads from flash.  No data is lost.
- **One HID claimer at a time.**  When the official Bodnar app holds
  the device, our tool can't read status (and vice versa).  This is
  not a bug — it's standard HID behavior.
- The `ANT_OK` bit reported as false (raw status `0x7B` instead of
  `0x7F`) does NOT necessarily mean a hardware short.  It can also
  result from another process holding the HID device, leaving the
  Bodnar app's display reading a stale buffer.  Disconnect the
  antenna only if a multimeter check confirms a real short.

---

## How to add support for a new feature you've reverse-engineered

The general procedure:

1. Set the unit to a known baseline state via the official app.
2. Run `python3 bodnar_cli.py --raw-status` and save the output.
3. Toggle ONE setting in the official app.  Quit the app cleanly so
   it releases the HID device.
4. Run `python3 bodnar_cli.py --raw-status` again and diff against
   step 2.
5. Identify the byte (and bit, if it's a bitmask) that changed.
6. Add the mapping to `read_status()` in `bodnar_gui.py` and the
   display path in `bodnar_cli.py`.
7. Append the new before/after dump to the [Verification log](#verification-log)
   above so future maintainers can see the evidence.

For new write commands (e.g., constellation enable, FLL toggle):

1. Capture USB traffic while the official app sets the feature you
   want to control.  On macOS this is awkward (the IORegistry
   interface doesn't easily capture HID feature reports); the cleanest
   path is to run the Windows Bodnar app inside a VM with
   [USBPcap](https://desowin.org/usbpcap/) configured to capture the
   right device.
2. From the captured packet, identify the opcode byte and payload
   format.
3. Add the opcode constant + a `set_<feature>(...)` method to
   `HIDBackend` in `bodnar_gui.py`.  **Default to the temporary
   opcode** if the device exposes both temp and persistent variants.
4. Verify with a temp write followed by a `read_status()` read-back
   on a unit you can power-cycle if things go wrong.

## References

- Cervera, M. A., et al., *PHaRLAP* (Defence Sci. & Tech. Group, AU) —
  used elsewhere in this project, not directly relevant here.
- bvernoux, *lbe-142x*, https://github.com/bvernoux/lbe-142x
- jjcarrier, *gpsdo*, https://github.com/jjcarrier/gpsdo
- Silicon Labs Si5351A datasheet (the actual frequency synthesizer
  inside Bodnar GPSDOs) — useful for understanding what the
  fractional-divider math looks like on the chip side.
- u-blox NEO-M8N datasheet (the GNSS receiver inside the LBE-1425) —
  for understanding which constellations the hardware can support.
