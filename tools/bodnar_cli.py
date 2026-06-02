#!/usr/bin/env python3
"""bodnar_cli.py

Command-line utility to control the Leo Bodnar LBE-1425 (and family)
GPSDO via USB HID.  No GUI, just a thin wrapper around the
HIDBackend class in bodnar_gui.py.

Frequency precision
-------------------
The CLI accepts floats (`--out1 5000123.456`) but writes are
INTEGER-Hz only on this LBE-1425 firmware revision.  Fractional Hz
in the input is silently truncated by `int(hz)`.  Practical
precision is therefore 1 Hz, which is far below the GPS-disciplined
oscillator's actual short-term stability and is plenty for HF
phase-coherence work.

(Earlier reverse-engineering references claimed a Q32.32 sub-Hz
encoding, but our LBE-1425 ignores the fractional bytes -- see
BODNAR_LBE1425_PROTOCOL.md for the verification.)

Examples
--------

Status (default action when no flags are given):

    python3 bodnar_cli.py
    python3 bodnar_cli.py --status

Set a single output:

    python3 bodnar_cli.py --out1 10MHz
    python3 bodnar_cli.py --out2 24.000kHz
    python3 bodnar_cli.py --out2 24123.456                # bare = Hz

Set both outputs in one shot (writes are persisted to flash unless
you pass --temp):

    python3 bodnar_cli.py --out1 10MHz --out2 24kHz
    python3 bodnar_cli.py --out1 5MHz  --out2 1MHz --temp

Display (frequency presets accepted):

    --out1 wwv5     # 5 MHz
    --out1 wwv10    # 10 MHz
    --out1 wwv15    # 15 MHz
    --out1 wwv20    # 20 MHz
    --out1 wwv25    # 25 MHz
    --out1 chu7.85  # 7.85 MHz
    --out1 ism      # 13.56 MHz

JSON output mode (machine-readable, useful for shell scripts and
automation):

    python3 bodnar_cli.py --status --json

Exit codes:
    0 : success
    1 : device not found / connection error
    2 : invalid arguments (e.g. unparseable frequency)
    3 : write or read failed at the HID layer
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Allow running this file directly without installing -- find the
# bodnar_gui.py sibling.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bodnar_gui import HIDBackend     # noqa: E402


PRESETS = {
    'wwv2.5': 2_500_000.0,
    'wwv5':   5_000_000.0,
    'wwv10': 10_000_000.0,
    'wwv15': 15_000_000.0,
    'wwv20': 20_000_000.0,
    'wwv25': 25_000_000.0,
    'chu3.33':  3_330_000.0,
    'chu7.85':  7_850_000.0,
    'chu14.67': 14_670_000.0,
    'ism':     13_560_000.0,
    '10mhz_ref': 10_000_000.0,
}

_FREQ_RE = re.compile(r'^\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(\w*)\s*$')


def parse_freq(s: str) -> float:
    """Parse '10MHz', '24.5kHz', '24123.456', '24123.456 hz' into Hz.

    Also recognizes preset names (wwv5, ism, chu7.85, ...).  Case-insensitive.
    """
    key = s.strip().lower().replace(' ', '')
    if key in PRESETS:
        return PRESETS[key]
    m = _FREQ_RE.match(s)
    if not m:
        raise ValueError(f"can't parse frequency: {s!r}")
    val = float(m.group(1))
    unit = m.group(2).lower()
    factor = {
        '': 1.0, 'hz': 1.0,
        'k': 1e3, 'khz': 1e3,
        'm': 1e6, 'mhz': 1e6,
        'g': 1e9, 'ghz': 1e9,
    }
    if unit not in factor:
        raise ValueError(f"unknown frequency unit: {unit!r}")
    return val * factor[unit]


def fmt_hz(hz: float) -> str:
    if hz >= 1e6 - 1: return f'{hz / 1e6:,.6f} MHz'
    if hz >= 1e3 - 1: return f'{hz / 1e3:,.6f} kHz'
    return f'{hz:,.6f} Hz'


def print_status(bb: HIDBackend, as_json: bool = False) -> int:
    try:
        s = bb.read_status()
    except Exception as e:
        if as_json:
            print(json.dumps({'error': str(e)}))
        else:
            print(f"ERROR reading status: {e}", file=sys.stderr)
        return 3

    if as_json:
        print(json.dumps(s, indent=2))
        return 0

    locks = []
    if s['gps_locked']: locks.append('GPS')
    if s['pll_locked']: locks.append('PLL')
    if s['ant_ok']:     locks.append('ANT')
    if s['pps_enabled']: locks.append('PPS')
    if not locks: locks = ['(none)']

    print(f"Bodnar {s['model']}  status:")
    print(f"  locks:        {', '.join(locks)}")
    print(f"  raw status:   0x{s['raw_status_byte']:02x}")
    print(f"  output 1:     {fmt_hz(s['out1_freq_hz'])}  "
          f"[{'enabled' if s['out1_enabled'] else 'DISABLED'}, "
          f"{'low-power' if s['out1_low_power'] else 'normal'}]")
    print(f"  output 2:     {fmt_hz(s['out2_freq_hz'])}  "
          f"[{'enabled' if s['out2_enabled'] else 'DISABLED'}, "
          f"{'low-power' if s['out2_low_power'] else 'normal'}]")
    print(f"  FLL mode:     {'ON  (frequency-lock; bad for HF phase work)' if s['fll_enabled'] else 'off (PLL mode -- recommended)'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--out1', metavar='FREQ',
                   help='set output 1 frequency (e.g. 10MHz, 24.5kHz, '
                        'wwv5)')
    p.add_argument('--out2', metavar='FREQ',
                   help='set output 2 frequency')
    p.add_argument('--temp', action='store_true',
                   help='use temporary opcodes (do NOT persist to flash). '
                        'Default is persistent.')
    p.add_argument('--status', action='store_true',
                   help='print current status (default if no other action)')
    p.add_argument('--json', action='store_true',
                   help='output status as JSON (for scripting)')
    p.add_argument('--quiet', '-q', action='store_true',
                   help='suppress confirmation messages on writes')
    p.add_argument('--raw-status', action='store_true',
                   help='dump the full 64-byte HID status response in '
                        'hex for protocol debugging')
    args = p.parse_args()

    # Default to status if no other action requested
    do_status = args.status or (args.out1 is None and args.out2 is None)
    do_writes = (args.out1 is not None) or (args.out2 is not None)

    # Validate frequency parses BEFORE opening the device
    parsed = {}
    if args.out1 is not None:
        try:
            parsed[0] = parse_freq(args.out1)
        except ValueError as e:
            print(f"ERROR --out1: {e}", file=sys.stderr); return 2
    if args.out2 is not None:
        try:
            parsed[1] = parse_freq(args.out2)
        except ValueError as e:
            print(f"ERROR --out2: {e}", file=sys.stderr); return 2

    bb = HIDBackend(persist=(not args.temp))
    try:
        bb.connect()
    except Exception as e:
        print(f"ERROR connecting to Bodnar: {e}", file=sys.stderr)
        return 1

    rc = 0
    try:
        # Writes first, then status read shows the result
        for idx, hz in parsed.items():
            label = f"output {idx + 1}"
            try:
                bb.set_frequency_hz(idx, hz)
                if not args.quiet:
                    persistence = "TEMP" if args.temp else "PERSIST"
                    print(f"  set {label} -> {fmt_hz(hz)}  [{persistence}]")
            except Exception as e:
                print(f"ERROR setting {label}: {e}", file=sys.stderr)
                rc = 3
        if args.raw_status and rc == 0:
            try:
                raw = bb.read_status_raw()
                print(f"raw status response ({len(raw)} bytes):")
                # 16 bytes per line, ASCII gutter
                for i in range(0, len(raw), 16):
                    chunk = raw[i:i+16]
                    hexs = ' '.join(f'{b:02x}' for b in chunk)
                    asc = ''.join(chr(b) if 32 <= b < 127 else '.'
                                   for b in chunk)
                    print(f'  {i:3d}: {hexs:<48} {asc}')
            except Exception as e:
                print(f"ERROR reading raw status: {e}", file=sys.stderr)
                rc = 3
        elif do_status and rc == 0:
            if do_writes and not args.quiet:
                print()                # blank line before status
            rc = print_status(bb, as_json=args.json)
    finally:
        bb.disconnect()
    return rc


if __name__ == '__main__':
    sys.exit(main())
