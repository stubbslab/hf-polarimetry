#!/usr/bin/env python3
"""bodnar_gui.py — macOS/Linux Tkinter GUI for Leo Bodnar GPSDO units.

Tested with the **LBE-1425 GPSDO Locked Clock Source** (single output,
USB VID 0x1DD2 / PID 0x2269), which presents a USB CDC interface that
streams standard NMEA-0183 sentences continuously.  The GUI reads those
sentences live to drive a status panel: GPS fix quality, satellite count
(per constellation), HDOP, position, and a recent activity log.

Frequency control on the LBE-1425 is via USB HID on a separate composite
interface and uses Bodnar's documented register-write protocol (you'll
need the manual to wire it up).  The frequency-set buttons in the GUI
print a warning until that protocol is implemented.

Backends
--------
* ``Simulator``    -- in-process, no hardware.
* ``Serial``       -- pyserial CDC reader; works for the LBE-1425 today.
                      Requires:  pip install pyserial
* ``HID``          -- placeholder for HID-protocol frequency control;
                      not yet implemented.

To run
------
    python3 bodnar_gui.py
"""

from __future__ import annotations

import glob
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional, Tuple


# Per Apple/Linux usbmodem naming: the LBE-1425 shows up at
# /dev/cu.usbmodem<SERIAL> (macOS) or /dev/ttyACM* (Linux).
DEFAULT_SERIAL_PATTERNS = [
    "/dev/cu.usbmodem*",     # macOS
    "/dev/tty.usbmodem*",    # macOS (alternate)
    "/dev/ttyACM*",          # Linux CDC
]


# --------------------------------------------------------------- NMEA parser

@dataclass
class FixState:
    """Most recent fix as observed from NMEA traffic."""
    fix_quality: int = 0          # GGA field 6: 0=no fix, 1=GPS, 2=DGPS, ...
    n_sats_used: int = 0          # GGA field 7
    hdop: float = float("nan")    # GGA field 9
    pdop: float = float("nan")    # GSA field 15 (when present)
    vdop: float = float("nan")    # GSA field 17 (when present)
    lat_deg: float = float("nan")
    lon_deg: float = float("nan")
    alt_m: float = float("nan")
    utc_time: str = ""            # HHMMSS.ss from GGA
    utc_date: str = ""            # DDMMYY from RMC
    sats_in_view: Dict[str, int] = field(default_factory=dict)
    sat_snrs: Dict[str, List[int]] = field(default_factory=dict)
    last_sentence_time: float = 0.0
    n_sentences: int = 0
    # Quality tracking
    fix_locked_since: float = 0.0   # wallclock when fix_quality first > 0
    n_fix_drops: int = 0            # transitions from locked to unlocked


def _nmea_checksum_ok(line: str) -> bool:
    """Validate the *...*HH checksum of an NMEA sentence."""
    if "*" not in line:
        return False
    body, _, csum = line[1:].partition("*")
    csum = csum.strip()[:2]
    try:
        target = int(csum, 16)
    except ValueError:
        return False
    actual = 0
    for c in body:
        actual ^= ord(c)
    return actual == target


def _parse_lat_lon(value: str, hemisphere: str) -> float:
    """Parse a NMEA lat/lon (DDMM.MMMM / DDDMM.MMMM) into decimal deg."""
    if not value or not hemisphere:
        return float("nan")
    try:
        # latitude has 2 deg digits, longitude has 3
        deg_digits = 3 if hemisphere in ("E", "W") else 2
        deg = int(value[:deg_digits])
        minutes = float(value[deg_digits:])
        out = deg + minutes / 60.0
        if hemisphere in ("S", "W"):
            out = -out
        return out
    except (ValueError, IndexError):
        return float("nan")


def update_fix_from_sentence(state: FixState, line: str) -> None:
    """Apply one NMEA sentence to the fix state.  Tolerates malformed
    or unknown sentences silently (this is a status display, not a
    nav-grade parser)."""
    if not line.startswith("$") or "*" not in line:
        return
    if not _nmea_checksum_ok(line):
        # Most short-read fragments fail here; ignore quietly
        return
    state.last_sentence_time = time.time()
    state.n_sentences += 1
    body = line[1:].split("*", 1)[0]
    fields = body.split(",")
    if len(fields) < 2:
        return
    talker = fields[0][:2]   # GP=GPS, GL=GLONASS, GA=Galileo, GN=combined
    sentence = fields[0][2:]

    if sentence == "GGA" and len(fields) >= 11:
        # $xxGGA,utc,lat,latH,lon,lonH,fix,nsats,hdop,alt,M,...
        try:
            state.utc_time = fields[1]
            state.lat_deg = _parse_lat_lon(fields[2], fields[3])
            state.lon_deg = _parse_lat_lon(fields[4], fields[5])
            new_fq = int(fields[6]) if fields[6] else 0
            # Track lock acquisition / drop transitions
            if new_fq > 0 and state.fix_quality == 0:
                state.fix_locked_since = time.time()
            elif new_fq == 0 and state.fix_quality > 0:
                state.n_fix_drops += 1
            state.fix_quality = new_fq
            state.n_sats_used = int(fields[7]) if fields[7] else 0
            state.hdop = float(fields[8]) if fields[8] else float("nan")
            state.alt_m = float(fields[9]) if fields[9] else float("nan")
        except ValueError:
            pass

    elif sentence == "RMC" and len(fields) >= 10:
        # $xxRMC,utc,status,lat,latH,lon,lonH,sog,cog,date,...
        # date is DDMMYY (field index 9)
        try:
            state.utc_date = fields[9] if fields[9] else ""
        except ValueError:
            pass

    elif sentence == "GSA" and len(fields) >= 17:
        # $xxGSA,mode,fix3d,sat1..12, pdop, hdop, vdop *cs
        try:
            state.pdop = float(fields[15]) if fields[15] else float("nan")
            state.vdop = float(fields[17].split("*")[0]) if fields[17] else float("nan")
        except (ValueError, IndexError):
            pass

    elif sentence == "GSV" and len(fields) >= 4:
        # $xxGSV,total_msgs,msg_num,total_sats_in_view,(sat,el,az,snr)*4...
        try:
            n_in_view = int(fields[3]) if fields[3] else 0
            state.sats_in_view[talker] = n_in_view
            # Extract per-sat SNR.  Each block is 4 fields starting at
            # index 4.  The last block may be partial.
            snrs = state.sat_snrs.setdefault(talker, [])
            if int(fields[2]) == 1:
                snrs.clear()       # first message of new burst
            i = 4
            while i + 3 < len(fields):
                snr_str = fields[i + 3]
                if snr_str:
                    try:
                        snrs.append(int(snr_str))
                    except ValueError:
                        pass
                i += 4
        except ValueError:
            pass


# ----------------------------------------------------------------- backends

class GPSDOBackend:
    """Abstract base class.  All backends provide:

      * connect() / disconnect() / is_connected()
      * get_fix_state()  -> FixState
      * set_frequency_hz(output_index, hz)  -> raises NotImplementedError
        if the backend has no command channel for the connected unit.
    """
    name: str = "abstract"
    n_outputs: int = 1
    supports_frequency_control: bool = False

    def connect(self) -> None: raise NotImplementedError
    def disconnect(self) -> None: raise NotImplementedError
    def is_connected(self) -> bool: raise NotImplementedError
    def get_fix_state(self) -> FixState:
        raise NotImplementedError
    def set_frequency_hz(self, output_index: int, hz: float) -> None:
        raise NotImplementedError(
            f"{self.name} backend has no frequency-control channel "
            f"for this unit; consult the unit's manual and extend the "
            f"backend.")


# ............................................................. Simulator

class SimulatorBackend(GPSDOBackend):
    """In-memory simulator for GUI-without-hardware testing."""
    name = "Simulator"
    n_outputs = 2
    supports_frequency_control = True

    def __init__(self):
        self._connected = False
        self._connect_t = 0.0
        self._freqs = [10_000_000.0, 25_000_000.0]
        self._state = FixState()

    def connect(self):
        self._connected = True
        self._connect_t = time.time()

    def disconnect(self):
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_fix_state(self) -> FixState:
        s = self._state
        s.last_sentence_time = time.time()
        s.n_sentences += 1
        if not self._connected:
            s.fix_quality = 0
            s.n_sats_used = 0
            return s
        elapsed = time.time() - self._connect_t
        if elapsed < 3.0:
            s.fix_quality = 0
            s.n_sats_used = int(min(elapsed * 4, 10))
            s.hdop = float("nan")
        else:
            s.fix_quality = 1
            s.n_sats_used = 9 + int(time.time() % 5)
            s.hdop = 0.7 + 0.1 * (time.time() % 3)
            s.lat_deg = 42.3744; s.lon_deg = -71.1169; s.alt_m = 12.0
            s.sats_in_view = {"GP": 11, "GL": 9}
            s.sat_snrs = {"GP": [22, 33, 18, 25, 30, 38],
                          "GL": [27, 23, 17, 30]}
        s.utc_time = time.strftime("%H%M%S", time.gmtime())
        return s

    def set_frequency_hz(self, output_index: int, hz: float) -> None:
        if not self._connected:
            raise RuntimeError("not connected")
        self._freqs[output_index] = float(hz)


# .................................................................. Serial

class SerialBackend(GPSDOBackend):
    """USB-CDC serial backend for the LBE-1425 (and likely siblings).

    The device streams standard NMEA-0183 GPS sentences on its CDC
    interface at 115200 baud.  We open the port, run a background
    reader thread, parse incoming sentences, and update a thread-safe
    FixState that the GUI polls.
    """
    name = "Serial"
    n_outputs = 1                  # LBE-1425 has one RF output
    supports_frequency_control = False

    def __init__(self, port: Optional[str] = None, baud: int = 115200):
        self._explicit_port = port
        self._baud = baud
        self._serial = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._state = FixState()
        self._state_lock = threading.Lock()

    @staticmethod
    def auto_find_ports() -> List[str]:
        out = []
        for pat in DEFAULT_SERIAL_PATTERNS:
            out.extend(sorted(glob.glob(pat)))
        # Deduplicate while preserving order
        seen = set()
        return [p for p in out if not (p in seen or seen.add(p))]

    @staticmethod
    def find_bodnar_port() -> Optional[str]:
        """Best-effort identification of the Bodnar device on macOS:
        the LBE-1425 has a serial number starting with characters from
        the device descriptor visible in the device path."""
        candidates = SerialBackend.auto_find_ports()
        return candidates[0] if candidates else None

    def connect(self) -> None:
        try:
            import serial
        except ImportError as e:
            raise RuntimeError(
                "pyserial not installed.  Run: pip install pyserial") from e
        port = self._explicit_port or self.find_bodnar_port()
        if port is None:
            raise RuntimeError(
                "no USB serial device found.  Is the Bodnar plugged in?\n"
                "Patterns tried: " + ", ".join(DEFAULT_SERIAL_PATTERNS))
        try:
            self._serial = serial.Serial(port, self._baud, timeout=0.5)
        except Exception as e:
            raise RuntimeError(f"could not open {port}: {e}") from e
        self._stop_evt.clear()
        with self._state_lock:
            self._state = FixState()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True,
            name="bodnar-nmea-reader")
        self._reader_thread.start()

    def disconnect(self) -> None:
        self._stop_evt.set()
        thr = self._reader_thread
        self._reader_thread = None
        if thr is not None:
            thr.join(timeout=1.0)
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
        self._serial = None

    def is_connected(self) -> bool:
        return self._serial is not None

    def get_fix_state(self) -> FixState:
        with self._state_lock:
            return _copy_fix_state(self._state)

    def _reader_loop(self) -> None:
        buf = b""
        while not self._stop_evt.is_set():
            try:
                chunk = self._serial.read(256)
            except Exception:
                break
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line_str = line.decode("ascii", errors="replace").strip()
                if not line_str:
                    continue
                with self._state_lock:
                    update_fix_from_sentence(self._state, line_str)


def _copy_fix_state(s: FixState) -> FixState:
    return FixState(
        fix_quality=s.fix_quality,
        n_sats_used=s.n_sats_used,
        hdop=s.hdop, pdop=s.pdop, vdop=s.vdop,
        lat_deg=s.lat_deg, lon_deg=s.lon_deg, alt_m=s.alt_m,
        utc_time=s.utc_time, utc_date=s.utc_date,
        sats_in_view=dict(s.sats_in_view),
        sat_snrs={k: list(v) for k, v in s.sat_snrs.items()},
        last_sentence_time=s.last_sentence_time,
        n_sentences=s.n_sentences,
        fix_locked_since=s.fix_locked_since,
        n_fix_drops=s.n_fix_drops,
    )


def format_gps_time(utc_time: str, utc_date: str) -> str:
    """Format an NMEA HHMMSS.ss + DDMMYY pair into ISO-ish UTC."""
    if not utc_time:
        return "—"
    try:
        hh, mm = utc_time[0:2], utc_time[2:4]
        ss = utc_time[4:].rstrip(".0") or "0"
        # Reconstruct the seconds field nicely
        ss_full = utc_time[4:] if len(utc_time) > 4 else "00"
        time_str = f"{hh}:{mm}:{ss_full}"
    except Exception:
        time_str = utc_time
    if utc_date and len(utc_date) >= 6:
        try:
            dd, mo, yr = utc_date[0:2], utc_date[2:4], utc_date[4:6]
            return f"20{yr}-{mo}-{dd}  {time_str} UTC"
        except Exception:
            return f"{time_str} UTC"
    return f"{time_str} UTC"


def assess_quality(s: FixState) -> Tuple[str, str, str]:
    """Compress the GPS fix metrics into a single overall quality
    judgement.  Returns (label, colour, detail)."""
    if s.fix_quality == 0:
        return ("NO LOCK", "#a83232",
                f"{s.n_sats_used} sats, no fix")
    pieces = []
    score = 0
    # HDOP: <1 excellent, <2 good, <5 acceptable, >5 marginal
    if s.hdop == s.hdop:
        if s.hdop < 1.0:
            score += 3; pieces.append(f"HDOP {s.hdop:.2f} (excellent)")
        elif s.hdop < 2.0:
            score += 2; pieces.append(f"HDOP {s.hdop:.2f} (good)")
        elif s.hdop < 5.0:
            score += 1; pieces.append(f"HDOP {s.hdop:.2f} (ok)")
        else:
            pieces.append(f"HDOP {s.hdop:.2f} (poor)")
    # Sat count
    if s.n_sats_used >= 8:
        score += 3
    elif s.n_sats_used >= 5:
        score += 2
    elif s.n_sats_used >= 4:
        score += 1
    pieces.append(f"{s.n_sats_used} sats")
    # SNR: best across constellations
    all_snrs = []
    for arr in s.sat_snrs.values():
        all_snrs.extend(arr)
    if all_snrs:
        best_snr = max(all_snrs)
        med_snr = sorted(all_snrs)[len(all_snrs) // 2]
        pieces.append(f"med SNR {med_snr} dB")
        if med_snr >= 30: score += 2
        elif med_snr >= 20: score += 1
    # Drops
    if s.n_fix_drops > 0:
        pieces.append(f"{s.n_fix_drops} dropouts")
        score -= 1
    if score >= 7:
        label, colour = "EXCELLENT", "#0a8a0a"
    elif score >= 5:
        label, colour = "GOOD", "#1d8c1d"
    elif score >= 3:
        label, colour = "FAIR", "#aa6e1a"
    else:
        label, colour = "MARGINAL", "#a83232"
    return label, colour, " · ".join(pieces)


# .................................................................... HID

class HIDBackend(GPSDOBackend):
    """Placeholder.  Frequency control on the Bodnar units uses a HID
    feature-report protocol documented in the unit's manual.  Not yet
    implemented; falls back to serial NMEA for status reporting."""
    name = "HID (not implemented)"
    n_outputs = 1
    supports_frequency_control = False

    def connect(self):
        raise RuntimeError(
            "HID frequency-control protocol not yet implemented.\n"
            "Use 'Serial' to read GPS status from the LBE-1425's CDC "
            "interface.  For frequency control, refer to the Leo "
            "Bodnar manual and extend HIDBackend.set_frequency_hz.")
    def disconnect(self): pass
    def is_connected(self): return False
    def get_fix_state(self): return FixState()


BACKENDS = {
    "Simulator": SimulatorBackend,
    "Serial":    SerialBackend,
    "HID":       HIDBackend,
}


# ----------------------------------------------------------------- helpers

def parse_freq(s: str, unit: str) -> float:
    s = s.strip().replace(",", "")
    if not s:
        raise ValueError("empty")
    val = float(s)
    factor = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6}[unit]
    return val * factor


def format_hz(hz: float) -> str:
    if hz >= 1e6 - 1:  return f"{hz/1e6:,.6f} MHz"
    if hz >= 1e3 - 1:  return f"{hz/1e3:,.6f} kHz"
    return f"{hz:,.6f} Hz"


WWV_PRESETS = [2_500_000, 5_000_000, 10_000_000, 15_000_000,
               20_000_000, 25_000_000]
OTHER_PRESETS = [
    ("CHU 3.33", 3_330_000),
    ("CHU 7.85", 7_850_000),
    ("CHU 14.67", 14_670_000),
    ("ISM 13.56", 13_560_000),
    ("10 MHz ref", 10_000_000),
]


# ------------------------------------------------------------------ GUI

class StatusFrame(ttk.LabelFrame):
    """Live GPS status panel driven by the backend's FixState."""

    def __init__(self, master):
        super().__init__(master, text="GPS / GPSDO Status", padding=8)
        self._labels: Dict[str, tk.StringVar] = {}

        # Big lock indicator
        self.lock_var = tk.StringVar(value="—")
        self.lock_lbl = ttk.Label(self, textvariable=self.lock_var,
                                  font=("Helvetica", 14, "bold"))
        self.lock_lbl.grid(row=0, column=0, columnspan=4, sticky="w",
                           pady=(0, 2))

        # GPS time line, prominent
        self.gps_time_var = tk.StringVar(value="—")
        ttk.Label(self, textvariable=self.gps_time_var,
                  font=("Menlo", 13)).grid(row=1, column=0, columnspan=4,
                                            sticky="w", pady=(0, 6))

        # Quality summary line
        self.qc_label_var = tk.StringVar(value="—")
        self.qc_label = ttk.Label(self, textvariable=self.qc_label_var,
                                   font=("Helvetica", 11, "bold"))
        self.qc_label.grid(row=2, column=0, sticky="w", padx=(0, 6))
        self.qc_detail_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.qc_detail_var,
                  font=("Menlo", 10), foreground="#444"
                  ).grid(row=2, column=1, columnspan=3, sticky="w",
                         pady=(0, 8))

        rows_left = [
            ("Sats used:",      "sats_used"),
            ("HDOP / VDOP:",    "dop"),
            ("Sentences/s:",    "rate"),
            ("Locked for:",     "locked_for"),
        ]
        rows_right = [
            ("Latitude:",       "lat"),
            ("Longitude:",      "lon"),
            ("Altitude:",       "alt"),
            ("Drop count:",     "drops"),
        ]
        for i, (label, key) in enumerate(rows_left):
            ttk.Label(self, text=label).grid(row=i + 3, column=0,
                                              sticky="e", padx=(0, 4))
            v = tk.StringVar(value="—")
            self._labels[key] = v
            ttk.Label(self, textvariable=v, font=("Menlo", 11)
                      ).grid(row=i + 3, column=1, sticky="w", padx=(0, 18))
        for i, (label, key) in enumerate(rows_right):
            ttk.Label(self, text=label).grid(row=i + 3, column=2,
                                              sticky="e", padx=(0, 4))
            v = tk.StringVar(value="—")
            self._labels[key] = v
            ttk.Label(self, textvariable=v, font=("Menlo", 11)
                      ).grid(row=i + 3, column=3, sticky="w")

        ttk.Label(self, text="Constellations:"
                  ).grid(row=7, column=0, sticky="ne", padx=(0, 4),
                         pady=(8, 0))
        self.constellation_var = tk.StringVar(value="—")
        ttk.Label(self, textvariable=self.constellation_var,
                  font=("Menlo", 10)
                  ).grid(row=7, column=1, columnspan=3, sticky="w",
                         pady=(8, 0))

        self._last_n_sentences = 0
        self._last_rate_t = time.time()

    def update_from_state(self, st: FixState, connected: bool):
        """Refresh all labels from the fix state."""
        if not connected:
            self.lock_var.set("disconnected")
            self.lock_lbl.config(foreground="#888")
            self.gps_time_var.set("—")
            self.qc_label_var.set("—"); self.qc_detail_var.set("")
            self.qc_label.config(foreground="#888")
            for v in self._labels.values():
                v.set("—")
            self.constellation_var.set("—")
            return

        # Lock indicator + colour
        labels = {0: "no fix", 1: "GPS fix", 2: "DGPS fix",
                  4: "RTK fixed", 5: "RTK float"}
        fix_label = labels.get(st.fix_quality, f"fix {st.fix_quality}")
        if st.fix_quality > 0:
            self.lock_var.set(f"  ◉ LOCKED — {fix_label}")
            self.lock_lbl.config(foreground="#1d8c1d")
        else:
            if st.n_sats_used > 0:
                self.lock_var.set(f"  ◌ acquiring ({st.n_sats_used} sats)")
                self.lock_lbl.config(foreground="#aa6e1a")
            else:
                self.lock_var.set("  ◌ searching")
                self.lock_lbl.config(foreground="#a83232")

        # GPS time
        self.gps_time_var.set(format_gps_time(st.utc_time, st.utc_date))

        # Quality summary
        qc_label, qc_colour, qc_detail = assess_quality(st)
        self.qc_label_var.set(qc_label)
        self.qc_label.config(foreground=qc_colour)
        self.qc_detail_var.set(qc_detail)

        # Numeric readouts
        self._labels["sats_used"].set(str(st.n_sats_used))
        if st.hdop == st.hdop:
            v_str = f"{st.vdop:.2f}" if st.vdop == st.vdop else "?"
            self._labels["dop"].set(f"{st.hdop:.2f} / {v_str}")
        else:
            self._labels["dop"].set("—")
        self._labels["lat"].set(
            f"{st.lat_deg:+.5f}°" if st.lat_deg == st.lat_deg else "—")
        self._labels["lon"].set(
            f"{st.lon_deg:+.5f}°" if st.lon_deg == st.lon_deg else "—")
        self._labels["alt"].set(
            f"{st.alt_m:.1f} m" if st.alt_m == st.alt_m else "—")
        self._labels["drops"].set(str(st.n_fix_drops))

        if st.fix_quality > 0 and st.fix_locked_since > 0:
            locked_for = time.time() - st.fix_locked_since
            self._labels["locked_for"].set(_fmt_duration(locked_for))
        else:
            self._labels["locked_for"].set("—")

        now = time.time()
        if now - self._last_rate_t > 1.0:
            d_n = st.n_sentences - self._last_n_sentences
            dt = now - self._last_rate_t
            rate = d_n / dt if dt > 0 else 0.0
            self._labels["rate"].set(f"{rate:.0f} /s")
            self._last_n_sentences = st.n_sentences
            self._last_rate_t = now

        labels_c = {"GP": "GPS", "GL": "GLONASS", "GA": "Galileo",
                    "GB": "BeiDou", "GN": "Combined"}
        bits = []
        for talker in sorted(st.sats_in_view.keys()):
            n_iv = st.sats_in_view[talker]
            snrs = st.sat_snrs.get(talker, [])
            if snrs:
                best, med = max(snrs), sorted(snrs)[len(snrs)//2]
                snr_str = f"SNR best={best} med={med} dB"
            else:
                snr_str = ""
            bits.append(f"  {labels_c.get(talker, talker):<10} "
                        f"{n_iv:>2} in view   {snr_str}")
        self.constellation_var.set("\n".join(bits) if bits else "(none yet)")


def _fmt_duration(s: float) -> str:
    """Format seconds as H:MM:SS (or M:SS if <1 hour)."""
    if s < 0: return "—"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


class OutputPanel(ttk.LabelFrame):
    """One frequency-control panel.  set/preset buttons surface a
    NotImplementedError as a log message until the HID command channel
    is wired up for the user's specific unit."""

    def __init__(self, master, output_index: int, gui: "BodnarGUI"):
        super().__init__(master, text=f"RF output {output_index + 1}",
                         padding=10)
        self.output_index = output_index
        self.gui = gui

        row = 0
        ttk.Label(self, text="Frequency:").grid(row=row, column=0,
                                                 sticky="w", pady=2)
        self.freq_entry = ttk.Entry(self, width=18)
        self.freq_entry.insert(0, "10")
        self.freq_entry.grid(row=row, column=1, sticky="we", padx=4)
        self.freq_entry.bind("<Return>", lambda *_: self.apply_frequency())
        self.unit_var = tk.StringVar(value="MHz")
        ttk.Combobox(self, textvariable=self.unit_var, width=5,
                     state="readonly", values=["Hz", "kHz", "MHz"]
                     ).grid(row=row, column=2, sticky="w", padx=4)
        ttk.Button(self, text="Set", command=self.apply_frequency
                   ).grid(row=row, column=3, padx=4)

        row += 1
        ttk.Label(self, text="WWV presets:").grid(row=row, column=0,
                                                    sticky="w",
                                                    pady=(8, 2))
        wwv = ttk.Frame(self); wwv.grid(row=row, column=1, columnspan=3,
                                         sticky="we")
        for hz in WWV_PRESETS:
            ttk.Button(wwv,
                       text=f"{hz//1_000_000} M" if hz % 1_000_000 == 0
                       else f"{hz/1e6:g} M",
                       width=6,
                       command=lambda h=hz: self.set_preset(h)
                       ).pack(side="left", padx=1)

        row += 1
        ttk.Label(self, text="Other:").grid(row=row, column=0, sticky="w")
        oth = ttk.Frame(self); oth.grid(row=row, column=1, columnspan=3,
                                         sticky="we")
        for label, hz in OTHER_PRESETS:
            ttk.Button(oth, text=label, width=10,
                       command=lambda h=hz: self.set_preset(h)
                       ).pack(side="left", padx=1)

        row += 1
        ttk.Label(self, text="Fine ±:").grid(row=row, column=0,
                                              sticky="w",
                                              pady=(8, 2))
        adj = ttk.Frame(self); adj.grid(row=row, column=1, columnspan=3,
                                         sticky="we")
        for label, d in [("−1 Hz", -1), ("−0.1", -0.1), ("−0.01", -0.01),
                         ("+0.01", +0.01), ("+0.1", +0.1), ("+1 Hz", +1)]:
            ttk.Button(adj, text=label, width=6,
                       command=lambda dd=d: self.nudge(dd)
                       ).pack(side="left", padx=1)

        self.columnconfigure(1, weight=1)

    def set_preset(self, hz: float):
        self.freq_entry.delete(0, "end")
        self.freq_entry.insert(0, f"{hz/1e6:g}")
        self.unit_var.set("MHz")
        self.apply_frequency()

    def nudge(self, delta_hz: float):
        try:
            current = parse_freq(self.freq_entry.get(), self.unit_var.get())
        except ValueError:
            return
        new_hz = current + delta_hz
        unit = self.unit_var.get()
        scale = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6}[unit]
        self.freq_entry.delete(0, "end")
        self.freq_entry.insert(0, f"{new_hz/scale:.9g}")
        self.apply_frequency()

    def apply_frequency(self):
        try:
            hz = parse_freq(self.freq_entry.get(), self.unit_var.get())
        except ValueError as e:
            self.gui.log(f"output {self.output_index+1}: invalid: {e}",
                         level="error")
            return
        try:
            self.gui.backend.set_frequency_hz(self.output_index, hz)
            self.gui.log(
                f"output {self.output_index+1}: set to {format_hz(hz)}")
        except NotImplementedError as e:
            self.gui.log(f"frequency control unavailable: {e}",
                         level="warn")
        except Exception as e:
            self.gui.log(f"output {self.output_index+1}: set failed: {e}",
                         level="error")


class BodnarGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Bodnar GPSDO Control & Status")
        self.geometry("860x720")
        self.backend: GPSDOBackend = SimulatorBackend()
        self._poll_after = None
        try:
            self.tk.call("ttk::style", "theme", "use", "aqua")
        except tk.TclError:
            pass
        self._build_ui()
        self._poll_status()

    def _build_ui(self):
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        # Connection bar
        conn = ttk.LabelFrame(outer, text="Connection", padding=8)
        conn.pack(fill="x", pady=(0, 8))
        ttk.Label(conn, text="Backend:").grid(row=0, column=0, sticky="w")
        self.backend_var = tk.StringVar(value="Serial")
        backend_combo = ttk.Combobox(
            conn, textvariable=self.backend_var, width=12, state="readonly",
            values=list(BACKENDS.keys()))
        backend_combo.grid(row=0, column=1, sticky="w", padx=4)
        backend_combo.bind("<<ComboboxSelected>>", self._on_backend_changed)
        self._on_backend_changed()

        ttk.Label(conn, text="Port:").grid(row=0, column=2, sticky="e",
                                            padx=(20, 2))
        self.port_var = tk.StringVar(value="(auto)")
        self.port_combo = ttk.Combobox(conn, textvariable=self.port_var,
                                        width=32)
        self.port_combo.grid(row=0, column=3, sticky="w")
        self._refresh_ports()
        ttk.Button(conn, text="↻", width=2,
                   command=self._refresh_ports).grid(row=0, column=4,
                                                     padx=(2, 8))

        self.connect_btn = ttk.Button(conn, text="Connect",
                                       command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=5, padx=4)
        ttk.Button(conn, text="Probe (no-op safe)",
                   command=self.probe_connection
                   ).grid(row=0, column=6, padx=4)

        # Status frame
        self.status = StatusFrame(outer)
        self.status.pack(fill="x", pady=(0, 8))

        # Output panels
        outputs_frame = ttk.Frame(outer)
        outputs_frame.pack(fill="x", pady=(0, 8))
        self.panels: List[OutputPanel] = []
        for i in range(2):    # build two; show only n the backend has
            p = OutputPanel(outputs_frame, i, self)
            p.grid(row=0, column=i, sticky="nsew",
                   padx=(0, 8) if i == 0 else 0)
            outputs_frame.columnconfigure(i, weight=1)
            self.panels.append(p)
        self._update_visible_panels()

        # Buttons
        files = ttk.Frame(outer)
        files.pack(fill="x", pady=(0, 6))
        ttk.Button(files, text="Save config…",
                   command=self.save_config).pack(side="left")
        ttk.Button(files, text="Load config…",
                   command=self.load_config).pack(side="left", padx=6)
        ttk.Button(files, text="Quit",
                   command=self.on_quit).pack(side="right")

        # Log
        log_frame = ttk.LabelFrame(outer, text="Activity log", padding=4)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, height=10,
                                font=("Menlo", 10),
                                bg="#f5f5f7", fg="#111", state="disabled")
        self.log_text.pack(fill="both", expand=True)
        self.protocol("WM_DELETE_WINDOW", self.on_quit)

    def _refresh_ports(self):
        cands = SerialBackend.auto_find_ports()
        values = ["(auto)"] + cands
        self.port_combo["values"] = values
        if not cands:
            self.log("no usbmodem/ttyACM devices found.  "
                     "Plug the Bodnar in and click ↻.")

    def _on_backend_changed(self, *_):
        if hasattr(self, "backend") and self.backend.is_connected():
            self.backend.disconnect()
        cls = BACKENDS[self.backend_var.get()]
        self.backend = cls()
        if hasattr(self, "connect_btn"):
            self.connect_btn.config(text="Connect")
            self._update_visible_panels()
            self.log(f"backend → {self.backend.name}", level="info")

    def _update_visible_panels(self):
        if not hasattr(self, "panels"):
            return
        n = self.backend.n_outputs
        for i, p in enumerate(self.panels):
            if i < n:
                p.grid()
                state = ("normal" if self.backend.supports_frequency_control
                         else "disabled")
                # Disable the entry/buttons for read-only backends
                self._set_panel_state(p, state)
            else:
                p.grid_remove()

    def _set_panel_state(self, panel: OutputPanel, state: str):
        for child in panel.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass
            for sub in getattr(child, "winfo_children", lambda: [])():
                try:
                    sub.configure(state=state)
                except tk.TclError:
                    pass

    def toggle_connection(self):
        if self.backend.is_connected():
            self.backend.disconnect()
            self.connect_btn.config(text="Connect")
            self.log("disconnected.")
            return
        # Wire up explicit port for Serial
        if isinstance(self.backend, SerialBackend):
            chosen = self.port_var.get().strip()
            if chosen and chosen != "(auto)":
                self.backend._explicit_port = chosen
            else:
                self.backend._explicit_port = None
        try:
            self.backend.connect()
            self.connect_btn.config(text="Disconnect")
            label = (f" on {self.backend._explicit_port or 'auto-detected port'}"
                      if isinstance(self.backend, SerialBackend) else "")
            self.log(f"connected via {self.backend.name}{label}")
        except Exception as e:
            self.log(f"connect failed: {e}", level="error")
            messagebox.showerror("Connect failed", str(e))

    def probe_connection(self):
        """One-shot diagnostic: open the port, listen briefly, report."""
        ports = SerialBackend.auto_find_ports()
        self.log(f"probing {len(ports)} candidate port(s)…")
        for p in ports:
            ok, info = _probe_one_port(p)
            level = "info" if ok else "warn"
            self.log(f"  {p}: {info}", level=level)

    def _poll_status(self):
        try:
            connected = self.backend.is_connected()
            st = self.backend.get_fix_state() if connected else FixState()
            self.status.update_from_state(st, connected)
        except Exception as e:
            self.log(f"status poll error: {e}", level="error")
        self._poll_after = self.after(500, self._poll_status)

    def log(self, msg: str, level: str = "info"):
        self.log_text.config(state="normal")
        ts = time.strftime("%H:%M:%S")
        prefix = {"info": "      ",
                  "warn": "WARN  ",
                  "error": "ERROR "}.get(level, "      ")
        self.log_text.insert("end", f"{ts}  {prefix}{msg}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def save_config(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON config", "*.json"), ("all files", "*")],
            initialfile="bodnar_config.json")
        if not path:
            return
        cfg = {"backend": self.backend_var.get(),
               "port": self.port_var.get(),
               "outputs": []}
        for p in self.panels:
            try:
                hz = parse_freq(p.freq_entry.get(), p.unit_var.get())
                cfg["outputs"].append({"hz": hz, "unit": p.unit_var.get()})
            except ValueError:
                cfg["outputs"].append(None)
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        self.log(f"saved config to {path}")

    def load_config(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON config", "*.json"), ("all files", "*")])
        if not path:
            return
        try:
            with open(path) as f: cfg = json.load(f)
        except Exception as e:
            messagebox.showerror("Load failed", str(e)); return
        if "backend" in cfg and cfg["backend"] in BACKENDS:
            self.backend_var.set(cfg["backend"])
            self._on_backend_changed()
        if "port" in cfg:
            self.port_var.set(cfg["port"])
        for p, item in zip(self.panels, cfg.get("outputs") or []):
            if item is None:
                continue
            unit = item.get("unit", "Hz")
            scale = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6}[unit]
            p.unit_var.set(unit)
            p.freq_entry.delete(0, "end")
            p.freq_entry.insert(0, f"{item['hz']/scale:.9g}")
        self.log(f"loaded config from {path}")

    def on_quit(self):
        try:
            if self._poll_after:
                self.after_cancel(self._poll_after)
            if self.backend.is_connected():
                self.backend.disconnect()
        finally:
            self.destroy()


def _probe_one_port(port: str, timeout_s: float = 2.0
                    ) -> Tuple[bool, str]:
    """Open the port, listen up to timeout_s, count NMEA sentences,
    and report a one-line summary.  Closes the port before returning.
    """
    try:
        import serial
    except ImportError:
        return False, "pyserial not installed"
    try:
        s = serial.Serial(port, 115200, timeout=0.5)
    except Exception as e:
        return False, f"open failed: {e}"
    try:
        deadline = time.time() + timeout_s
        buf = b""
        n_nmea = 0
        n_gga_fix = -1
        n_sats = -1
        while time.time() < deadline:
            chunk = s.read(256)
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line_str = line.decode("ascii", errors="replace").strip()
                if line_str.startswith("$") and "*" in line_str:
                    if _nmea_checksum_ok(line_str):
                        n_nmea += 1
                        # Glance at GGA for a quick fix indication
                        body = line_str[1:].split("*", 1)[0].split(",")
                        if len(body) > 7 and body[0][2:] == "GGA":
                            try:
                                n_gga_fix = int(body[6]) if body[6] else 0
                                n_sats   = int(body[7]) if body[7] else 0
                            except ValueError:
                                pass
        if n_nmea == 0:
            return False, "no NMEA sentences seen (port silent or wrong device)"
        return True, (f"{n_nmea} NMEA sentences in {timeout_s:.0f}s; "
                      f"fix={n_gga_fix}, sats={n_sats}")
    finally:
        try: s.close()
        except Exception: pass


if __name__ == "__main__":
    BodnarGUI().mainloop()
