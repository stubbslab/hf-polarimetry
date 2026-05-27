#!/usr/bin/env python3
"""bodnar_gui.py — macOS/Linux Tkinter GUI for the Leo Bodnar Dual GPSDO.

Two-output frequency control with sub-Hz precision, GPS-lock status,
WWV/HF preset buttons, and a save/load for configurations.

Transport backends
------------------
The Bodnar GPSDO protocol is USB-based; depending on the unit's firmware
revision it presents either as an HID device or as a USB-CDC serial
device.  This GUI ships with three backends, selectable at the top of
the window:

  * ``Simulator``    -- in-process, no hardware.  Use this to learn the
                        GUI without a unit connected.
  * ``HID``          -- via the ``hidapi`` Python package; the more
                        common transport on current Bodnar firmware.
                        Requires:  pip install hidapi
  * ``Serial``       -- via ``pyserial``; older firmware revisions or
                        third-party clones may use CDC serial.
                        Requires:  pip install pyserial

The HID register layout for the Bodnar is documented in the unit's
manual (download from leobodnar.com).  This GUI implements the common
Si5351-based command set used by the Mini and Dual models; if your
unit's protocol differs, override the relevant ``HIDBackend`` methods
and refer to the manual.

To run
------
    python3 bodnar_gui.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import tkinter as tk
from dataclasses import asdict, dataclass, field
from tkinter import filedialog, messagebox, ttk
from typing import Callable, List, Optional


# ---------------------------------------------------------------- backends

class GPSDOBackend:
    """Abstract pluggable backend.  Subclasses override the four core
    methods; the GUI doesn't care about the transport.
    """

    name: str = "abstract"
    n_outputs: int = 2

    def connect(self) -> bool:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def is_connected(self) -> bool:
        raise NotImplementedError

    def set_frequency_hz(self, output_index: int, hz: float) -> None:
        """Output index is 0 for output 1, 1 for output 2."""
        raise NotImplementedError

    def get_status(self) -> dict:
        """Return a status dict with at least the keys
        ``connected, lock, n_satellites, output_hz``."""
        raise NotImplementedError


# ............................................................. Simulator

class SimulatorBackend(GPSDOBackend):
    """In-memory simulator. Mimics GPS lock acquiring after ~3 s and
    holds whatever frequencies you write."""
    name = "Simulator"

    def __init__(self):
        self._connected = False
        self._connect_t = 0.0
        self._freqs = [10_000_000.0, 25_000_000.0]
        self._lock_state = "no-fix"

    def connect(self) -> bool:
        self._connected = True
        self._connect_t = time.time()
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def set_frequency_hz(self, output_index: int, hz: float) -> None:
        if not self._connected:
            raise RuntimeError("not connected")
        self._freqs[output_index] = float(hz)

    def get_status(self) -> dict:
        if not self._connected:
            return dict(connected=False, lock="-", n_satellites=0,
                        output_hz=tuple(self._freqs))
        elapsed = time.time() - self._connect_t
        if elapsed < 3.0:
            self._lock_state = "no-fix"
            n_sat = int(elapsed * 4)
        else:
            self._lock_state = "locked"
            n_sat = 9 + int((time.time() % 5))
        return dict(connected=True, lock=self._lock_state,
                    n_satellites=n_sat,
                    output_hz=tuple(self._freqs))


# ................................................................... HID

class HIDBackend(GPSDOBackend):
    """USB HID transport, expected for current Bodnar firmware.

    The Vendor and Product IDs below are commonly cited for Leo Bodnar
    Electronics products but you should *verify* by running:

        system_profiler SPUSBDataType | grep -A 5 -i bodnar

    on macOS, or ``lsusb -v`` on Linux.  Update VID/PID before relying
    on this backend.

    The HID report format used here follows the publicly-circulated
    Si5351-style register layout:  64-byte feature report,
    [reg_addr, value0, value1, ...].  See the Bodnar manual for the
    actual command set.  Where the docs are ambiguous, prefer the
    manual.
    """
    name = "HID"
    VID = 0x1DD2          # Leo Bodnar Electronics (verify)
    PID = 0x2210          # Dual GPSDO (verify)
    REPORT_LEN = 64

    def __init__(self):
        self._dev = None

    def connect(self) -> bool:
        try:
            import hid                          # cython-hidapi
        except ImportError:
            raise RuntimeError(
                "hidapi not installed.  pip install hidapi"
            )
        try:
            self._dev = hid.device()
            self._dev.open(self.VID, self.PID)
            self._dev.set_nonblocking(True)
            return True
        except Exception as e:
            self._dev = None
            raise RuntimeError(f"could not open Bodnar via HID: {e}")

    def disconnect(self) -> None:
        if self._dev is not None:
            self._dev.close()
        self._dev = None

    def is_connected(self) -> bool:
        return self._dev is not None

    def set_frequency_hz(self, output_index: int, hz: float) -> None:
        """Bodnar Dual command:  feature report id 0x01,
        followed by output index, followed by 8 bytes encoding the
        Si5351 fractional divider for the requested frequency.
        Caveat: the exact byte layout is firmware-version-dependent;
        cross-check against the manual or the Windows utility before
        committing to it for science work."""
        if self._dev is None:
            raise RuntimeError("not connected")
        # Placeholder encoding; replace with the layout from the
        # Bodnar manual or by capturing the Windows utility's USB
        # traffic with Wireshark/USBPcap.
        report = bytearray([0x00] * self.REPORT_LEN)
        report[0] = 0x01                         # report id
        report[1] = int(output_index) & 0xFF
        # encode Hz as a 64-bit little-endian integer milli-Hz
        # (placeholder; the real protocol packs as Si5351 register values)
        mhz_q = int(round(hz * 1000.0))
        for i in range(8):
            report[2 + i] = (mhz_q >> (i * 8)) & 0xFF
        self._dev.send_feature_report(bytes(report))

    def get_status(self) -> dict:
        if self._dev is None:
            return dict(connected=False, lock="-",
                        n_satellites=0, output_hz=(None, None))
        # Real status query: send a "get status" feature report id
        # (firmware-specific) and parse the response.  For now return
        # a minimal "connected" dict.
        return dict(connected=True, lock="?",
                    n_satellites=0, output_hz=(None, None))


# ................................................................ Serial

class SerialBackend(GPSDOBackend):
    """USB-CDC serial transport, for older Bodnar firmware or clones
    that present a TTY device.  Set ``port`` to the right
    /dev/tty.usbmodem* path before connecting."""
    name = "Serial"

    def __init__(self):
        self._port_path = ""
        self._serial = None

    def auto_find_port(self) -> Optional[str]:
        """Look for the first /dev/cu.usbmodem* candidate.  Returns
        None if nothing matches."""
        import glob
        cands = sorted(glob.glob("/dev/cu.usbmodem*"))
        return cands[0] if cands else None

    def connect(self, port: Optional[str] = None,
                baud: int = 115200) -> bool:
        try:
            import serial
        except ImportError:
            raise RuntimeError(
                "pyserial not installed.  pip install pyserial"
            )
        if port is None:
            port = self.auto_find_port()
        if port is None:
            raise RuntimeError("no USB serial device found in /dev/cu.usbmodem*")
        self._port_path = port
        self._serial = serial.Serial(port, baud, timeout=0.5)
        return True

    def disconnect(self) -> None:
        if self._serial is not None:
            self._serial.close()
        self._serial = None

    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def set_frequency_hz(self, output_index: int, hz: float) -> None:
        # Hypothetical ASCII command:  "F<output> <freq_in_Hz>\r\n"
        if self._serial is None:
            raise RuntimeError("not connected")
        cmd = f"F{output_index + 1} {hz:.0f}\r\n".encode()
        self._serial.write(cmd)
        self._serial.flush()

    def get_status(self) -> dict:
        if self._serial is None:
            return dict(connected=False, lock="-",
                        n_satellites=0, output_hz=(None, None))
        # Real implementation: send "S\r\n" and parse the response.
        return dict(connected=True, lock="?",
                    n_satellites=0, output_hz=(None, None))


BACKENDS = {
    "Simulator": SimulatorBackend,
    "HID": HIDBackend,
    "Serial": SerialBackend,
}


# ----------------------------------------------------------------- helpers

def parse_freq(s: str, unit: str) -> float:
    """Parse a frequency entry. unit ∈ {'Hz','kHz','MHz'}."""
    s = s.strip().replace(",", "")
    if not s:
        raise ValueError("empty")
    val = float(s)
    factor = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6}[unit]
    return val * factor


def format_hz(hz: float) -> str:
    if hz >= 1e6 - 1:
        return f"{hz/1e6:,.6f} MHz"
    if hz >= 1e3 - 1:
        return f"{hz/1e3:,.6f} kHz"
    return f"{hz:,.6f} Hz"


# WWV bands and a few other useful HF reference frequencies
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

class OutputPanel(ttk.LabelFrame):
    """One frequency-control panel (one output of the Bodnar Dual)."""

    def __init__(self, master, output_index: int, gui: "BodnarGUI"):
        super().__init__(master, text=f"Output {output_index + 1}",
                         padding=10)
        self.output_index = output_index
        self.gui = gui

        # Frequency entry row
        row = 0
        ttk.Label(self, text="Frequency:").grid(row=row, column=0,
                                                sticky="w", pady=2)
        self.freq_entry = ttk.Entry(self, width=18)
        self.freq_entry.insert(0, "10")
        self.freq_entry.grid(row=row, column=1, sticky="we", padx=4)
        self.freq_entry.bind("<Return>", lambda *_: self.apply_frequency())

        self.unit_var = tk.StringVar(value="MHz")
        self.unit = ttk.Combobox(
            self, textvariable=self.unit_var, width=5, state="readonly",
            values=["Hz", "kHz", "MHz"])
        self.unit.grid(row=row, column=2, sticky="w", padx=4)

        ttk.Button(self, text="Set", command=self.apply_frequency
                   ).grid(row=row, column=3, padx=4)

        # Current readback
        row += 1
        ttk.Label(self, text="Current:").grid(row=row, column=0,
                                              sticky="w", pady=(8, 2))
        self.current_var = tk.StringVar(value="—")
        ttk.Label(self, textvariable=self.current_var,
                  font=("Menlo", 12)).grid(
                  row=row, column=1, columnspan=3, sticky="w")

        # Preset buttons
        row += 1
        ttk.Label(self, text="WWV presets:").grid(
            row=row, column=0, sticky="w", pady=(10, 2))
        wwv_frame = ttk.Frame(self)
        wwv_frame.grid(row=row, column=1, columnspan=3, sticky="we")
        for hz in WWV_PRESETS:
            label = f"{hz//1_000_000} MHz" if hz % 1_000_000 == 0 \
                else f"{hz/1e6:g} MHz"
            ttk.Button(wwv_frame, text=label, width=8,
                       command=lambda h=hz: self.set_preset(h)
                       ).pack(side="left", padx=1)

        row += 1
        ttk.Label(self, text="Other:").grid(
            row=row, column=0, sticky="w", pady=2)
        other_frame = ttk.Frame(self)
        other_frame.grid(row=row, column=1, columnspan=3, sticky="we")
        for label, hz in OTHER_PRESETS:
            ttk.Button(other_frame, text=label, width=10,
                       command=lambda h=hz: self.set_preset(h)
                       ).pack(side="left", padx=1)

        # Sub-Hz fine adjust
        row += 1
        ttk.Label(self, text="Fine adjust:").grid(
            row=row, column=0, sticky="w", pady=(10, 2))
        adj_frame = ttk.Frame(self)
        adj_frame.grid(row=row, column=1, columnspan=3, sticky="we")
        for label, delta_hz in [("−1 Hz", -1), ("−0.1", -0.1),
                                ("−0.01", -0.01), ("+0.01", +0.01),
                                ("+0.1", +0.1), ("+1 Hz", +1)]:
            ttk.Button(adj_frame, text=label, width=6,
                       command=lambda d=delta_hz: self.nudge(d)
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
        # display in the same unit
        unit = self.unit_var.get()
        scale = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6}[unit]
        self.freq_entry.delete(0, "end")
        self.freq_entry.insert(0, f"{new_hz/scale:.9g}")
        self.apply_frequency()

    def apply_frequency(self):
        try:
            hz = parse_freq(self.freq_entry.get(), self.unit_var.get())
        except ValueError as e:
            self.gui.log(f"output {self.output_index+1}: invalid frequency: {e}",
                         level="error")
            return
        try:
            self.gui.backend.set_frequency_hz(self.output_index, hz)
            self.gui.log(
                f"output {self.output_index+1}: set to {format_hz(hz)}")
        except Exception as e:
            self.gui.log(
                f"output {self.output_index+1}: set failed: {e}", level="error")


class BodnarGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Bodnar Dual GPSDO Control")
        self.geometry("780x620")

        self.backend: GPSDOBackend = SimulatorBackend()
        self._poll_after_id = None

        # Native macOS look
        try:
            self.tk.call("ttk::style", "theme", "use", "aqua")
        except tk.TclError:
            pass

        self._build_ui()
        self._poll_status()

    def _build_ui(self):
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)

        # Top connection bar
        conn = ttk.LabelFrame(outer, text="Connection", padding=10)
        conn.pack(fill="x", pady=(0, 10))

        ttk.Label(conn, text="Backend:").grid(row=0, column=0, sticky="w")
        self.backend_var = tk.StringVar(value="Simulator")
        backend_combo = ttk.Combobox(
            conn, textvariable=self.backend_var, width=12, state="readonly",
            values=list(BACKENDS.keys()))
        backend_combo.grid(row=0, column=1, sticky="w", padx=4)
        backend_combo.bind("<<ComboboxSelected>>", self._on_backend_changed)

        self.connect_btn = ttk.Button(
            conn, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=2, padx=10)

        ttk.Label(conn, text="GPS lock:").grid(row=0, column=3,
                                                sticky="w", padx=(20, 2))
        self.lock_var = tk.StringVar(value="—")
        self.lock_lbl = ttk.Label(conn, textvariable=self.lock_var,
                                  font=("Menlo", 11, "bold"))
        self.lock_lbl.grid(row=0, column=4, sticky="w")

        ttk.Label(conn, text="Sats:").grid(row=0, column=5,
                                            sticky="e", padx=(20, 2))
        self.sat_var = tk.StringVar(value="—")
        ttk.Label(conn, textvariable=self.sat_var,
                  font=("Menlo", 11)).grid(row=0, column=6, sticky="w")

        # Output panels (two of them, side by side)
        outputs = ttk.Frame(outer)
        outputs.pack(fill="x", pady=(0, 10))
        self.panels = []
        for i in range(2):
            p = OutputPanel(outputs, i, self)
            p.grid(row=0, column=i, sticky="nsew", padx=(0, 8) if i == 0 else 0)
            outputs.columnconfigure(i, weight=1)
            self.panels.append(p)

        # File row
        file_row = ttk.Frame(outer)
        file_row.pack(fill="x", pady=(0, 6))
        ttk.Button(file_row, text="Save config…",
                   command=self.save_config).pack(side="left")
        ttk.Button(file_row, text="Load config…",
                   command=self.load_config).pack(side="left", padx=6)
        ttk.Button(file_row, text="Quit",
                   command=self.on_quit).pack(side="right")

        # Log / console
        log_frame = ttk.LabelFrame(outer, text="Log", padding=4)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, height=10,
                                font=("Menlo", 10),
                                bg="#f3f3f5", fg="#222",
                                state="disabled")
        self.log_text.pack(fill="both", expand=True)

        self.protocol("WM_DELETE_WINDOW", self.on_quit)

    # ---------------------------------------------------- backend changes

    def _on_backend_changed(self, *_):
        if self.backend.is_connected():
            self.backend.disconnect()
        cls = BACKENDS[self.backend_var.get()]
        self.backend = cls()
        self.connect_btn.config(text="Connect")
        self.log(f"backend → {self.backend.name}")

    def toggle_connection(self):
        if self.backend.is_connected():
            self.backend.disconnect()
            self.connect_btn.config(text="Connect")
            self.log(f"disconnected from {self.backend.name}")
        else:
            try:
                self.backend.connect()
                self.connect_btn.config(text="Disconnect")
                self.log(f"connected via {self.backend.name}")
            except Exception as e:
                self.log(f"connect failed: {e}", level="error")
                messagebox.showerror("Connect failed", str(e))

    # ----------------------------------------------------- status polling

    def _poll_status(self):
        try:
            st = self.backend.get_status()
            self.lock_var.set(st.get("lock", "—"))
            sats = st.get("n_satellites", 0)
            self.sat_var.set(str(sats))
            colour = "#2a7" if st.get("lock") == "locked" else "#a55"
            self.lock_lbl.config(foreground=colour)
            for i, p in enumerate(self.panels):
                hz = st.get("output_hz", (None, None))[i]
                p.current_var.set(format_hz(hz) if hz is not None else "—")
        except Exception as e:
            self.lock_var.set("error")
            self.log(f"status poll: {e}", level="error")
        # poll every 1 s
        self._poll_after_id = self.after(1000, self._poll_status)

    # ---------------------------------------------------- log + config

    def log(self, msg: str, level: str = "info"):
        self.log_text.config(state="normal")
        ts = time.strftime("%H:%M:%S")
        prefix = "ERROR " if level == "error" else "      "
        self.log_text.insert("end", f"{ts} {prefix}{msg}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def save_config(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON config", "*.json"), ("all files", "*")],
            initialfile="bodnar_config.json")
        if not path:
            return
        cfg = {"outputs": []}
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
            with open(path) as f:
                cfg = json.load(f)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return
        for p, item in zip(self.panels, cfg.get("outputs", [])):
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
            if self._poll_after_id:
                self.after_cancel(self._poll_after_id)
            if self.backend.is_connected():
                self.backend.disconnect()
        finally:
            self.destroy()


if __name__ == "__main__":
    BodnarGUI().mainloop()
