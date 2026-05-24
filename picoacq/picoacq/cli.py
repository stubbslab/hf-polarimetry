"""picoacq.cli — command-line interface for triloop captures."""

import json
import os
import sys
from datetime import datetime, timezone

import click

from .recorder import capture as capture_func


@click.group()
@click.version_option(prog_name="picoacq")
def cli():
    """picoacq — PicoScope 5444D data acquisition for triloop."""


@cli.command(name="capture")
@click.option("-o", "--output", "output_file", type=click.Path(dir_okay=False),
              default=None,
              help="output HDF5 file path (default: capture_<timestamp>.h5 "
                   "in the current working directory)")
@click.option("--duration", type=float, default=8.0, show_default=True,
              help="capture duration (s).  At the default 15.625 MS/s, "
                   "8 s fills the 128 MS/channel onboard buffer in 4-ch mode.")
@click.option("--rate", "sample_rate", type=float, default=15_625_000.0,
              show_default=True,
              help="sample rate (Hz, requested).  Default 15.625 MS/s "
                   "places WWV 2.5/5/10/15/20 MHz in non-overlapping "
                   "Nyquist zones; see manual for alias map.")
@click.option("--channels", default="A,B,C,D", show_default=True,
              help="comma-separated channels to enable")
@click.option("--range", "range_volts", type=float, default=2.0,
              show_default=True, help="±range volts (per channel)")
@click.option("--coupling", type=click.Choice(["DC", "AC"]),
              default="DC", show_default=True)
@click.option("--auto-range/--no-auto-range", default=True,
              show_default=True,
              help="probe each channel briefly before the real capture, "
                   "then pick voltage range with `headroom` margin above peak")
@click.option("--headroom", type=float, default=3.0, show_default=True,
              help="auto-range headroom factor (range_FS = headroom * peak)")
@click.option("--probe", "probe_duration_s", type=float, default=0.1,
              show_default=True,
              help="auto-range probe duration in seconds")
@click.option("--rf-bands", "rf_bands", default="2.5e6,5e6,10e6,15e6,20e6",
              show_default=True,
              help="comma-separated expected RF frequencies (Hz), recorded "
                   "in capture_settings for downstream alias-to-RF mapping. "
                   "Set to empty string to omit.")
@click.option("--simulate/--no-simulate", default=False,
              help="force simulator (no hardware needed)")
@click.option("--sim-pol", type=click.Choice(["linear_vertical","linear_horizontal",
                                              "rcp","lcp","elliptical"]),
              default="linear_vertical", show_default=True,
              help="(simulator only) polarization to inject")
@click.option("--sim-az", type=float, default=273.0, show_default=True)
@click.option("--sim-el", type=float, default=12.0, show_default=True)
@click.option("--sim-snr", type=float, default=30.0, show_default=True)
@click.option("--sim-faraday", type=float, default=0.0, show_default=True,
              help="(simulator only) Faraday rotation rate, deg/s")
@click.option("--sim-carrier", type=float, default=25_000.0, show_default=True,
              help="(simulator only) IF carrier frequency, Hz")
def capture_cmd(output_file, duration, sample_rate, channels, range_volts,
                coupling, auto_range, headroom, probe_duration_s,
                rf_bands, simulate, sim_pol, sim_az, sim_el, sim_snr,
                sim_faraday, sim_carrier):
    """Capture a recording and write a triloop HDF5 file."""
    if output_file is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_file = f"capture_{ts}.h5"   # cwd-relative
    chans = [c.strip().upper() for c in channels.split(",")]
    rf_band_list = [float(b) for b in rf_bands.split(",") if b.strip()] \
        if rf_bands else []
    # If auto-range is on, ranges_volts=None tells the recorder to probe.
    # If auto-range is off, the user-provided --range applies to all chans.
    if auto_range:
        ranges = None
    else:
        ranges = {c: range_volts for c in chans}
    sim_kwargs = dict(
        carrier_hz=sim_carrier, az_deg=sim_az, el_deg=sim_el,
        pol=sim_pol, snr_db=sim_snr, faraday_rate_dps=sim_faraday,
    )
    capture_func(output_file,
                 duration_s=duration, sample_rate=sample_rate,
                 channels=chans, ranges_volts=ranges, coupling=coupling,
                 auto_range_enabled=auto_range,
                 headroom=headroom,
                 probe_duration_s=probe_duration_s,
                 rf_bands=rf_band_list,
                 simulate=simulate, sim_kwargs=sim_kwargs)
    # Resolve relative to cwd for clarity in the output message.
    abs_path = os.path.abspath(output_file)
    click.echo(f"wrote {abs_path}")


# ----------------------------------------------------------------- monitor
@cli.command(name="monitor")
@click.option("-d", "--out-dir", "out_dir", type=click.Path(file_okay=False),
              default=".", show_default=True,
              help="directory in which to write capture files "
                   "(default: current working directory)")
@click.option("--prefix", default="cap", show_default=True,
              help="filename prefix (the file is "
                   "<prefix>_YYYYMMDDTHHMMSSZ.h5)")
@click.option("--interval", type=float, default=600.0, show_default=True,
              help="seconds between capture starts (cycle period)")
@click.option("--duration", type=float, default=8.0, show_default=True,
              help="duration of each capture (s).  Default 8 s fits in "
                   "the 128 MS/channel buffer at 15.625 MS/s.")
@click.option("--rate", "sample_rate", type=float, default=15_625_000.0,
              show_default=True,
              help="sample rate (Hz).  Default 15.625 MS/s captures all "
                   "WWV bands via bandpass undersampling.")
@click.option("--channels", default="A,B,C,D", show_default=True)
@click.option("--range", "range_volts", type=float, default=2.0,
              show_default=True)
@click.option("--coupling", type=click.Choice(["DC", "AC"]),
              default="DC", show_default=True)
@click.option("--auto-range/--no-auto-range", default=True,
              show_default=True,
              help="auto-range each capture (probe + pick range)")
@click.option("--headroom", type=float, default=3.0, show_default=True)
@click.option("--probe", "probe_duration_s", type=float, default=0.1,
              show_default=True,
              help="auto-range probe duration (s)")
@click.option("--rf-bands", "rf_bands", default="2.5e6,5e6,10e6,15e6,20e6",
              show_default=True,
              help="comma-separated expected RF frequencies (Hz), recorded "
                   "in capture_settings.  Set to empty string to omit.")
@click.option("--simulate/--no-simulate", default=False)
@click.option("--n-captures", type=int, default=0, show_default=True,
              help="stop after N captures (0 = run forever)")
def monitor_cmd(out_dir, prefix, interval, duration, sample_rate, channels,
                range_volts, coupling, auto_range, headroom, probe_duration_s,
                rf_bands, simulate, n_captures):
    """Continuous capture: take a recording every --interval seconds,
    write it to OUT_DIR, repeat.  Stop with Ctrl-C or after --n-captures."""
    import os, time, signal, sys
    os.makedirs(out_dir, exist_ok=True)
    chans = [c.strip().upper() for c in channels.split(",")]
    rf_band_list = [float(b) for b in rf_bands.split(",") if b.strip()] \
        if rf_bands else []
    if auto_range:
        ranges = None
    else:
        ranges = {c: range_volts for c in chans}

    stop = {"flag": False}
    def _h(*_): stop["flag"] = True
    signal.signal(signal.SIGINT, _h)
    signal.signal(signal.SIGTERM, _h)

    n_done = 0
    while not stop["flag"]:
        start = time.time()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = os.path.join(out_dir, f"{prefix}_{ts}.h5")
        click.echo(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] "
                   f"capturing -> {os.path.basename(out)} "
                   f"({duration:.1f} s @ {sample_rate:.0f} Hz)")
        try:
            capture_func(out, duration_s=duration, sample_rate=sample_rate,
                         channels=chans, ranges_volts=ranges, coupling=coupling,
                         auto_range_enabled=auto_range, headroom=headroom,
                         probe_duration_s=probe_duration_s,
                         rf_bands=rf_band_list,
                         simulate=simulate)
        except Exception as e:
            click.echo(f"  capture failed: {e}", err=True)

        n_done += 1
        if n_captures and n_done >= n_captures:
            click.echo(f"reached --n-captures {n_captures}; stopping")
            break

        # Sleep until the next interval boundary, ignoring the time the
        # capture itself took.
        elapsed = time.time() - start
        sleep_for = max(0.0, interval - elapsed)
        if sleep_for > 0 and not stop["flag"]:
            click.echo(f"  sleeping {sleep_for:.1f} s until next capture")
            for _ in range(int(sleep_for)):
                if stop["flag"]: break
                time.sleep(1.0)
    click.echo(f"monitor stopped after {n_done} captures.")


def main():
    cli()


if __name__ == "__main__":
    main()
