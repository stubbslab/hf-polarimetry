"""triloop.cli — command-line interface (Click-based).

Subcommands:
    triloop analyze FILE   --carrier ... --bw ... --az ... --el ...
    triloop summary FILE
    triloop notebook FILE  (launches the template notebook with FILE preset)
    triloop simulate       (writes a synthetic capture for testing)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import click
import numpy as np


@click.group()
@click.version_option(prog_name="triloop")
def cli():
    """triloop — three-loop magnetic-field array analysis."""


# -------------------------------------------------------------------- analyze
@cli.command(name="analyze")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--carrier", "carrier_hz", type=float, required=True,
              help="target carrier frequency, Hz")
@click.option("--bw", "bw_hz", type=float, default=2000.0, show_default=True,
              help="analysis bandwidth, Hz")
@click.option("--az", "az_deg", type=float, required=True,
              help="initial guess azimuth (deg from N CW)")
@click.option("--el", "el_deg", type=float, required=True,
              help="initial guess elevation (deg above horizon)")
@click.option("--loops-channels", default="A,B,C", show_default=True,
              help="comma-separated channel names for L1,L2,L3")
@click.option("--out", "out_path", type=click.Path(), default=None,
              help="write JSON summary to this file (default: stdout)")
def analyze_cmd(input_file, carrier_hz, bw_hz, az_deg, el_deg,
                loops_channels, out_path):
    """Analyze a triloop HDF5 capture and emit a JSON summary."""
    from . import read_capture, analyze, default_loops_config
    cap = read_capture(input_file)
    chans = [c.strip() for c in loops_channels.split(",")]
    if len(chans) != 3:
        click.echo("must give exactly 3 channel names for the loops",
                   err=True); sys.exit(2)
    try:
        B = [cap["channels"][c] for c in chans]
    except KeyError as e:
        click.echo(f"channel {e} not found in capture; available: "
                   f"{list(cap['channels'].keys())}", err=True); sys.exit(2)

    loops_cfg = cap.get("loops_config") or default_loops_config()
    res = analyze(cap["time"], B[0], B[1], B[2],
                  carrier_hz, bw_hz, az_deg, el_deg,
                  loops_config=loops_cfg)

    summary = {
        "input_file": os.path.abspath(input_file),
        "sample_rate_Hz": res.sample_rate,
        "duration_s": res.duration_s,
        "f_peak_Hz": res.f_peak,
        "snr_db_per_loop": res.snr_db_per_loop.tolist(),
        "az_initial_deg": az_deg,
        "el_initial_deg": el_deg,
        "median_pol_fraction": float(np.median(res.pol_fraction)),
        "median_ellipticity_deg": float(np.median(res.ellipticity_deg)),
        "median_position_angle_deg": float(np.median(res.position_angle_deg)),
        "instant_freq_mean_Hz": res.instant_freq_mean,
        "intensity_mean": float(np.mean(res.intensity)),
        "intensity_std": float(np.std(res.intensity)),
    }
    js = json.dumps(summary, indent=2)
    if out_path:
        with open(out_path, "w") as f: f.write(js)
        click.echo(f"wrote {out_path}")
    else:
        click.echo(js)


# -------------------------------------------------------------------- summary
# --------------------------------------------------------------------- locate
@cli.command(name="locate")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--carrier", "carrier_hz", type=float, required=True)
@click.option("--bw", "bw_hz", type=float, default=2000.0, show_default=True)
@click.option("--az0", "az0_deg", type=float, default=None,
              help="initial azimuth guess for null sweep "
                   "(default: use eigendecomp result)")
@click.option("--el0", "el0_deg", type=float, default=None,
              help="initial elevation guess for null sweep")
@click.option("--half-width", type=float, default=20.0, show_default=True,
              help="half-width of null-sweep grid (deg)")
@click.option("--n-points", type=int, default=81, show_default=True)
@click.option("--no-refine", is_flag=True,
              help="skip the 2-D null sweep, return only the eigendecomp")
@click.option("--loops-channels", default="A,B,C", show_default=True)
@click.option("--out-png", type=click.Path(), default=None,
              help="if given, also write a residual-map PNG here")
def locate_cmd(input_file, carrier_hz, bw_hz, az0_deg, el0_deg,
               half_width, n_points, no_refine, loops_channels, out_png):
    """Estimate the source direction by null-eigenvector + 2-D null sweep,
    then lock and analyze."""
    from . import (read_capture, default_loops_config, lock_and_analyze)
    cap = read_capture(input_file)
    chans = [c.strip() for c in loops_channels.split(",")]
    B = [cap["channels"][c] for c in chans]
    cfg = cap.get("loops_config") or default_loops_config()
    res = lock_and_analyze(cap["time"], B[0], B[1], B[2],
                           carrier_hz, bw_hz,
                           az0=az0_deg, el0=el0_deg,
                           loops_config=cfg,
                           refine=not no_refine,
                           half_width_deg=half_width,
                           n_points=n_points)

    out = {
        "input_file": os.path.abspath(input_file),
        "f_peak_Hz": res["f_peak"],
        "snr_db_per_loop": res["snr_db_per_loop"].tolist(),
        "eig_az_deg": res["eig"]["az_deg"],
        "eig_el_deg": res["eig"]["el_deg"],
        "eig_eigenvalues": res["eig"]["eigenvalues"].tolist(),
        "eig_null_strength": res["eig"]["null_strength"],
        "az_locked_deg": res["az_locked"],
        "el_locked_deg": res["el_locked"],
        "khat": res["khat"].tolist(),
        "median_pol_fraction": float(np.median(res["pol_fraction"])),
        "median_ellipticity_deg": float(np.median(res["ellipticity_deg"])),
        "median_position_angle_deg": float(np.median(res["position_angle_deg"])),
        "median_intensity": float(np.median(res["intensity"])),
    }
    click.echo(json.dumps(out, indent=2))

    if out_png and res["sweep"] is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sw = res["sweep"]
        fig, ax = plt.subplots(figsize=(8, 5))
        # log-scale residual since the dynamic range can be huge
        R = sw["residual"]
        Rdb = 10 * np.log10(R / R.max())
        im = ax.imshow(Rdb, origin="lower", aspect="auto", cmap="viridis_r",
                       extent=(sw["az_grid"][0], sw["az_grid"][-1],
                               sw["el_grid"][0], sw["el_grid"][-1]))
        fig.colorbar(im, ax=ax, label="perp-residual energy (dB, peak-norm)")
        ax.plot(res["eig"]["az_deg"], res["eig"]["el_deg"], "rx", ms=12, mew=2,
                label=f"eigendecomp ({res['eig']['az_deg']:.2f}°, "
                      f"{res['eig']['el_deg']:.2f}°)")
        ax.plot(res["az_locked"], res["el_locked"], "wo", ms=12, mew=2,
                mfc="none", label=f"null-locked "
                f"({res['az_locked']:.2f}°, {res['el_locked']:.2f}°)")
        ax.set_xlabel("azimuth (deg, CW from N)")
        ax.set_ylabel("elevation (deg)")
        ax.set_title("Null-eigenvector direction finding\n"
                     "minimum = best direction estimate")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
        click.echo(f"\nwrote {out_png}", err=True)


# --------------------------------------------------------------------- batch
@cli.command(name="batch")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--carrier", "carrier_hz", type=float, required=True,
              help="target carrier frequency (Hz)")
@click.option("--bw", "bw_hz", type=float, default=2000.0, show_default=True)
@click.option("--az", "az_deg", type=float, required=True)
@click.option("--el", "el_deg", type=float, required=True)
@click.option("--loops-channels", default="A,B,C", show_default=True)
@click.option("--pattern", default="*.h5", show_default=True,
              help="glob pattern for capture files within INPUT_DIR")
@click.option("--out", "out_path", type=click.Path(), default=None,
              help="write combined JSON-lines summary to this file "
                   "(default: <input_dir>/triloop_batch_summary.jsonl)")
@click.option("--workers", type=int, default=1, show_default=True,
              help="parallel worker processes")
def batch_cmd(input_dir, carrier_hz, bw_hz, az_deg, el_deg,
              loops_channels, pattern, out_path, workers):
    """Analyze every capture file in INPUT_DIR matching --pattern.
    Writes one JSON line per file to a combined summary."""
    import glob
    from concurrent.futures import ProcessPoolExecutor, as_completed

    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        click.echo(f"no files matched {pattern} in {input_dir}", err=True)
        sys.exit(2)

    if out_path is None:
        out_path = os.path.join(input_dir, "triloop_batch_summary.jsonl")
    chans = tuple(c.strip() for c in loops_channels.split(","))

    click.echo(f"found {len(files)} files to analyze; writing {out_path}")

    def _do_one(path):
        try:
            from . import read_capture, analyze, default_loops_config
            cap = read_capture(path)
            B = [cap["channels"][c] for c in chans]
            cfg = cap.get("loops_config") or default_loops_config()
            res = analyze(cap["time"], B[0], B[1], B[2],
                          carrier_hz, bw_hz, az_deg, el_deg,
                          loops_config=cfg)
            return path, dict(
                input_file=os.path.abspath(path),
                start_time_utc=cap["start_time_utc"],
                sample_rate_Hz=res.sample_rate,
                duration_s=res.duration_s,
                f_peak_Hz=res.f_peak,
                snr_db_per_loop=res.snr_db_per_loop.tolist(),
                az_deg=az_deg, el_deg=el_deg,
                median_pol_fraction=float(np.median(res.pol_fraction)),
                median_ellipticity_deg=float(np.median(res.ellipticity_deg)),
                median_position_angle_deg=float(np.median(res.position_angle_deg)),
                intensity_mean=float(np.mean(res.intensity)),
                intensity_std=float(np.std(res.intensity)),
            ), None
        except Exception as e:
            return path, None, repr(e)

    n_ok = n_err = 0
    with open(out_path, "w") as fout:
        if workers <= 1:
            iterator = (_do_one(p) for p in files)
        else:
            ex = ProcessPoolExecutor(max_workers=workers)
            futures = [ex.submit(_do_one, p) for p in files]
            iterator = (f.result() for f in as_completed(futures))
        for i, item in enumerate(iterator):
            path, summary, err = item
            if err is not None:
                n_err += 1
                click.echo(f"  [{i+1}/{len(files)}] FAIL {os.path.basename(path)}: {err}",
                           err=True)
                continue
            n_ok += 1
            fout.write(json.dumps(summary) + "\n")
            fout.flush()
            click.echo(f"  [{i+1}/{len(files)}] {os.path.basename(path):40s}  "
                       f"f_peak={summary['f_peak_Hz']:.2f} Hz  "
                       f"pol_frac={summary['median_pol_fraction']:.3f}  "
                       f"ellip={summary['median_ellipticity_deg']:+.1f}°")
    click.echo(f"\nfinished: {n_ok} ok, {n_err} errored.  summary: {out_path}")


# --------------------------------------------------------------------- browse
@cli.command(name="browse")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--out-html", type=click.Path(), default=None,
              help="output HTML path (default: <input>_browse.html)")
@click.option("--az", "az_deg", type=float, default=9.0, show_default=True,
              help="initial direction guess for multiband analysis")
@click.option("--el", "el_deg", type=float, default=35.0, show_default=True)
@click.option("--bw", "bw_hz", type=float, default=4000.0, show_default=True,
              help="per-band extraction bandwidth (Hz)")
@click.option("--decim", "decim_rate_hz", type=float, default=20000.0,
              show_default=True, help="output IF rate after decimation (Hz)")
@click.option("--n-anim-frames", type=int, default=60, show_default=True,
              help="number of frames in the polarization-ellipse animation")
@click.option("--loops-channels", default="A,B,C", show_default=True)
@click.option("--open-browser", "open_browser", is_flag=True,
              help="open the resulting HTML in the default browser")
def browse_cmd(input_file, out_html, az_deg, el_deg, bw_hz, decim_rate_hz,
               n_anim_frames, loops_channels, open_browser):
    """Build an interactive HTML browser for a capture file.

    The page has a tabbed layout: a full-Nyquist PSD overview, a
    time-domain panel, and one tab per RF band with PSD zoom, intensity
    time history (linear + dB), intensity CDF with P10/P1 fade depths,
    and an animated polarization ellipse with a time slider."""
    from . import read_capture
    from .browse import write_browser
    cap = read_capture(input_file)
    cap["_path"] = input_file
    chans = tuple(c.strip() for c in loops_channels.split(","))
    if out_html is None:
        base, _ = os.path.splitext(input_file)
        out_html = base + "_browse.html"
    path = write_browser(cap, out_html, az_deg=az_deg, el_deg=el_deg,
                         loops_channels=chans, bw_hz=bw_hz,
                         decim_rate_hz=decim_rate_hz,
                         n_anim_frames=n_anim_frames)
    click.echo(f"wrote {path}")
    if open_browser:
        import webbrowser
        webbrowser.open("file://" + path)


# ----------------------------------------------------------------------- view
@cli.command(name="view")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--out-png", type=click.Path(), default=None,
              help="output PNG path (default: <input>_view.png next to the input)")
@click.option("--bw", "bw_hz", type=float, default=8000.0, show_default=True,
              help="zoom-panel bandwidth around each band centre (Hz)")
@click.option("--rf-bands", "rf_bands", default=None,
              help="comma-separated RF frequencies (Hz) to annotate; "
                   "default: read rf_bands_hz from capture_settings")
def view_cmd(input_file, out_png, bw_hz, rf_bands):
    """QC plot: full-Nyquist spectrum, per-band zooms, time-domain traces,
    and the auto-range probe table."""
    from . import read_capture
    from .view import make_view_figure
    cap = read_capture(input_file)
    cap["_path"] = input_file
    bands = ([float(b) for b in rf_bands.split(",") if b.strip()]
             if rf_bands else None)
    if out_png is None:
        base, _ = os.path.splitext(input_file)
        out_png = base + "_view.png"
    make_view_figure(cap, rf_bands=bands, bw_hz=bw_hz, out_path=out_png)
    click.echo(f"wrote {out_png}")


# --------------------------------------------------------------- analyze-multi
@cli.command(name="analyze-multi")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--az", "az_deg", type=float, required=True)
@click.option("--el", "el_deg", type=float, required=True)
@click.option("--bw", "bw_hz", type=float, default=4000.0, show_default=True,
              help="per-band extraction bandwidth (Hz)")
@click.option("--decim", "decim_rate_hz", type=float, default=20000.0,
              show_default=True,
              help="approximate output sample rate after decimation (Hz)")
@click.option("--rf-bands", "rf_bands", default=None,
              help="comma-separated RF frequencies (Hz) to analyze; "
                   "default: read rf_bands_hz from capture_settings")
@click.option("--loops-channels", default="A,B,C", show_default=True)
@click.option("--out-png", type=click.Path(), default=None,
              help="output PNG path (default: <input>_multiband.png)")
@click.option("--out-json", type=click.Path(), default=None,
              help="write per-band summary JSON (default: <input>_multiband.json)")
def analyze_multi_cmd(input_file, az_deg, el_deg, bw_hz, decim_rate_hz,
                      rf_bands, loops_channels, out_png, out_json):
    """Run the analyze pipeline on every RF band recorded in the capture."""
    from . import read_capture
    from .multiband import analyze_all_bands, make_multiband_figure
    cap = read_capture(input_file)
    cap["_path"] = input_file
    chans = tuple(c.strip() for c in loops_channels.split(","))
    bands = ([float(b) for b in rf_bands.split(",") if b.strip()]
             if rf_bands else None)

    results = analyze_all_bands(
        cap, az_deg=az_deg, el_deg=el_deg,
        loops_channels=chans, bw_hz=bw_hz,
        decim_rate_hz=decim_rate_hz, rf_bands=bands,
    )

    base, _ = os.path.splitext(input_file)
    if out_png is None:
        out_png = base + "_multiband.png"
    if out_json is None:
        out_json = base + "_multiband.json"

    make_multiband_figure(results, out_path=out_png)

    summary = {
        "input_file": os.path.abspath(input_file),
        "az_deg": az_deg, "el_deg": el_deg,
        "bw_hz": bw_hz, "decim_rate_hz": decim_rate_hz,
        "loops_channels": list(chans),
        "bands": [],
    }
    for f_rf in sorted(results.keys()):
        r = results[f_rf]
        ar = r.analysis
        summary["bands"].append({
            "rf_hz": f_rf,
            "baseband_hz": r.extraction.baseband_hz,
            "nyquist_zone": r.extraction.nyquist_zone,
            "inverted": bool(r.extraction.inverted),
            "decim_rate_hz": r.extraction.decim_rate_hz,
            "snr_db_per_loop": r.snrs_db.tolist(),
            "median_pol_fraction": float(np.median(ar.pol_fraction)),
            "median_ellipticity_deg": float(np.median(ar.ellipticity_deg)),
            "median_position_angle_deg": float(np.median(ar.position_angle_deg)),
            "intensity_mean": float(np.mean(ar.intensity)),
            "intensity_std": float(np.std(ar.intensity)),
            "instant_freq_mean_Hz": float(ar.instant_freq_mean),
        })
    with open(out_json, "w") as f:
        f.write(json.dumps(summary, indent=2))
    click.echo(f"wrote {out_png}")
    click.echo(f"wrote {out_json}")
    # also print a 1-line per band digest to stdout
    for b in summary["bands"]:
        click.echo(
            f"  {b['rf_hz']/1e6:5.2f} MHz  zone={b['nyquist_zone']}"
            + (" inv " if b["inverted"] else "     ")
            + f"  SNR={','.join(f'{s:5.1f}' for s in b['snr_db_per_loop'])} dB"
            + f"  pol_frac={b['median_pol_fraction']:.3f}"
            + f"  ellip={b['median_ellipticity_deg']:+5.1f}°"
        )


@cli.command(name="summary")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
def summary_cmd(input_file):
    """Print metadata + channel inventory of a triloop HDF5 capture."""
    from . import read_capture
    cap = read_capture(input_file)
    click.echo(f"file:           {input_file}")
    click.echo(f"format_version: {cap['format_version']}")
    click.echo(f"start_time_utc: {cap['start_time_utc']}")
    click.echo(f"sample_rate:    {cap['sample_rate']:.6g} Hz")
    click.echo(f"duration_s:     {cap['duration_s']:.6g} s")
    click.echo(f"scope:          {cap['scope_model']} (s/n {cap['scope_serial']})")
    click.echo(f"channels:")
    for name, arr in cap["channels"].items():
        click.echo(f"  {name}: n={arr.size}, "
                   f"range=[{arr.min():+.4g}, {arr.max():+.4g}]")
    if cap["loops_config"]:
        click.echo(f"loops_config: {len(cap['loops_config'].get('loops', []))} loops")


# -------------------------------------------------------------------- simulate
@cli.command(name="simulate")
@click.argument("output_file", type=click.Path(dir_okay=False))
@click.option("--duration", type=float, default=2.0)
@click.option("--rate", "sample_rate", type=float, default=200_000.0,
              show_default=True, help="sample rate (Hz)")
@click.option("--carrier", type=float, default=25_000.0, show_default=True,
              help="IF carrier frequency to inject (Hz)")
@click.option("--pol", type=click.Choice(["linear_vertical", "linear_horizontal",
                                          "rcp", "lcp", "elliptical"]),
              default="linear_vertical", show_default=True)
@click.option("--az", type=float, default=273.0, show_default=True)
@click.option("--el", type=float, default=12.0, show_default=True)
@click.option("--snr", type=float, default=30.0, show_default=True)
@click.option("--faraday", type=float, default=0.0, show_default=True,
              help="Faraday rotation rate, deg/s")
def simulate_cmd(output_file, duration, sample_rate, carrier,
                 pol, az, el, snr, faraday):
    """Generate a synthetic triloop capture, suitable for testing."""
    # local import keeps the CLI startup fast
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), "..", "..", "three_loop_array", "code"
    ))
    from three_loop import simulate_wwv
    from . import write_capture, default_loops_config

    t, B1, B2, B3, truth = simulate_wwv(
        duration_s=duration, sample_rate=sample_rate,
        f_RF=20.0e6, fs_offset=carrier,
        az_deg=az, el_deg=el,
        pol=pol, amp=1.0, snr_db=snr,
        faraday_rate_dps=faraday,
        seed=0,
    )
    cfg = default_loops_config()
    write_capture(
        output_file,
        channels={"A": B1.astype(np.float32),
                  "B": B2.astype(np.float32),
                  "C": B3.astype(np.float32)},
        sample_rate=sample_rate,
        start_time_utc=datetime.now(timezone.utc).isoformat(),
        scope_model="SIMULATOR",
        scope_serial="sim-0",
        capture_settings={"truth": truth},
        loops_config=cfg,
    )
    click.echo(f"wrote {output_file}")


# -------------------------------------------------------------------- notebook
@cli.command(name="notebook")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
def notebook_cmd(input_file):
    """Launch the analysis notebook preloaded with INPUT_FILE."""
    here = os.path.dirname(os.path.abspath(__file__))
    template = os.path.join(here, "..", "notebooks", "triloop_analysis.ipynb")
    if not os.path.exists(template):
        click.echo(f"notebook template not found at {template}", err=True)
        sys.exit(2)
    workdir = tempfile.mkdtemp(prefix="triloop-nb-")
    out = os.path.join(workdir, "triloop_analysis.ipynb")
    # Substitute the INPUT_FILE_PATH placeholder
    with open(template, "r") as f: text = f.read()
    text = text.replace("__INPUT_FILE_PATH__", os.path.abspath(input_file))
    with open(out, "w") as f: f.write(text)
    click.echo(f"launching jupyter on {out}")
    subprocess.call(["jupyter", "lab", out])


def main():
    cli()


if __name__ == "__main__":
    main()
