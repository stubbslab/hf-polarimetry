#!/usr/bin/env python3
"""tools/predict_phase_batch.py

Run predict_phase.compute_predictability across an entire directory of
KiwiSDR IQ recordings, parse the filename for date / time / port /
frequency, and produce a JSONL summary plus aggregate plots.

Filename conventions
--------------------
The Kiwi recordings in this project follow the pattern

    YYYYMMDD.HHMM.PORT.proxy.kiwisdr.com.freqFFFFF.RRRRR.SUFFIX.wav

where FFFFF is the WWV/CHU frequency in kHz, RRRRR is the sample rate
in Hz, and SUFFIX is one of {wwv1, wwv2, triplet, ...}.  Other patterns
are skipped.

Outputs
-------
* ``<out-dir>/predict_phase_results.jsonl``  — one record per processed
  file with the per-tau RMS prediction errors plus metadata.
* ``<out-dir>/predict_phase_dpred20ms_vs_freq.png`` — scatter and
  per-band median of D_pred(20 ms) by RF frequency, separated by
  predictor.
* ``<out-dir>/predict_phase_dpred20ms_vs_hour.png`` — same metric vs
  UTC hour-of-day.
* ``<out-dir>/predict_phase_correctable_fraction.png`` — fraction of
  files (per band) achieving D_pred(20 ms) < 0.5 rad.

Usage
-----
    python3 tools/predict_phase_batch.py /path/to/wav_dir \
            --out-dir /tmp/predict_batch \
            --workers 8 \
            --pattern '*wwv*.wav' \
            --max-files 0          # 0 = no limit
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np

# Local import; assume the script is run with cwd at the project root or
# that the tools directory is on the path.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
from predict_phase import compute_predictability


FNAME_RE = re.compile(
    r"^(?P<date>\d{8})\.(?P<time>\d{4})\.(?P<port>\d+)\."
    r".*freq(?P<freq>\d+)\.\d+\.(?P<suffix>[a-z0-9]+)\.wav$",
    re.IGNORECASE,
)


def parse_filename(path: str):
    """Return dict of filename metadata or None if unparseable."""
    m = FNAME_RE.match(os.path.basename(path))
    if not m:
        return None
    try:
        ts = datetime.strptime(m["date"] + m["time"],
                                "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return dict(
        path=path,
        utc_time=ts.isoformat(),
        utc_hour=float(ts.hour + ts.minute / 60.0),
        port=m["port"],
        freq_khz=int(m["freq"]),
        suffix=m["suffix"].lower(),
    )


_SKIP_KALMAN = True   # set by main()


def _do_one(path: str) -> dict:
    meta = parse_filename(path)
    if meta is None:
        return {"path": path, "error": "filename did not parse"}
    res = compute_predictability(path, skip_kalman=_SKIP_KALMAN)
    if "error" in res:
        return {**meta, **res}
    # Pull out canonical scalars at tau = 20 ms and 50 ms
    taus = np.array(res["taus_s"])
    out = dict(meta)
    out.update({k: res[k] for k in (
        "sample_rate_Hz", "carrier_beat_Hz",
        "bandwidth_hz", "n_total_samples", "n_fade_samples",
        "good_fraction", "n_splices",
        "residual_slope_Hz",
        "median_amp") if k in res})
    for tau_target in (0.020, 0.050):
        i = int(np.argmin(np.abs(taus - tau_target)))
        tag = f"{int(round(tau_target*1000))}ms"
        out[f"rms_const_{tag}_rad"]  = res["rms_pred_constant_rad"][i]
        out[f"rms_linear_{tag}_rad"] = res["rms_pred_linear_rad"][i]
        out[f"rms_kalman_{tag}_rad"] = res["rms_pred_kalman_rad"][i]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_dir", help="directory containing the .wav files")
    p.add_argument("--pattern", default="*.wav",
                   help="glob pattern for files within input_dir")
    p.add_argument("--out-dir", default=None,
                   help="output directory for plots and JSONL "
                        "(default: <input_dir>/predict_phase_batch_out)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-files", type=int, default=0,
                   help="if >0, process only the first N matching files")
    p.add_argument("--max-slope-std-hz", type=float, default=5000.0,
                   help="skip files whose per-segment-slope std exceeds "
                        "this; default 5000 Hz only excludes pathological "
                        "files where unwrapping has blown up.  The linear "
                        "and Kalman predictors handle moderate residual "
                        "carrier offsets internally.")
    p.add_argument("--quick", action="store_true",
                   help="process every Nth file for fast iteration; N=20")
    p.add_argument("--with-kalman", action="store_true",
                   help="enable the Kalman predictor (slow; default off)")
    args = p.parse_args()
    global _SKIP_KALMAN
    _SKIP_KALMAN = not args.with_kalman

    out_dir = args.out_dir or os.path.join(args.input_dir,
                                           "predict_phase_batch_out")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if args.quick:
        files = files[::20]
    if args.max_files > 0:
        files = files[:args.max_files]
    print(f"found {len(files)} files matching '{args.pattern}'; "
          f"will write to {out_dir}", flush=True)
    if not files:
        return

    jsonl_path = os.path.join(out_dir, "predict_phase_results.jsonl")
    n_ok = n_err = 0
    t0 = time.time()
    use_pool = args.workers > 1
    with open(jsonl_path, "w") as fout:
        if use_pool:
            try:
                ex = ProcessPoolExecutor(max_workers=args.workers)
                futures = {ex.submit(_do_one, p): p for p in files}
                iterator = as_completed(futures)

                def _next_record(fut):
                    try:
                        return fut.result()
                    except Exception as e:
                        return {"path": futures[fut], "error": repr(e)}
            except (PermissionError, OSError) as e:
                print(f"  ProcessPoolExecutor unavailable ({e}); "
                      f"falling back to serial.", flush=True)
                use_pool = False

        if not use_pool:
            iterator = (_do_one(p) for p in files)

        for i, item in enumerate(iterator):
            if use_pool:
                rec = _next_record(item)
            else:
                rec = item
            fout.write(json.dumps(rec) + "\n")
            if rec.get("error"):
                n_err += 1
            else:
                n_ok += 1
            if (i + 1) % 100 == 0 or i + 1 == len(files):
                rate = (i + 1) / max(time.time() - t0, 1e-3)
                print(f"  [{i+1:5d}/{len(files)}]  "
                      f"ok={n_ok:5d}  err={n_err:5d}  "
                      f"({rate:.1f} files/s)", flush=True)
        if use_pool:
            ex.shutdown(wait=True)

    print(f"\nfinished: {n_ok} ok, {n_err} errored.")
    print(f"  results: {jsonl_path}")
    aggregate(jsonl_path, out_dir, args.max_slope_std_hz)


def aggregate(jsonl_path: str, out_dir: str, max_slope_std_hz: float,
              min_good_fraction: float = 0.2,
              max_residual_slope_hz: float = 50.0):
    print(f"\nAggregating {jsonl_path}…")
    rows = []
    n_total = n_err = n_no_carrier = n_bad_carrier = n_too_faded = 0
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            if rec.get("error"):
                n_err += 1
                continue
            gf = rec.get("good_fraction")
            rs = rec.get("residual_slope_Hz")
            if gf is None and rs is None:
                n_no_carrier += 1
                continue
            if gf is not None and gf < min_good_fraction:
                n_too_faded += 1
                continue
            if rs is not None and rs == rs and abs(rs) > max_residual_slope_hz:
                n_bad_carrier += 1
                continue
            rows.append(rec)
    print(f"  total records:            {n_total}")
    print(f"    errored:                {n_err}")
    print(f"    no carrier metric:      {n_no_carrier}")
    print(f"    too faded (<{min_good_fraction*100:.0f}% good): "
          f"{n_too_faded}")
    print(f"    bad carrier (|slope|>{max_residual_slope_hz:.0f} Hz): "
          f"{n_bad_carrier}")
    print(f"    usable:                 {len(rows)}")
    if not rows:
        print("  nothing to aggregate; exiting")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freq_arr  = np.array([r["freq_khz"] / 1000.0 for r in rows])
    hour_arr  = np.array([r["utc_hour"] for r in rows])
    snr_arr   = np.array([20 * np.log10(max(r["median_amp"], 1e-9))
                          for r in rows])
    rmsC_20  = np.array([r["rms_const_20ms_rad"]  for r in rows])
    rmsL_20  = np.array([r["rms_linear_20ms_rad"] for r in rows])
    rmsK_20  = np.array([r["rms_kalman_20ms_rad"] for r in rows])
    rmsC_50  = np.array([r["rms_const_50ms_rad"]  for r in rows])

    # ---------- plot 1: D_pred(20 ms) vs RF frequency
    fig, ax = plt.subplots(figsize=(9, 5))
    bands = sorted(set(freq_arr.tolist()))
    box_const  = [rmsC_20[freq_arr == b] for b in bands]
    box_linear = [rmsL_20[freq_arr == b] for b in bands]
    box_kalman = [rmsK_20[freq_arr == b] for b in bands]
    pos = np.arange(len(bands))
    width = 0.25
    bp_c = ax.boxplot(box_const,  positions=pos - width, widths=width,
                       patch_artist=True,
                       boxprops=dict(facecolor="#1f77b4aa"),
                       medianprops=dict(color="white"))
    bp_l = ax.boxplot(box_linear, positions=pos,         widths=width,
                       patch_artist=True,
                       boxprops=dict(facecolor="#ff7f0eaa"),
                       medianprops=dict(color="white"))
    bp_k = ax.boxplot(box_kalman, positions=pos + width, widths=width,
                       patch_artist=True,
                       boxprops=dict(facecolor="#2ca02caa"),
                       medianprops=dict(color="white"))
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{b:g}\nMHz" for b in bands])
    ax.set_xlabel("RF band")
    ax.set_ylabel("RMS prediction error at τ=20 ms (rad)")
    ax.set_title(f"Phase predictability at 20 ms vs WWV/CHU band  "
                 f"(N={len(rows)} files)")
    ax.set_yscale("log")
    ax.axhline(0.5, color="0.4", lw=1, ls="--",
               label="0.5 rad correction threshold")
    ax.legend([bp_c["boxes"][0], bp_l["boxes"][0], bp_k["boxes"][0],
               ax.lines[-1]],
              ["constant-phase", "linear extrap.", "Kalman",
               "0.5 rad goal"], loc="upper left")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "predict_phase_dpred20ms_vs_freq.png")
    fig.savefig(p1, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p1}")

    # ---------- plot 2: D_pred(20 ms) vs UTC hour, separated by band
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    for k, b in enumerate(bands):
        mask = (freq_arr == b)
        if not np.any(mask):
            continue
        # bin by hour
        h_bins = np.arange(0, 25)
        h_centers = 0.5 * (h_bins[:-1] + h_bins[1:])
        med = np.array([
            np.median(rmsC_20[mask & (hour_arr >= lo) & (hour_arr < hi)])
            if np.any(mask & (hour_arr >= lo) & (hour_arr < hi))
            else np.nan
            for lo, hi in zip(h_bins[:-1], h_bins[1:])
        ])
        ax.plot(h_centers, med, "-o", color=cmap(k % 10), ms=4,
                label=f"{b:g} MHz")
    ax.set_xlabel("UTC hour")
    ax.set_ylabel("median RMS const-phase pred. err at 20 ms (rad)")
    ax.set_title("Phase predictability vs UTC hour (constant-phase predictor)")
    ax.set_yscale("log")
    ax.axhline(0.5, color="0.4", lw=1, ls="--",
               label="0.5 rad correction threshold")
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xlim(0, 24)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "predict_phase_dpred20ms_vs_hour.png")
    fig.savefig(p2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p2}")

    # ---------- plot 3: correctable-fraction per band
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for predictor, arr, color in [("constant-phase", rmsC_20, "C0"),
                                   ("linear",        rmsL_20, "C1"),
                                   ("Kalman",        rmsK_20, "C2")]:
        frac = []
        for b in bands:
            mask = (freq_arr == b) & np.isfinite(arr)
            if not np.any(mask):
                frac.append(np.nan); continue
            frac.append(float(np.mean(arr[mask] < 0.5)))
        ax.plot(bands, np.array(frac) * 100.0, "-o", color=color,
                label=predictor)
    ax.set_xlabel("RF band (MHz)")
    ax.set_ylabel("% of files with D_pred(20 ms) < 0.5 rad")
    ax.set_title(f"Fraction of files achieving the 0.5 rad correction "
                 f"threshold  (N={len(rows)})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-3, 103)
    fig.tight_layout()
    p3 = os.path.join(out_dir, "predict_phase_correctable_fraction.png")
    fig.savefig(p3, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p3}")

    # ---------- text summary
    print(f"\nMedian D_pred(20 ms) by band, constant-phase predictor:")
    print(f"  {'band':>6}  {'n':>5}  {'med rms':>8}  {'<0.5 rad':>9}")
    for b in bands:
        mask = freq_arr == b
        if not np.any(mask): continue
        med = float(np.median(rmsC_20[mask]))
        frac = float(np.mean(rmsC_20[mask] < 0.5)) * 100.0
        print(f"  {b:>5g}M  {int(mask.sum()):>5d}  "
              f"{med:8.3f}  {frac:8.1f}%")


if __name__ == "__main__":
    main()
