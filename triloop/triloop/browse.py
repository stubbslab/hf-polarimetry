"""triloop.browse — interactive single-file HTML browser for a capture.

Builds a self-contained HTML page (no server, no external network) using
Plotly.  Layout:

  Tab 1 "Overview":   full-Nyquist PSD with each channel as a toggle-able
                      trace, pan/zoom, RF-band markers
  Tab 2 "Time series": per-channel time-domain amplitude (decimated)
  Tab N "<RF MHz>":   one tab per band, with four panels —
                      (a) PSD zoom around the band centre
                      (b) intensity time history (linear and dB)
                      (c) intensity CDF / fade depth
                      (d) animated polarization ellipse in (p̂,q̂)
                          with time slider and play button.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional, Sequence

import numpy as np

from .bands import alias_of
from .multiband import analyze_all_bands


# Per-channel trace colours (consistent across all panels).
_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
            "#ff7f0e", "#8c564b", "#17becf", "#e377c2"]


# --------------------------------------------------------------- helpers

def _channel_psd(x: np.ndarray, sr: float, nfft: Optional[int] = None):
    n = x.size
    if nfft is None:
        nfft = 1 << int(np.ceil(np.log2(min(n, 1 << 18))))
    nfft = min(nfft, n)
    xw = x[:nfft] * np.hanning(nfft)
    S = np.fft.rfft(xw)
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    p_db = 10.0 * np.log10(np.maximum(np.abs(S) ** 2 / (nfft * sr), 1e-30))
    return f, p_db


def _band_zoom_spectrum(x: np.ndarray, sr: float, f_bb: float, bw: float,
                        inverted: bool, nfft: int = 1 << 16):
    n = x.size
    nfft = min(nfft, n)
    xw = x[:nfft] * np.hanning(nfft)
    S = np.fft.fft(xw)
    f = np.fft.fftfreq(nfft, 1.0 / sr)
    df = f - f_bb
    mask = np.abs(df) <= bw / 2.0
    p_db = 10.0 * np.log10(np.maximum(np.abs(S[mask]) ** 2, 1e-30))
    f_off = df[mask]
    order = np.argsort(f_off)
    f_off = f_off[order]
    p_db = p_db[order]
    if inverted:
        f_off = -f_off[::-1]
        p_db = p_db[::-1]
    return f_off, p_db


def _smooth_decimate(y: np.ndarray, target_n: int) -> np.ndarray:
    """Box-average down to ~target_n samples for fast plotting / CDF."""
    if y.size <= target_n:
        return y
    stride = int(np.ceil(y.size / target_n))
    n_keep = (y.size // stride) * stride
    return y[:n_keep].reshape(-1, stride).mean(axis=1)


def _cdf(values: np.ndarray):
    v = np.sort(values)
    p = np.linspace(0.0, 1.0, v.size, endpoint=False) + 1.0 / v.size
    return v, p


# --------------------------------------------------------------- public

def build_browser_html(cap: dict, *,
                       az_deg: float = 9.0,
                       el_deg: float = 35.0,
                       loops_channels: Sequence[str] = ("A", "B", "C"),
                       bw_hz: float = 4000.0,
                       decim_rate_hz: float = 20_000.0,
                       n_anim_frames: int = 60,
                       title: Optional[str] = None) -> str:
    """Return a self-contained HTML string."""
    import plotly.graph_objects as go
    from plotly.io import to_html

    sr = float(cap["sample_rate"])
    cs = cap.get("capture_settings", {}) or {}
    rf_bands = [float(f) for f in (cs.get("rf_bands_hz") or [])]
    chans = list(cap["channels"].keys())
    if title is None:
        title = os.path.basename(cap.get("_path", "capture"))

    # ------------------------------------------------ multiband analysis
    if rf_bands:
        results = analyze_all_bands(
            cap, az_deg=az_deg, el_deg=el_deg,
            loops_channels=tuple(loops_channels),
            bw_hz=bw_hz, decim_rate_hz=decim_rate_hz,
        )
    else:
        results = {}

    # ------------------------------------------------ Overview PSD
    overview = go.Figure()
    for i, ch in enumerate(chans):
        f, p = _channel_psd(cap["channels"][ch], sr)
        # downsample the PSD for display (still informative)
        keep = max(1, len(f) // 6000)
        overview.add_trace(go.Scattergl(
            x=f[::keep] / 1e6, y=p[::keep], mode="lines",
            name=ch, line=dict(width=1.0, color=_PALETTE[i % len(_PALETTE)]),
        ))
    band_shapes = []
    band_annotations = []
    for f_rf in rf_bands:
        a = alias_of(f_rf, sr)
        band_shapes.append(dict(
            type="line", x0=a.baseband_hz / 1e6, x1=a.baseband_hz / 1e6,
            xref="x", yref="paper", y0=0, y1=1,
            line=dict(color="rgba(0,0,0,0.45)", width=1, dash="dash"),
        ))
        tag = f"{f_rf/1e6:g} MHz<br>z{a.nyquist_zone}"
        if a.inverted:
            tag += " inv"
        band_annotations.append(dict(
            x=a.baseband_hz / 1e6, y=1.0, xref="x", yref="paper",
            text=tag, showarrow=False, yanchor="bottom",
            font=dict(size=10),
        ))
    overview.update_layout(
        title=f"Full-Nyquist PSD — sr={sr/1e6:.4f} MS/s, "
              f"dur={cap['duration_s']:.2f} s",
        xaxis_title="frequency (MHz, baseband)",
        yaxis_title="PSD (dB, arb.)",
        shapes=band_shapes, annotations=band_annotations,
        hovermode="x unified",
        margin=dict(l=60, r=20, t=60, b=50),
        height=540,
    )

    # ------------------------------------------------ Time-domain
    n0 = next(iter(cap["channels"].values())).size
    stride = max(1, n0 // 8000)
    t_plot = cap["time"][::stride]
    timefig = go.Figure()
    for i, ch in enumerate(chans):
        timefig.add_trace(go.Scattergl(
            x=t_plot, y=cap["channels"][ch][::stride], mode="lines",
            name=ch, line=dict(width=0.8,
                               color=_PALETTE[i % len(_PALETTE)]),
        ))
    timefig.update_layout(
        title="Time-domain amplitude (decimated)",
        xaxis_title="time (s)", yaxis_title="amplitude (V)",
        hovermode="x unified",
        margin=dict(l=60, r=20, t=60, b=50),
        height=380,
    )

    # ------------------------------------------------ Per-band tabs
    band_html_blocks = []
    for f_rf in sorted(results.keys()):
        r = results[f_rf]
        be = r.extraction
        ar = r.analysis
        band_label = f"{f_rf/1e6:g} MHz"

        # (a) PSD zoom
        zoom = go.Figure()
        for i, ch in enumerate(chans):
            f_off, p_db = _band_zoom_spectrum(
                cap["channels"][ch], sr,
                be.baseband_hz, bw_hz, be.inverted
            )
            zoom.add_trace(go.Scattergl(
                x=f_off / 1e3, y=p_db, mode="lines",
                name=ch, line=dict(width=1.0,
                                   color=_PALETTE[i % len(_PALETTE)]),
            ))
        zoom.update_layout(
            title=f"PSD zoom — {band_label} (zone {be.nyquist_zone}"
                  + (", inverted" if be.inverted else "") + ")",
            xaxis_title="Δf from RF (kHz)",
            yaxis_title="PSD (dB)",
            hovermode="x unified",
            margin=dict(l=50, r=10, t=50, b=40), height=320,
        )

        # (b) intensity time history (linear and dB)
        I = ar.intensity
        I_med = float(np.median(I))
        I_db = 10.0 * np.log10(np.maximum(I / I_med, 1e-12))
        # decimate for display
        t_d = be.t
        max_pts = 8000
        if t_d.size > max_pts:
            keep = max(1, t_d.size // max_pts)
            t_dd = t_d[::keep]; I_dd = I[::keep]; I_db_dd = I_db[::keep]
        else:
            t_dd, I_dd, I_db_dd = t_d, I, I_db

        intens = go.Figure()
        intens.add_trace(go.Scattergl(
            x=t_dd, y=I_dd, mode="lines", name="|B⊥|²",
            line=dict(width=1.0, color="#1f77b4"),
        ))
        intens.add_trace(go.Scattergl(
            x=t_dd, y=I_db_dd, mode="lines", name="dB rel. median",
            line=dict(width=1.0, color="#d62728"),
            yaxis="y2", visible="legendonly",
        ))
        intens.update_layout(
            title=f"Intensity time history — {band_label}",
            xaxis_title="time (s)",
            yaxis=dict(title="|B⊥|²", side="left"),
            yaxis2=dict(title="dB rel. median", overlaying="y",
                        side="right"),
            hovermode="x unified", showlegend=True,
            margin=dict(l=50, r=50, t=50, b=40), height=320,
        )

        # (c) CDF
        v, p = _cdf(_smooth_decimate(I, 4000))
        v_db = 10.0 * np.log10(np.maximum(v / I_med, 1e-12))
        cdf = go.Figure()
        cdf.add_trace(go.Scattergl(
            x=v_db, y=p, mode="lines", name="CDF",
            line=dict(width=1.5, color="#2ca02c"),
        ))
        # P10 and P1 fade-depth markers
        for q, label, col in [(0.1, "P10", "#ff7f0e"),
                              (0.01, "P1",  "#9467bd")]:
            if v_db.size:
                idx = int(q * v_db.size)
                fd = float(v_db[idx])
                cdf.add_shape(
                    type="line", x0=fd, x1=fd, y0=0, y1=q,
                    line=dict(color=col, width=1.5, dash="dot"),
                )
                cdf.add_annotation(
                    x=fd, y=q, text=f"{label}: {fd:.1f} dB",
                    showarrow=False, xanchor="left",
                    yanchor="bottom", font=dict(color=col, size=11),
                )
        cdf.update_layout(
            title=f"Intensity CDF — {band_label}",
            xaxis_title="intensity (dB rel. median)",
            yaxis=dict(title="P(intensity ≤ x)", range=[0, 1]),
            hovermode="x unified", showlegend=False,
            margin=dict(l=50, r=10, t=50, b=40), height=320,
        )

        # (d) polarization-ellipse animation
        anim = _build_pol_animation(ar, n_frames=n_anim_frames,
                                    band_label=band_label)

        band_html_blocks.append(dict(
            label=band_label,
            blocks=[zoom, intens, cdf, anim],
        ))

    # ------------------------------------------------ assemble HTML
    parts = []
    parts.append(_HEAD.format(title=title))
    parts.append(f"<h1>triloop browser — {title}</h1>")
    parts.append(_meta_block(cap))

    # Tabs
    tab_labels = ["Overview", "Time series"] + [b["label"] for b in band_html_blocks]
    parts.append('<div class="tabs">')
    for i, lbl in enumerate(tab_labels):
        active = " active" if i == 0 else ""
        parts.append(f'<button class="tab{active}" '
                     f'onclick="showTab({i})" id="btn{i}">{lbl}</button>')
    parts.append('</div>')

    # Tab panels
    parts.append('<div class="panel active" id="panel0">')
    parts.append(to_html(overview, full_html=False, include_plotlyjs="cdn",
                         div_id="overview"))
    parts.append('</div>')

    parts.append('<div class="panel" id="panel1">')
    parts.append(to_html(timefig, full_html=False, include_plotlyjs=False,
                         div_id="timefig"))
    parts.append('</div>')

    for k, b in enumerate(band_html_blocks):
        parts.append(f'<div class="panel" id="panel{k+2}">')
        parts.append('<div class="grid2">')
        for j, fig in enumerate(b["blocks"]):
            div_id = f"band{k}_panel{j}"
            parts.append(to_html(fig, full_html=False, include_plotlyjs=False,
                                 div_id=div_id))
        parts.append('</div>')
        parts.append('</div>')

    parts.append(_TAIL)
    return "\n".join(parts)


# --------------------------------------------------- polarization animation

def _build_pol_animation(ar, n_frames: int = 60, band_label: str = ""):
    """Animated locus of the instantaneous (Re A_p, Re A_q) over slow time.

    For each frame, draw the polarization ellipse over a short window
    (~ 2× the carrier period at the decimated rate) so the user sees how
    the ellipse rotates / changes shape with time.  A persistent
    background trace shows the cumulative locus to date for context.
    """
    import plotly.graph_objects as go

    A_p = ar.A_p
    A_q = ar.A_q
    n = A_p.size
    # Window size: enough samples to draw a few cycles of the residual
    # carrier-phase variation.  A few hundred samples works.
    win = max(64, min(400, n // n_frames))
    if n_frames * win > n:
        n_frames = max(2, n // win)

    # Frame centres uniformly distributed across the recording
    centres = np.linspace(win // 2, n - win // 2 - 1, n_frames).astype(int)

    # Determine global axis limits from the entire trace so frames don't
    # rescale.
    lim = float(np.max(np.abs(np.concatenate([A_p.real, A_p.imag,
                                              A_q.real, A_q.imag])))) * 1.05

    frames = []
    for fi, c in enumerate(centres):
        s = slice(max(0, c - win // 2), min(n, c + win // 2))
        x = A_p[s].real
        y = A_q[s].real
        frames.append(go.Frame(
            data=[go.Scatter(x=x, y=y, mode="lines",
                             line=dict(width=1.5, color="#1f77b4")),
                  go.Scatter(x=[x[-1]], y=[y[-1]], mode="markers",
                             marker=dict(size=8, color="#d62728"))],
            name=f"{ar.intensity[c]:.4g}",
            layout=dict(
                title=f"Polarization ellipse — {band_label}<br>"
                      f"<sub>t = {ar.intensity.size and (c / ar.sample_rate):.3f} s</sub>"
            )
        ))

    # Initial frame uses the first frame's data
    init_x = A_p[centres[0] - win // 2 : centres[0] + win // 2].real
    init_y = A_q[centres[0] - win // 2 : centres[0] + win // 2].real
    fig = go.Figure(
        data=[
            go.Scatter(x=init_x, y=init_y, mode="lines",
                       line=dict(width=1.5, color="#1f77b4"),
                       name="ellipse"),
            go.Scatter(x=[init_x[-1]], y=[init_y[-1]], mode="markers",
                       marker=dict(size=8, color="#d62728"), name="now"),
        ],
        frames=frames,
    )
    fig.update_layout(
        title=f"Polarization ellipse — {band_label}<br>"
              f"<sub>t = {centres[0] / ar.sample_rate:.3f} s</sub>",
        xaxis=dict(title="Re A_p", range=[-lim, lim],
                   scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Re A_q", range=[-lim, lim]),
        margin=dict(l=50, r=10, t=70, b=40), height=420,
        showlegend=False,
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.0, y=-0.10,
            xanchor="left", yanchor="top",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=80, redraw=True),
                                       transition=dict(duration=0),
                                       fromcurrent=True)]),
                dict(label="❚❚ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                          mode="immediate",
                                          transition=dict(duration=0))]),
            ]
        )],
        sliders=[dict(
            active=0, x=0.10, y=-0.10, len=0.85,
            xanchor="left", yanchor="top",
            currentvalue=dict(prefix="frame: ",
                              font=dict(size=11)),
            steps=[dict(method="animate",
                        args=[[f.name], dict(mode="immediate",
                                              frame=dict(duration=0, redraw=True),
                                              transition=dict(duration=0))],
                        label=str(i))
                   for i, f in enumerate(frames)],
        )],
    )
    return fig


# --------------------------------------------------- HTML scaffolding

_HEAD = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>{title} — triloop browser</title>
<style>
  body {{ font-family: -apple-system, system-ui, Segoe UI, sans-serif;
          margin: 0 18px 40px; color: #222; }}
  h1   {{ margin: 12px 0 4px; font-size: 17px; font-weight: 600; }}
  .meta {{ color: #555; font-size: 12.5px; margin-bottom: 14px;
          padding-bottom: 8px; border-bottom: 1px solid #ddd; }}
  .meta b {{ color: #222; }}
  .tabs {{ border-bottom: 2px solid #1f77b4; margin: 8px 0 12px; }}
  .tab  {{ background: #f3f3f5; border: 1px solid #ccc; border-bottom: none;
          padding: 6px 14px; margin-right: 2px; cursor: pointer;
          font-size: 13px; border-radius: 6px 6px 0 0; }}
  .tab.active {{ background: #1f77b4; color: white; border-color: #1f77b4; }}
  .panel {{ display: none; }}
  .panel.active {{ display: block; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  @media (max-width: 1100px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
</style>
</head><body>
"""

_TAIL = """
<script>
function showTab(i) {
  document.querySelectorAll('.tab').forEach((b, k) => {
    b.classList.toggle('active', k === i);
  });
  document.querySelectorAll('.panel').forEach((p, k) => {
    p.classList.toggle('active', k === i);
  });
  // Plotly resize hack: when a tab becomes visible, ask Plotly to relayout
  // any plot inside it so it doesn't stay at zero width.
  window.dispatchEvent(new Event('resize'));
}
</script>
</body></html>
"""


def _meta_block(cap):
    cs = cap.get("capture_settings", {}) or {}
    rf = cs.get("rf_bands_hz") or []
    rf_str = ", ".join(f"{f/1e6:g} MHz" for f in rf) or "—"
    return (
        f'<div class="meta">'
        f'<b>start:</b> {cap.get("start_time_utc","")}  &nbsp;|&nbsp;  '
        f'<b>scope:</b> {cap.get("scope_model","?")} '
        f'(s/n {cap.get("scope_serial","?")})  &nbsp;|&nbsp;  '
        f'<b>fs:</b> {cap["sample_rate"]/1e6:.4f} MS/s  &nbsp;|&nbsp;  '
        f'<b>dur:</b> {cap["duration_s"]:.3f} s  &nbsp;|&nbsp;  '
        f'<b>channels:</b> {", ".join(cap["channels"].keys())}  &nbsp;|&nbsp;  '
        f'<b>RF bands:</b> {rf_str}'
        f'</div>'
    )


def write_browser(cap: dict, out_path: str, **kwargs) -> str:
    """Build the HTML and write it.  Returns the absolute output path."""
    html = build_browser_html(cap, **kwargs)
    with open(out_path, "w") as f:
        f.write(html)
    return os.path.abspath(out_path)
