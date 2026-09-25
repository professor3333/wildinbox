"""Error-versus-coverage curves as static SVG (light and dark), for reports.

x: share of events handled automatically (coverage); y: error rate among what
automation would do. Increasing automation moves right; the curve shows what it
costs. One axis per chart; the table view lives in the report next to it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

W, H = 640, 360
L, R, T, B = 64, 150, 44, 52  # plot margins; right margin holds direct labels
STYLE = """
.bg{fill:#fcfcfb}.t1{fill:#0b0b0b}.t2{fill:#52514e}.grid{stroke:#e6e5e1}.axis{stroke:#8a8983}
.s1{stroke:#2a78d6;fill:#2a78d6}.s2{stroke:#eb6834;fill:#eb6834}.band{fill:#2a78d6;opacity:.14}
.target{stroke:#52514e}.ring{stroke:#fcfcfb}
@media (prefers-color-scheme: dark){
.bg{fill:#1a1a19}.t1{fill:#ffffff}.t2{fill:#c3c2b7}.grid{stroke:#333331}.axis{stroke:#6f6e69}
.s1{stroke:#3987e5;fill:#3987e5}.s2{stroke:#d95926;fill:#d95926}.band{fill:#3987e5;opacity:.2}
.target{stroke:#c3c2b7}.ring{stroke:#1a1a19}}
text{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif}
"""


@dataclass(frozen=True)
class Point:
    x: float  # coverage, 0-1
    y: float  # error rate, 0-1
    lo: float | None = None
    hi: float | None = None
    label: str = ""  # threshold, for the tooltip


@dataclass(frozen=True)
class Series:
    name: str
    points: Sequence[Point]
    band: bool = False  # draw the 95% interval


def _nice_max(v: float) -> float:
    for m in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        if v <= m:
            return m
    return 1.0


def render(
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[Series],
    target: float,
    target_label: str,
    chosen: Point | None = None,
    chosen_label: str = "",
) -> str:
    pts = [p for s in series for p in s.points]
    x_max = _nice_max(max([p.x for p in pts] + [0.01]))
    y_top = max([p.hi if p.hi is not None else p.y for p in pts] + [target * 1.5])
    y_max = _nice_max(y_top)
    pw, ph = W - L - R, H - T - B

    def sx(x: float) -> float:
        return L + pw * x / x_max

    def sy(y: float) -> float:
        return T + ph * (1 - min(y, y_max) / y_max)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" role="img" aria-label="{escape(title)}">',
        f"<title>{escape(title)}</title><style>{STYLE}</style>",
        f'<rect class="bg" width="{W}" height="{H}"/>',
        f'<text class="t1" x="{L}" y="24" font-size="15" font-weight="600">{escape(title)}</text>',
    ]
    for i in range(5):
        yv = y_max * i / 4
        out.append(
            f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{sy(yv):.1f}" y2="{sy(yv):.1f}"/>'
            f'<text class="t2" x="{L - 8}" y="{sy(yv) + 4:.1f}" font-size="11" '
            f'text-anchor="end">{100 * yv:g}%</text>'
        )
        xv = x_max * i / 4
        out.append(
            f'<text class="t2" x="{sx(xv):.1f}" y="{T + ph + 18}" font-size="11" '
            f'text-anchor="middle">{100 * xv:g}%</text>'
        )
    out.append(f'<line class="axis" x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}"/>')
    out.append(
        f'<text class="t2" x="{L + pw / 2}" y="{H - 12}" font-size="12" '
        f'text-anchor="middle">{escape(x_label)}</text>'
        f'<text class="t2" transform="translate(16 {T + ph / 2}) rotate(-90)" font-size="12" '
        f'text-anchor="middle">{escape(y_label)}</text>'
    )
    out.append(
        f'<line class="target" x1="{L}" x2="{L + pw}" y1="{sy(target):.1f}" '
        f'y2="{sy(target):.1f}" stroke-width="1.5" stroke-dasharray="5 4"/>'
        f'<text class="t2" x="{L + pw + 6}" y="{sy(target) + 4:.1f}" '
        f'font-size="11">{escape(target_label)}</text>'
    )
    for k, s in enumerate(series, start=1):
        ordered = sorted(s.points, key=lambda p: p.x)
        if s.band and all(p.lo is not None and p.hi is not None for p in ordered):
            upper = " ".join(f"{sx(p.x):.1f},{sy(p.hi or 0):.1f}" for p in ordered)
            lower = " ".join(f"{sx(p.x):.1f},{sy(p.lo or 0):.1f}" for p in reversed(ordered))
            out.append(f'<polygon class="band" points="{upper} {lower}"/>')
        path = " ".join(f"{sx(p.x):.1f},{sy(p.y):.1f}" for p in ordered)
        out.append(f'<polyline class="s{k}" points="{path}" fill="none" stroke-width="2"/>')
        for p in ordered:
            tip = (
                f"{s.name} · threshold {p.label} · coverage {100 * p.x:.1f}% · "
                f"error {100 * p.y:.1f}%"
            )
            out.append(
                f'<circle class="s{k}" cx="{sx(p.x):.1f}" cy="{sy(p.y):.1f}" r="3">'
                f"<title>{escape(tip)}</title></circle>"
            )
        last = ordered[-1]
        out.append(
            f'<text class="t1" x="{L + pw + 6}" y="{sy(last.y) + 4 + 13 * (k - 1):.1f}" '
            f'font-size="11">{escape(s.name)}</text>'
        )
    if chosen is not None:
        out.append(
            f'<circle class="s1 ring" cx="{sx(chosen.x):.1f}" cy="{sy(chosen.y):.1f}" r="6" '
            'stroke-width="2"/>'
            f'<text class="t1" x="{sx(chosen.x) + 9:.1f}" y="{sy(chosen.y) - 9:.1f}" '
            f'font-size="11">{escape(chosen_label)}</text>'
        )
    # Legend (identity is never color-alone: names are also direct labels).
    lx = L
    for k, s in enumerate(series, start=1):
        out.append(
            f'<line class="s{k}" x1="{lx}" x2="{lx + 16}" y1="36" y2="36" stroke-width="2"/>'
            f'<text class="t2" x="{lx + 20}" y="40" font-size="11">{escape(s.name)}</text>'
        )
        lx += 30 + 7 * len(s.name)
    out.append("</svg>")
    return "\n".join(out) + "\n"
