"""インラインSVGの小さなグラフ.

色はテンプレート側のCSS変数 (--series-1 など) を参照するので、
ダークモードの切り替えもCSSだけで済む。
"""

from __future__ import annotations

import json
from html import escape
from typing import Any, Sequence


def sparkline(values: Sequence[float | None], *, width: int = 120, height: int = 32, lo: float = 20, hi: float = 80) -> str:
    """古い→新しい順の値を折れ線で描く (スピード指数用). 最後の点に印."""
    pts = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(pts) < 2:
        return ""
    n = len(values)
    pad = 5

    def x(i: int) -> float:
        return pad + i * (width - 2 * pad) / max(n - 1, 1)

    def y(v: float) -> float:
        v = min(max(v, lo), hi)
        return height - pad - (v - lo) * (height - 2 * pad) / (hi - lo)

    path = " ".join(f"{'M' if k == 0 else 'L'}{x(i):.1f},{y(v):.1f}" for k, (i, v) in enumerate(pts))
    mid = y(50)
    li, lv = pts[-1]
    label = escape(" → ".join(f"{v:.0f}" for _, v in pts))
    return (
        f'<svg class="spark" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="スピード指数の推移 {label}">'
        f'<title>スピード指数 {label}</title>'
        f'<line x1="{pad}" x2="{width - pad}" y1="{mid:.1f}" y2="{mid:.1f}" class="grid"/>'
        f'<path d="{path}" class="line"/>'
        f'<circle cx="{x(li):.1f}" cy="{y(lv):.1f}" r="4" class="dot"/>'
        "</svg>"
    )


def line_chart(
    series: list[dict[str, Any]],
    *,
    x_labels: Sequence[str] | None = None,
    width: int = 720,
    height: int = 260,
    y_format: str = "yen",
    chart_id: str = "chart",
) -> str:
    """複数系列の折れ線 (同じ単位のものだけ). series: [{name, values, slot}]

    マウス位置に最も近いXを縦線で示し、全系列の値をツールチップに出す。
    """
    series = [s for s in series if s["values"]]
    if not series:
        return ""
    n = max(len(s["values"]) for s in series)
    all_vals = [v for s in series for v in s["values"]]
    lo, hi = min(all_vals + [0]), max(all_vals + [0])
    if hi == lo:
        hi = lo + 1
    span = hi - lo
    lo, hi = lo - span * 0.05, hi + span * 0.05
    left, right, top, bottom = 64, 16, 12, 28
    pw, ph = width - left - right, height - top - bottom

    def x(i: float) -> float:
        return left + i * pw / max(n - 1, 1)

    def y(v: float) -> float:
        return top + (hi - v) * ph / (hi - lo)

    ticks = _nice_ticks(lo, hi, 5)
    parts = [
        f'<svg class="chart" id="{chart_id}" viewBox="0 0 {width} {height}" role="img" aria-label="累積損益の推移">'
    ]
    for t in ticks:
        cls = "axis" if t == 0 else "grid"
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="{cls}"/>')
        parts.append(f'<text x="{left - 8}" y="{y(t) + 4:.1f}" class="tick" text-anchor="end">{_fmt(t, y_format)}</text>')
    if x_labels:
        step = max(1, len(x_labels) // 6)
        for i in range(0, len(x_labels), step):
            parts.append(f'<text x="{x(i):.1f}" y="{height - 8}" class="tick" text-anchor="middle">{escape(str(x_labels[i]))}</text>')
    for s in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(v):.1f}" for i, v in enumerate(s["values"]))
        parts.append(f'<path d="{d}" class="line" style="stroke: var(--series-{s.get("slot", 1)})"/>')
        last = len(s["values"]) - 1
        parts.append(
            f'<circle cx="{x(last):.1f}" cy="{y(s["values"][-1]):.1f}" r="4" class="dot" '
            f'style="fill: var(--series-{s.get("slot", 1)})"/>'
        )
    parts.append(f'<line class="crosshair" x1="0" x2="0" y1="{top}" y2="{top + ph}" visibility="hidden"/>')
    parts.append(f'<rect class="hit" x="{left}" y="{top}" width="{pw}" height="{ph}" fill="transparent"/>')
    parts.append("</svg>")
    payload = {
        "n": n, "left": left, "pw": pw, "width": width, "format": y_format,
        "labels": list(x_labels) if x_labels else None,
        "series": [{"name": s["name"], "slot": s.get("slot", 1), "values": s["values"]} for s in series],
    }
    data = escape(json.dumps(payload, ensure_ascii=False), quote=True)
    return f'<div class="chart-wrap" data-chart="{data}">{"".join(parts)}<div class="tooltip" hidden></div></div>'


def _fmt(v: float, kind: str) -> str:
    if kind == "yen":
        a = abs(v)
        sign = "-" if v < 0 else ""
        if a >= 10_000:
            return f"{sign}{a / 10_000:.0f}万"
        if a >= 1000:
            return f"{sign}{a / 1000:.0f}千"
        return f"{v:.0f}"
    if kind == "pct":
        return f"{v:.0%}"
    return f"{v:g}"


def _nice_ticks(lo: float, hi: float, count: int) -> list[float]:
    import math

    raw = (hi - lo) / max(count, 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag)
    start = math.ceil(lo / step) * step
    ticks = []
    t = start
    while t <= hi + 1e-9:
        ticks.append(round(t, 10))
        t += step
    return ticks


# 折れ線のツールチップ (値は textContent で入れる)
CHART_JS = """
document.querySelectorAll('.chart-wrap').forEach(function (wrap) {
  var cfg = JSON.parse(wrap.dataset.chart);
  var svg = wrap.querySelector('svg');
  var hit = svg.querySelector('.hit');
  var cross = svg.querySelector('.crosshair');
  var tip = wrap.querySelector('.tooltip');
  function fmt(v) {
    if (cfg.format === 'yen') return (v >= 0 ? '+' : '') + Math.round(v).toLocaleString('ja-JP') + '円';
    if (cfg.format === 'pct') return (v * 100).toFixed(1) + '%';
    return String(v);
  }
  function show(evt) {
    var box = svg.getBoundingClientRect();
    var sx = cfg.width / box.width;
    var px = (evt.clientX - box.left) * sx;
    var i = Math.round((px - cfg.left) / cfg.pw * (cfg.n - 1));
    i = Math.max(0, Math.min(cfg.n - 1, i));
    var cx = cfg.left + i * cfg.pw / Math.max(cfg.n - 1, 1);
    cross.setAttribute('x1', cx); cross.setAttribute('x2', cx);
    cross.setAttribute('visibility', 'visible');
    tip.replaceChildren();
    var head = document.createElement('div');
    head.className = 'tip-head';
    head.textContent = cfg.labels ? cfg.labels[i] : (i + 1) + '件目';
    tip.appendChild(head);
    cfg.series.forEach(function (s) {
      if (i >= s.values.length) return;
      var row = document.createElement('div'); row.className = 'tip-row';
      var key = document.createElement('span'); key.className = 'tip-key';
      key.style.background = 'var(--series-' + s.slot + ')';
      var val = document.createElement('strong'); val.textContent = fmt(s.values[i]);
      var name = document.createElement('span'); name.className = 'tip-name'; name.textContent = s.name;
      row.append(key, val, name); tip.appendChild(row);
    });
    tip.hidden = false;
    var left = (cx / sx) + 12;
    if (left + tip.offsetWidth > box.width) left = (cx / sx) - tip.offsetWidth - 12;
    tip.style.left = left + 'px';
    tip.style.top = '8px';
  }
  function hide() { cross.setAttribute('visibility', 'hidden'); tip.hidden = true; }
  hit.addEventListener('pointermove', show);
  hit.addEventListener('pointerleave', hide);
});
"""
