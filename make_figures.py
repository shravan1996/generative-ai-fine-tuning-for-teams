"""Draw the article's three figures as SVG from results/summary.json.

    python make_figures.py <output-folder>

Figures are 680 wide, transparent, and use the site's palette, so they
sit on the article's figure plate in light and dark mode.
"""

import json
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).parent
W = 680
INK, SOFT, FAINT, RULE = "#191F1D", "#3F4945", "#5C6964", "#D3DBD7"
GREEN, AMBER, AMBER_FILL, GREEN_FILL = "#1F6B4C", "#8F6019", "#D9A55B", "#1F6B4C"

STYLE = f"""  <style>
    text {{ stroke: none; }}
    .serif {{ font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif; }}
    .sans  {{ font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
    .h1 {{ font-size: 26px; font-weight: 600; fill: {INK}; }}
    .sub {{ font-size: 14.5px; fill: {SOFT}; }}
    .label {{ font-size: 11px; font-weight: 700; letter-spacing: 1.1px; fill: {FAINT}; }}
    .row {{ font-size: 14.5px; fill: {INK}; }}
    .rowb {{ font-size: 14.5px; font-weight: 600; fill: {INK}; }}
    .cell {{ font-size: 13.5px; fill: {SOFT}; }}
    .val {{ font-size: 14px; font-weight: 700; fill: {GREEN}; }}
    .warn {{ font-size: 14px; font-weight: 700; fill: {AMBER}; }}
    .num {{ font-size: 14px; fill: {SOFT}; }}
    .foot {{ font-size: 13.5px; fill: {SOFT}; }}
  </style>"""


def wrap(text, width_px, size=13.5):
    """Greedy word wrap using an average glyph width for the sans face."""
    per_line = max(10, int(width_px / (size * 0.53)))
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > per_line:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        lines.append(cur)
    return lines


class Svg:
    def __init__(self, title, desc):
        self.title, self.desc, self.parts, self.y = title, desc, [], 0

    def text(self, x, y, cls, s, anchor=None):
        a = f' text-anchor="{anchor}"' if anchor else ""
        self.parts.append(f'  <text class="{cls}" x="{x}" y="{y:.1f}"{a}>{escape(s)}</text>')

    def rule(self, y):
        self.parts.append(f'  <line x1="32" y1="{y:.1f}" x2="{W - 32}" y2="{y:.1f}" stroke="{RULE}" stroke-width="1"/>')

    def rect(self, x, y, w, h, fill):
        self.parts.append(f'  <rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 1.5):.1f}" height="{h}" fill="{fill}"/>')

    def head(self, subs):
        self.text(32, 44, "serif h1", self.title)
        y = 72
        for s in subs:
            self.text(32, y, "sans sub", s)
            y += 20
        self.rule(y)
        self.y = y + 28

    def save(self, path):
        h = int(self.y + 16)
        body = "\n".join(self.parts)
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" '
            f'role="img" aria-labelledby="title desc">\n'
            f'  <title id="title">{escape(self.title)}</title>\n'
            f'  <desc id="desc">{escape(self.desc)}</desc>\n{STYLE}\n{body}\n</svg>\n')


def text_table(title, subs, rows, left_w=200, foot=None):
    """Two-column table: a bold prompt on the left, wrapped detail on the right."""
    svg = Svg(title, " ".join(f"{a}: {b}" for a, b in rows) + (f" {foot}" if foot else ""))
    svg.head(subs)
    x2 = 32 + left_w + 16
    for left, right in rows:
        top = svg.y
        ll = wrap(left, left_w, 14.5)
        rl = wrap(right, W - 32 - x2, 13.5)
        for i, line in enumerate(ll):
            svg.text(32, top + i * 20, "sans rowb", line)
        for i, line in enumerate(rl):
            svg.text(x2, top + i * 19, "sans cell", line)
        svg.y = top + max(len(ll) * 20, len(rl) * 19) + 12
        svg.rule(svg.y - 6)
        svg.y += 22
    if foot:
        for line in wrap(foot, W - 64, 13.5):
            svg.text(32, svg.y, "sans foot", line)
            svg.y += 19
    return svg


def pct(x):
    return f"{100 * x:.1f}%"


def result_figure(s):
    d1, d90 = s["day1"], s["day90"]
    rows1 = [
        ("Qwen 0.8B, queue list, untrained", "qwen_listed_day1"),
        ("Qwen 0.8B + 8 labels a queue", "qwen_k8_day1"),
        ("Qwen 0.8B + 32 labels a queue", "qwen_k32_day1"),
        ("Qwen 0.8B + every label", "qwen_all_day1"),
        ("Claude Haiku 5.5, queue list", "haiku_day1"),
        ("Claude Sonnet 5.5, queue list", "sonnet_day1"),
        ("Claude Haiku 5.5, a line per queue †", "haiku_day1_described"),
        ("Claude Sonnet 5.5, a line per queue †", "sonnet_day1_described"),
    ]
    rows90 = [
        ("Claude Haiku 5.5, 77 names", "haiku_day90"),
        ("Claude Sonnet 5.5, 77 names", "sonnet_day90"),
        ("Day-one Qwen, not retrained", "qwen_all_day1_on_day90"),
        ("Qwen retrained, 8 labels a new queue", "qwen_retrain_day90"),
        ("Day-one Qwen + Sonnet on new queues*", "watch_sonnet"),
    ]
    watch = s.get("new_queue_watch_exploratory", {}).get("sonnet_day90")
    if watch:
        d90 = dict(d90)
        d90["watch_sonnet"] = {"old": {"accuracy": watch["accuracy_old"]}, "new": {"accuracy": watch["accuracy_new"]}}
    desc = "; ".join(f"{n}: {pct(d1[a]['accuracy'])}" for n, a in rows1 if a in d1)
    desc += ". Day ninety, old and new queues: " + "; ".join(
        f"{n}: {pct(d90[a]['old']['accuracy'])} old, {pct(d90[a]['new']['accuracy'])} new" for n, a in rows90 if a in d90)
    svg = Svg("Right queue, day one and day ninety", desc)
    svg.head(["BANKING77 test messages, routed once each. Day one: 70 queues, 2,800 messages.",
              "Day ninety: seven queues added, 280 more messages. Run 15 and 16 October 2026."])
    bar_x, bar_w = 300, 300
    svg.text(32, svg.y, "sans label", "DAY ONE, SHARE ROUTED TO THE RIGHT QUEUE")
    svg.y += 22
    for name, arm in rows1:
        if arm not in d1:
            continue
        a = d1[arm]["accuracy"]
        svg.text(32, svg.y + 14, "sans row", name)
        fill = GREEN_FILL if arm.startswith("qwen_k") or arm == "qwen_all_day1" else AMBER_FILL if arm == "qwen_listed_day1" else "#788781"
        svg.rect(bar_x, svg.y, bar_w * a, 18, fill)
        svg.text(bar_x + bar_w * a + 8, svg.y + 14, "sans val" if fill == GREEN_FILL else "sans num", pct(a))
        svg.y += 30
    svg.y += 8
    svg.rule(svg.y)
    svg.y += 28
    svg.text(32, svg.y, "sans label", "DAY NINETY")
    svg.text(470, svg.y, "sans label", "70 OLD", anchor="end")
    svg.text(600, svg.y, "sans label", "7 NEW", anchor="end")
    svg.y += 26
    for name, arm in rows90:
        if arm not in d90:
            continue
        old, new = d90[arm]["old"]["accuracy"], d90[arm]["new"]["accuracy"]
        svg.text(32, svg.y, "sans row", name)
        svg.text(470, svg.y, "sans num", pct(old), anchor="end")
        svg.text(600, svg.y, "sans warn" if new < 0.7 else "sans val", pct(new), anchor="end")
        svg.y += 28
    svg.y += 4
    svg.rule(svg.y)
    svg.y += 26
    return svg


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    s = json.loads((ROOT / "results" / "summary.json").read_text())
    figs = json.loads((ROOT / "figures.json").read_text()) if (ROOT / "figures.json").exists() else {}
    if "reading_rule" in figs:
        f = figs["reading_rule"]
        text_table(f["title"], f["subs"], f["rows"], f.get("left_w", 200), f.get("foot")).save(out / "reading-rule.svg")
    svg = result_figure(s)
    for line in figs.get("result_foot", []):
        svg.text(32, svg.y, "sans foot", line)
        svg.y += 19
    svg.save(out / "prototype-result.svg")
    if "working_default" in figs:
        f = figs["working_default"]
        text_table(f["title"], f["subs"], f["rows"], f.get("left_w", 200), f.get("foot")).save(out / "working-default.svg")
    print("wrote", sorted(p.name for p in out.glob("*.svg")))


if __name__ == "__main__":
    main()
