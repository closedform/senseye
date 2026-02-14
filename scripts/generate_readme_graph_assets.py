#!/usr/bin/env python3
"""Generate additional SVG assets used by README diagrams."""

from __future__ import annotations

from pathlib import Path

BG = "#0b1020"
PANEL = "#10192e"
PANEL_ALT = "#132244"
STROKE = "#7dd3fc"
TEXT = "#e5f3ff"
MUTED = "#9fb7d5"
ACCENT_WARM = "#f59e0b"
ACCENT_HOT = "#fb7185"
ACCENT_GOOD = "#34d399"


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<defs>",
        (
            f'<marker id="arrow" markerWidth="12" markerHeight="12" refX="9" refY="6" '
            'orient="auto"><path d="M0,0 L0,12 L10,6 z" '
            f'fill="{STROKE}"/></marker>'
        ),
        (
            '<filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">'
            '<feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#020617" '
            'flood-opacity="0.55"/></filter>'
        ),
        (
            '<linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0%" stop-color="#0d1428"/>'
            '<stop offset="100%" stop-color="#0b1020"/>'
            "</linearGradient>"
        ),
        "</defs>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#bgGrad)"/>',
        (
            f'<text x="{width // 2}" y="42" fill="{TEXT}" '
            'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
            'font-size="28" text-anchor="middle">'
            f"{title}</text>"
        ),
    ]


def _svg_footer() -> list[str]:
    return ["</svg>"]


def _box(
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    subtitle: str | None = None,
    fill: str = PANEL,
) -> str:
    parts = [
        (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
            f'fill="{fill}" stroke="{STROKE}" stroke-width="2" filter="url(#softShadow)"/>'
        ),
        (
            f'<text x="{x + w // 2}" y="{y + 36}" fill="{TEXT}" '
            'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
            'font-size="24" text-anchor="middle">'
            f"{title}</text>"
        ),
    ]
    if subtitle:
        parts.append(
            (
                f'<text x="{x + w // 2}" y="{y + 62}" fill="{MUTED}" '
                'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
                'font-size="17" text-anchor="middle">'
                f"{subtitle}</text>"
            ),
        )
    return "\n".join(parts)


def _line(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    color: str = STROKE,
    width: int = 3,
    dashed: bool = False,
    arrow: bool = False,
) -> str:
    dash = ' stroke-dasharray="8 8"' if dashed else ""
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{width}"{dash}{marker}/>'
    )


def _label(x: int, y: int, text: str, color: str = MUTED, size: int = 16) -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{color}" '
        'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
        f'font-size="{size}" text-anchor="middle">{text}</text>'
    )


def _label_left(x: int, y: int, text: str, color: str = MUTED, size: int = 16) -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{color}" '
        'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
        f'font-size="{size}" text-anchor="start">{text}</text>'
    )


def _node(x: int, y: int, label: str, fill: str = "#1e3a8a") -> str:
    return "\n".join(
        [
            (
                f'<circle cx="{x}" cy="{y}" r="15" fill="{fill}" '
                f'stroke="{STROKE}" stroke-width="2"/>'
            ),
            (
                f'<text x="{x}" y="{y + 6}" fill="{TEXT}" '
                'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
                'font-size="14" text-anchor="middle">'
                f"{label}</text>"
            ),
        ],
    )


def build_phase1() -> str:
    width, height = 1200, 560
    parts = _svg_header(width, height, "Phase 1: Passive RF Sensing")
    parts += [
        _box(80, 130, 320, 108, "Zone 1", "candidate", fill=PANEL_ALT),
        _box(800, 130, 320, 108, "Zone 2", "candidate", fill=PANEL_ALT),
        _box(440, 320, 320, 108, "Zone 3", "candidate", fill=PANEL),
        _line(400, 184, 800, 184, width=5),
        _line(240, 238, 560, 320, color=ACCENT_WARM, width=5),
        _line(960, 238, 640, 320, color="#cbd5e1", width=4, dashed=True),
        _label(600, 112, "Strong Link", color=TEXT, size=19),
        _label(250, 288, "High Attenuation", color=ACCENT_WARM, size=17),
        _label(956, 288, "Weak Evidence", color="#cbd5e1", size=17),
        '<rect x="70" y="408" width="380" height="118" rx="10" '
        f'fill="{PANEL}" stroke="{STROKE}" stroke-width="1.5"/>',
        _label_left(92, 436, "Legend", color=TEXT, size=16),
        _line(95, 456, 170, 456, width=4),
        _label_left(188, 462, "Strong RF link", color=TEXT, size=15),
        _line(95, 482, 170, 482, color=ACCENT_WARM, width=4),
        _label_left(188, 488, "Attenuation spike", color=ACCENT_WARM, size=15),
        _line(95, 508, 170, 508, color="#cbd5e1", width=3, dashed=True),
        _label_left(188, 514, "Weak evidence link", color="#cbd5e1", size=15),
    ]
    parts += _svg_footer()
    return "\n".join(parts)


def build_phase2() -> str:
    width, height = 1200, 620
    parts = _svg_header(width, height, "Phase 2: Acoustic Calibration")
    parts += [
        '<rect x="210" y="110" width="780" height="420" rx="16" '
        f'fill="{PANEL}" stroke="{STROKE}" stroke-width="2" filter="url(#softShadow)"/>',
        # room split lines
        _line(600, 110, 600, 260, color=STROKE, width=2),
        _line(600, 330, 600, 530, color=STROKE, width=2),
        _line(210, 320, 990, 320, color=STROKE, width=2),
        _label(380, 230, "kitchen", color=TEXT, size=26),
        _label(800, 230, "hallway", color=TEXT, size=26),
        _label(380, 445, "bedroom", color=TEXT, size=26),
        _label(800, 445, "living", color=TEXT, size=26),
        _label(600, 298, "door", color=ACCENT_GOOD, size=16),
        _node(330, 180, "n1"),
        _node(320, 410, "n2"),
        _node(830, 420, "n3"),
        _label(600, 575, "cm-accurate ranges -> MDS -> structured floorplan", size=17),
    ]
    parts += _svg_footer()
    return "\n".join(parts)


def build_phase3() -> str:
    width, height = 1200, 640
    parts = _svg_header(width, height, "Phase 3: Motion-Refined Overlay")
    parts += [
        '<rect x="210" y="110" width="780" height="420" rx="16" '
        f'fill="{PANEL}" stroke="{STROKE}" stroke-width="2" filter="url(#softShadow)"/>',
        _line(600, 110, 600, 260, color=STROKE, width=2),
        _line(600, 330, 600, 530, color=STROKE, width=2),
        _line(210, 320, 990, 320, color=STROKE, width=2),
        # heat overlays
        '<rect x="610" y="330" width="370" height="190" rx="10" '
        'fill="#e11d48" opacity="0.55" stroke="#fb7185" stroke-width="2"/>',
        '<rect x="220" y="120" width="370" height="190" rx="10" fill="#1d4ed8" opacity="0.28"/>',
        _label(380, 230, "kitchen (low)", color=TEXT, size=24),
        _label(800, 230, "hallway", color=TEXT, size=24),
        _label(380, 445, "bedroom (idle)", color=TEXT, size=24),
        _label(800, 408, "living", color=TEXT, size=28),
        _label(800, 438, "(high motion)", color=TEXT, size=22),
        _node(340, 215, "n1"),
        _node(330, 420, "n2"),
        _node(905, 420, "n3"),
        _node(320, 275, "P", fill="#0f766e"),
        _label(360, 284, "phone", color=TEXT, size=15),
        _node(740, 470, "W", fill="#7c3aed"),
        _label(790, 479, "watch", color=TEXT, size=15),
        '<rect x="782" y="342" width="180" height="32" rx="8" fill="#7f1d1d" '
        'stroke="#fb7185" stroke-width="1.5"/>',
        _label(872, 364, "HIGH MOTION", color="#ffd6de", size=14),
        '<rect x="230" y="128" width="160" height="28" rx="8" fill="#1e3a8a" opacity="0.85"/>',
        _label(310, 147, "LOW MOTION", color="#dbeafe", size=13),
        _label(600, 586, "motion: living high, kitchen low", size=17),
    ]
    parts += _svg_footer()
    return "\n".join(parts)


def _network_panel(
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    nodes: dict[str, tuple[int, int]],
    edges: list[tuple[str, str]],
) -> str:
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{PANEL}" '
        f'stroke="{STROKE}" stroke-width="2" filter="url(#softShadow)"/>',
        _label(x + (w // 2), y + 30, title, color=TEXT, size=16),
    ]
    for a, b in edges:
        x1, y1 = nodes[a]
        x2, y2 = nodes[b]
        parts.append(_line(x1, y1, x2, y2, width=2, color="#3b82f6"))
    for name, (nx, ny) in nodes.items():
        parts.append(_node(nx, ny, name, fill=PANEL_ALT))
    return "\n".join(parts)


def build_connectivity() -> str:
    width, height = 1500, 700
    parts = _svg_header(width, height, "Node Connectivity Scaling")

    panel_w = 330
    panel_h = 560
    top = 90
    gap = 30

    p1x = 45
    p2x = p1x + panel_w + gap
    p3x = p2x + panel_w + gap
    p4x = p3x + panel_w + gap

    nodes1 = {
        "n1": (p1x + 165, top + 180),
        "r": (p1x + 165, top + 290),
        "b": (p1x + 165, top + 400),
    }
    edges1 = [("n1", "r"), ("r", "b")]

    nodes2 = {
        "n1": (p2x + 95, top + 210),
        "n2": (p2x + 235, top + 210),
        "n3": (p2x + 165, top + 345),
    }
    edges2 = [("n1", "n2"), ("n2", "n3"), ("n1", "n3")]

    nodes3 = {
        "n1": (p3x + 95, top + 180),
        "n2": (p3x + 235, top + 180),
        "n3": (p3x + 95, top + 360),
        "n4": (p3x + 235, top + 360),
    }
    edges3 = [
        ("n1", "n2"),
        ("n1", "n3"),
        ("n1", "n4"),
        ("n2", "n3"),
        ("n2", "n4"),
        ("n3", "n4"),
    ]

    nodes4 = {
        "1": (p4x + 75, top + 170),
        "2": (p4x + 165, top + 170),
        "3": (p4x + 255, top + 170),
        "4": (p4x + 75, top + 280),
        "5": (p4x + 165, top + 280),
        "6": (p4x + 255, top + 280),
        "7": (p4x + 75, top + 390),
        "8": (p4x + 165, top + 390),
        "9": (p4x + 255, top + 390),
    }
    # Dense local edges for readable "8+ style" panel.
    edges4 = [
        ("1", "2"), ("2", "3"),
        ("4", "5"), ("5", "6"),
        ("7", "8"), ("8", "9"),
        ("1", "4"), ("4", "7"),
        ("2", "5"), ("5", "8"),
        ("3", "6"), ("6", "9"),
        ("1", "5"), ("2", "4"), ("2", "6"), ("3", "5"),
        ("4", "8"), ("5", "7"), ("5", "9"), ("6", "8"),
    ]

    parts += [
        _network_panel(p1x, top, panel_w, panel_h, "1 node", nodes1, edges1),
        _network_panel(p2x, top, panel_w, panel_h, "2-3 nodes", nodes2, edges2),
        _network_panel(p3x, top, panel_w, panel_h, "4-5 nodes", nodes3, edges3),
        _network_panel(p4x, top, panel_w, panel_h, "8+ pattern", nodes4, edges4),
        _label(
            width // 2,
            678,
            "More fixed nodes -> more links -> better spatial resolution",
            size=18,
        ),
    ]

    parts += _svg_footer()
    return "\n".join(parts)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    assets = root / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    outputs = {
        "phase-1-rf-zones.svg": build_phase1(),
        "phase-2-acoustic-floorplan.svg": build_phase2(),
        "phase-3-motion-overlay.svg": build_phase3(),
        "node-connectivity-scaling.svg": build_connectivity(),
    }
    for name, svg in outputs.items():
        path = assets / name
        path.write_text(svg, encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
