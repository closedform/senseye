#!/usr/bin/env python3
"""Generate a polished SVG for the README node pipeline diagram."""

from __future__ import annotations

from pathlib import Path

CANVAS_W = 1200
CANVAS_H = 1400

BG = "#0b1020"
PANEL = "#10192e"
PANEL_ALT = "#132244"
STROKE = "#7dd3fc"
TEXT = "#e5f3ff"
MUTED = "#9fb7d5"


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
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" '
            f'fill="{fill}" stroke="{STROKE}" stroke-width="2"/>'
        ),
        (
            f'<text x="{x + (w // 2)}" y="{y + 36}" fill="{TEXT}" '
            'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
            'font-size="24" text-anchor="middle">'
            f"{title}</text>"
        ),
    ]

    if subtitle:
        parts.append(
            (
                f'<text x="{x + (w // 2)}" y="{y + 64}" fill="{MUTED}" '
                'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
                'font-size="18" text-anchor="middle">'
                f"{subtitle}</text>"
            ),
        )

    return "\n".join(parts)


def _arrow_line(x1: int, y1: int, x2: int, y2: int, width: int = 3) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{STROKE}" stroke-width="{width}" marker-end="url(#arrow)"/>'
    )


def _line(x1: int, y1: int, x2: int, y2: int, width: int = 3) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{STROKE}" stroke-width="{width}"/>'
    )


def build_svg() -> str:
    x_center = CANVAS_W // 2
    stack_x = 320
    stack_w = 560
    stack_h = 86

    signals_y = 110
    scan_y = 230
    kalman_y = 350
    infer_y = 470
    gossip_y = 590
    consensus_y = 710

    trilat_x = 200
    tomo_x = 700
    branch_y = 860
    branch_w = 300
    branch_h = 86

    world_x = 350
    world_y = 1030
    world_w = 500
    world_h = 96

    dash_x = 420
    dash_y = 1210
    dash_w = 360
    dash_h = 86

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" '
            f'height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}">'
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
        "</defs>",
        f'<rect x="0" y="0" width="{CANVAS_W}" height="{CANVAS_H}" fill="{BG}"/>',
        (
            f'<text x="{x_center}" y="58" fill="{TEXT}" '
            'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
            'font-size="36" text-anchor="middle">'
            "Senseye Node Pipeline</text>"
        ),
        (
            f'<text x="{x_center}" y="88" fill="{MUTED}" '
            'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
            'font-size="18" text-anchor="middle">'
            "scan → filter → infer → share → fuse → render</text>"
        ),
        '<g filter="url(#softShadow)">',
        _box(
            stack_x,
            signals_y,
            stack_w,
            stack_h,
            "Signals",
            "WiFi / BLE / Acoustic",
            fill=PANEL_ALT,
        ),
        _box(stack_x, scan_y, stack_w, stack_h, "Scan"),
        _box(stack_x, kalman_y, stack_w, stack_h, "Adaptive Kalman", "Per link"),
        _box(stack_x, infer_y, stack_w, stack_h, "Infer", "Links / Devices / Zones + Confidence"),
        _box(
            stack_x,
            gossip_y,
            stack_w,
            stack_h,
            "Gossip Mesh",
            "mDNS + TCP, sequence dedup, hop TTL",
            fill=PANEL_ALT,
        ),
        _box(stack_x, consensus_y, stack_w, stack_h, "Consensus Fusion"),
        _box(trilat_x, branch_y, branch_w, branch_h, "Trilateration"),
        _box(tomo_x, branch_y, branch_w, branch_h, "Tomography"),
        _box(world_x, world_y, world_w, world_h, "World State", "Static map + dynamic overlay"),
        _box(dash_x, dash_y, dash_w, dash_h, "Dashboard", fill=PANEL_ALT),
        "</g>",
        # vertical stack arrows
        _arrow_line(x_center, signals_y + stack_h, x_center, scan_y),
        _arrow_line(x_center, scan_y + stack_h, x_center, kalman_y),
        _arrow_line(x_center, kalman_y + stack_h, x_center, infer_y),
        _arrow_line(x_center, infer_y + stack_h, x_center, gossip_y),
        _arrow_line(x_center, gossip_y + stack_h, x_center, consensus_y),
        # consensus branch split
        _line(x_center, consensus_y + stack_h, x_center, 820),
        _line(x_center, 820, trilat_x + (branch_w // 2), 820),
        _line(x_center, 820, tomo_x + (branch_w // 2), 820),
        _arrow_line(trilat_x + (branch_w // 2), 820, trilat_x + (branch_w // 2), branch_y),
        _arrow_line(tomo_x + (branch_w // 2), 820, tomo_x + (branch_w // 2), branch_y),
        # merge to world state
        _arrow_line(trilat_x + (branch_w // 2), branch_y + branch_h, x_center, world_y),
        _arrow_line(tomo_x + (branch_w // 2), branch_y + branch_h, x_center, world_y),
        _arrow_line(x_center, world_y + world_h, x_center, dash_y),
        "</svg>",
    ]
    return "\n".join(parts)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    assets = root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    output = assets / "node-pipeline.svg"
    output.write_text(build_svg(), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
