"""Dynamic overlay: motion shading + device markers on rendered map."""

from __future__ import annotations

from rich.text import Text

from senseye.mapping.dynamic.state import WorldState
from senseye.ui.renderer import RenderedMap


# Motion intensity -> background style
def _motion_style(intensity: float) -> str | None:
    if intensity < 0.2:
        return None
    if intensity < 0.5:
        return "on dark_green"
    if intensity < 0.8:
        return "on yellow"
    return "on red"


# Signal type -> device marker color
_DEVICE_COLORS = {
    "ble": "cyan",
    "wifi": "blue",
    "acoustic": "magenta",
}

_DEVICE_MARKER = "\u25cf"  # ●


def compose(rendered: RenderedMap, state: WorldState) -> Text:
    """Compose the dynamic overlay onto the cached rendered map.

    Applies motion shading to room cells, places device markers,
    and colors node markers by online status. Returns styled rich.Text.
    """
    # Pre-compute motion styles per room
    room_styles: dict[str, str | None] = {}
    for room_name in rendered.room_cells:
        intensity = state.motion.zone_motion.get(room_name, 0.0)
        room_styles[room_name] = _motion_style(intensity)

    # Build cell -> style lookup for motion shading
    cell_style: dict[tuple[int, int], str] = {}
    for room_name, cells in rendered.room_cells.items():
        style = room_styles[room_name]
        if style is not None:
            for cell in cells:
                cell_style[cell] = style

    # Build cell -> (char, style) overrides for devices
    device_overrides: dict[tuple[int, int], tuple[str, str]] = {}
    for device in state.devices.values():
        if device.position is None:
            continue
        g = rendered.world_to_grid(*device.position)
        if g is None:
            continue
        color = _DEVICE_COLORS.get(device.signal_type, "white")
        device_overrides[g] = (_DEVICE_MARKER, color)

    # Build cell -> style for node markers (online/offline coloring)
    node_marker_styles: dict[tuple[int, int], str] = {}
    if state.floorplan is not None:
        for node_id, pos in state.floorplan.node_positions.items():
            g = rendered.world_to_grid(*pos)
            if g is None:
                continue
            node_info = state.nodes.get(node_id)
            if node_info is not None:
                node_marker_styles[g] = "green" if node_info.online else "red"
            else:
                node_marker_styles[g] = "dim"

    # Assemble into rich.Text line by line
    text = Text()
    for gy in range(rendered.height):
        if gy > 0:
            text.append("\n")
        for gx in range(rendered.width):
            cell = (gx, gy)

            # Device override takes priority
            if cell in device_overrides:
                char, color = device_overrides[cell]
                bg = cell_style.get(cell, "")
                style = f"bold {color}"
                if bg:
                    style += f" {bg}"
                text.append(char, style)
                continue

            char = rendered.grid[gy][gx]

            # Node marker coloring
            if cell in node_marker_styles:
                color = node_marker_styles[cell]
                bg = cell_style.get(cell, "")
                style = f"bold {color}"
                if bg:
                    style += f" {bg}"
                text.append(char, style)
                continue

            # Motion shading on room cells
            bg = cell_style.get(cell)
            if bg:
                text.append(char, bg)
            else:
                text.append(char)

    return text
