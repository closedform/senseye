"""Static floor plan -> character grid renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from senseye.mapping.static.floorplan import FloorPlan


@dataclass
class RenderedMap:
    grid: list[list[str]]
    width: int
    height: int
    world_to_grid: Callable[[float, float], tuple[int, int] | None]
    room_cells: dict[str, list[tuple[int, int]]] = field(default_factory=dict)


# Box-drawing characters
_H = "\u2500"      # ─
_V = "\u2502"      # │
_TL = "\u250c"     # ┌
_TR = "\u2510"     # ┐
_BL = "\u2514"     # └
_BR = "\u2518"     # ┘
_T_DOWN = "\u252c"  # ┬
_T_UP = "\u2534"    # ┴
_T_RIGHT = "\u251c" # ├
_T_LEFT = "\u2524"  # ┤
_CROSS = "\u253c"   # ┼

_NODE_FIXED = "\u25c8"  # ◈
_NODE_ROUTER = "\u25c6" # ◆


def _make_grid(width: int, height: int, fill: str = " ") -> list[list[str]]:
    return [[fill] * width for _ in range(height)]


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Integer Bresenham line from (x0,y0) to (x1,y1)."""
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


def render_floorplan(plan: FloorPlan, width: int = 60, height: int = 30) -> RenderedMap:
    """Render a FloorPlan into a character grid.

    Draws walls with box-drawing characters, labels rooms at their centers,
    and marks fixed nodes. Returns a RenderedMap with coordinate mapping.
    """
    grid = _make_grid(width, height)

    x_min, y_min, x_max, y_max = plan.bounds
    w_range = x_max - x_min
    h_range = y_max - y_min

    if w_range < 0.01:
        w_range = 1.0
    if h_range < 0.01:
        h_range = 1.0

    # Leave 1-cell border
    usable_w = width - 2
    usable_h = height - 2
    if usable_w < 1:
        usable_w = 1
    if usable_h < 1:
        usable_h = 1

    scale_x_max = usable_w / w_range
    scale_y_max = usable_h / h_range  # rows per meter

    # Standard terminal char is ~2x taller than wide.
    # To preserve aspect ratio: rows_per_meter should be ~0.5 * chars_per_meter
    # So we want scale_y = 0.5 * scale_x
    # We must satisfy: scale_x <= scale_x_max  AND  0.5 * scale_x <= scale_y_max
    # => scale_x <= scale_x_max  AND  scale_x <= 2 * scale_y_max

    scale_x = min(scale_x_max, 2.0 * scale_y_max)
    scale_y = 0.5 * scale_x

    def world_to_grid(wx: float, wy: float) -> tuple[int, int] | None:
        gx = int(round((wx - x_min) * scale_x)) + 1
        gy = int(round((wy - y_min) * scale_y)) + 1
        if 0 <= gx < width and 0 <= gy < height:
            return (gx, gy)
        return None

    # --- Draw walls ---
    wall_cells: set[tuple[int, int]] = set()
    for wall in plan.wall_segments:
        start_g = world_to_grid(*wall.start)
        end_g = world_to_grid(*wall.end)
        if start_g is None or end_g is None:
            continue
        points = _bresenham(start_g[0], start_g[1], end_g[0], end_g[1])
        for gx, gy in points:
            if 0 <= gx < width and 0 <= gy < height:
                wall_cells.add((gx, gy))

    # Choose wall characters based on neighbor connectivity
    for gx, gy in wall_cells:
        has_left = (gx - 1, gy) in wall_cells
        has_right = (gx + 1, gy) in wall_cells
        has_up = (gx, gy - 1) in wall_cells
        has_down = (gx, gy + 1) in wall_cells

        h_count = int(has_left) + int(has_right)
        v_count = int(has_up) + int(has_down)
        total = h_count + v_count

        if total == 0:
            ch = "\u00b7"  # ·
        elif v_count == 0:
            ch = _H
        elif h_count == 0:
            ch = _V
        elif total == 4:
            ch = _CROSS
        elif has_left and has_right and has_down and not has_up:
            ch = _T_DOWN
        elif has_left and has_right and has_up and not has_down:
            ch = _T_UP
        elif has_up and has_down and has_right and not has_left:
            ch = _T_RIGHT
        elif has_up and has_down and has_left and not has_right:
            ch = _T_LEFT
        elif has_right and has_down:
            ch = _TL
        elif has_left and has_down:
            ch = _TR
        elif has_right and has_up:
            ch = _BL
        elif has_left and has_up:
            ch = _BR
        else:
            ch = _H if h_count >= v_count else _V

        grid[gy][gx] = ch

    # --- Clear doorway gaps ---
    for conn in plan.rooms.connections:
        if conn.doorway_position is None:
            continue
        g = world_to_grid(*conn.doorway_position)
        if g is None:
            continue
        gx, gy = g
        radius = max(1, int(0.8 * min(scale_x, scale_y) / 2))
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) in wall_cells:
                    grid[ny][nx] = " "
                    wall_cells.discard((nx, ny))

    # --- Build room_cells by flood-filling from room centers ---
    # Assign each non-wall cell to the nearest room center
    room_cells: dict[str, list[tuple[int, int]]] = {}
    rooms_with_centers: list[tuple[str, int, int]] = []
    for room in plan.rooms.rooms:
        if room.center is None:
            continue
        g = world_to_grid(*room.center)
        if g is None:
            continue
        rooms_with_centers.append((room.name, g[0], g[1]))
        room_cells[room.name] = []

    if rooms_with_centers:
        # For each non-wall cell, assign to nearest room center
        for gy_cell in range(height):
            for gx_cell in range(width):
                if (gx_cell, gy_cell) in wall_cells:
                    continue
                best_room = None
                best_dist = float("inf")
                for rname, rcx, rcy in rooms_with_centers:
                    d = (gx_cell - rcx) ** 2 + (gy_cell - rcy) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_room = rname
                if best_room is not None:
                    room_cells[best_room].append((gx_cell, gy_cell))

    # --- Label rooms ---
    for room in plan.rooms.rooms:
        if room.center is None:
            continue
        label = plan.labels.get(room.name, room.name)
        g = world_to_grid(*room.center)
        if g is None:
            continue
        gx, gy = g
        start_x = gx - len(label) // 2
        for i, ch in enumerate(label):
            cx = start_x + i
            if 0 <= cx < width and 0 <= gy < height and (cx, gy) not in wall_cells:
                grid[gy][cx] = ch

    # --- Place fixed nodes ---
    for node_id, pos in plan.node_positions.items():
        g = world_to_grid(*pos)
        if g is None:
            continue
        gx, gy = g
        # Check if this node is labeled as a router
        is_router = plan.labels.get(node_id, "").lower() in ("router", "ap")
        marker = _NODE_ROUTER if is_router else _NODE_FIXED
        grid[gy][gx] = marker

    return RenderedMap(
        grid=grid,
        width=width,
        height=height,
        world_to_grid=world_to_grid,
        room_cells=room_cells,
    )


def render_no_map(width: int = 60, height: int = 12) -> RenderedMap:
    """Render a placeholder when no floorplan is available."""
    grid = _make_grid(width, height)

    def world_to_grid(wx: float, wy: float) -> tuple[int, int] | None:
        return None

    return RenderedMap(
        grid=grid,
        width=width,
        height=height,
        world_to_grid=world_to_grid,
        room_cells={},
    )
