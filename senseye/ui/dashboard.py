"""Live terminal dashboard using rich."""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from senseye.mapping.dynamic.state import WorldState
from senseye.ui.overlay import compose
from senseye.ui.renderer import RenderedMap, render_floorplan, render_no_map


def _header(state: WorldState) -> Panel:
    """Build the header panel: title + time + node count."""
    now = time.strftime("%H:%M:%S")
    online = sum(1 for n in state.nodes.values() if n.online)
    total = len(state.nodes)
    title = Text()
    title.append("senseye", "bold white")
    title.append(f"  {now}  ", "dim")
    title.append(f"{online}/{total} nodes", "green" if online > 0 else "red")
    return Panel(title, style="bold", height=3)


def _motion_bar(intensity: float, bar_width: int = 4) -> Text:
    """Render a small bar for motion intensity."""
    filled = int(round(intensity * bar_width))
    text = Text()
    if intensity >= 0.8:
        color = "red"
    elif intensity >= 0.5:
        color = "yellow"
    elif intensity >= 0.2:
        color = "green"
    else:
        color = "dim"
    text.append("\u2588" * filled, color)
    text.append("\u2591" * (bar_width - filled), "dim")
    return text


def _footer(state: WorldState) -> Panel:
    """Build the footer panel: motion bars + device list + map age."""
    content = Text()

    # Motion summary bars
    zones = sorted(state.motion.zone_motion.keys())
    if zones:
        for i, zone in enumerate(zones):
            if i > 0:
                content.append("  ")
            intensity = state.motion.zone_motion[zone]
            content.append(f"{zone} ")
            content.append_text(_motion_bar(intensity))
        content.append("\n")

    # Device list
    active_devices = [
        d for d in state.devices.values()
        if time.time() - d.last_seen < 60.0
    ]
    if active_devices:
        for i, dev in enumerate(active_devices[:6]):  # cap at 6 to avoid overflow
            if i > 0:
                content.append("  ")
            name = dev.name or dev.device_id[:8]
            location = dev.zone or "?"
            content.append(f"{name}", "cyan")
            content.append(f"\u2192{location}", "dim")
        content.append("\n")

    # Map age
    if state.floorplan is not None:
        age_min = int(state.map_age / 60)
        if age_min < 60:
            age_str = f"{age_min}m"
        else:
            age_str = f"{age_min // 60}h{age_min % 60}m"
        content.append(f"map age: {age_str}", "dim")
    else:
        content.append("no map", "dim red")

    return Panel(content, title="status", title_align="left", height=5)


def _status_screen(state: WorldState, width: int = 60) -> Text:
    """Show status when no floorplan is loaded."""
    text = Text(justify="center")
    text.append("\n\n")
    text.append("Scanning...\n\n", "bold yellow")
    online = sum(1 for n in state.nodes.values() if n.online)
    device_count = len(state.devices)
    text.append(f"{online} nodes connected", "green" if online > 0 else "dim")
    text.append(", ")
    text.append(f"{device_count} devices visible", "cyan" if device_count > 0 else "dim")
    text.append("\n\n")
    text.append("Run ", "dim")
    text.append("senseye calibrate", "bold white")
    text.append(" to build map.", "dim")
    text.append("\n")
    return text


class Dashboard:
    def __init__(self, config: object | None = None) -> None:
        self._config = config
        self._rendered: RenderedMap | None = None
        self._last_floorplan_id: int | None = None  # id() of last rendered FloorPlan
        self._console = Console()

    def _map_size(self) -> tuple[int, int]:
        """Compute map grid size based on current terminal dimensions."""
        term_w = self._console.width or 80
        term_h = self._console.height or 24
        # Panel borders take 2 chars on each side, header 3, footer 5, panel top/bottom borders 2
        map_w = min(term_w - 4, 76)
        map_h = max(term_h - 3 - 5 - 4, 8)  # header + footer + panel borders
        return (map_w, map_h)

    def _get_rendered(self, state: WorldState) -> RenderedMap:
        """Get or re-render the cached map. Only re-renders when floorplan changes."""
        map_w, map_h = self._map_size()

        if state.floorplan is None:
            if self._last_floorplan_id is not None or self._rendered is None:
                self._rendered = render_no_map(map_w, map_h)
                self._last_floorplan_id = None
            return self._rendered

        plan_id = id(state.floorplan)
        if plan_id != self._last_floorplan_id or self._rendered is None:
            self._rendered = render_floorplan(state.floorplan, map_w, map_h)
            self._last_floorplan_id = plan_id
        return self._rendered

    def _build_layout(self, state: WorldState) -> Layout:
        """Build the full dashboard layout for one frame."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=5),
        )

        layout["header"].update(_header(state))

        if state.floorplan is None:
            layout["body"].update(Panel(_status_screen(state), title="map", border_style="dim"))
        else:
            rendered = self._get_rendered(state)
            composed = compose(rendered, state)
            layout["body"].update(Panel(composed, title="map", border_style="blue"))

        layout["footer"].update(_footer(state))
        return layout

    def update(self, state: WorldState) -> Layout:
        """Update the display with new state. Returns the layout for rendering."""
        return self._build_layout(state)

    async def run(self, state_stream: AsyncIterator[WorldState]) -> None:
        """Consume a WorldState async iterator and update the live display."""
        # Start with an empty state
        current_state = WorldState()

        with Live(self._build_layout(current_state), refresh_per_second=2) as live:
            quit_event = asyncio.Event()

            async def _check_quit():
                """Check for 'q' key to quit. Best-effort — depends on terminal."""
                loop = asyncio.get_event_loop()
                try:
                    import sys
                    import termios
                    import tty
                    fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(fd)
                    try:
                        tty.setcbreak(fd)
                        while not quit_event.is_set():
                            ch = await loop.run_in_executor(None, sys.stdin.read, 1)
                            if ch == "q":
                                quit_event.set()
                                return
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except (ImportError, OSError, termios.error):
                    # Not a terminal or not Unix — skip keyboard handling
                    await quit_event.wait()

            quit_task = asyncio.create_task(_check_quit())

            try:
                async for state in state_stream:
                    if quit_event.is_set():
                        break
                    current_state = state
                    live.update(self._build_layout(current_state))
            finally:
                quit_event.set()
                quit_task.cancel()
                try:
                    await quit_task
                except asyncio.CancelledError:
                    pass
