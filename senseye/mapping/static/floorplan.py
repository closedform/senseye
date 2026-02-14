"""Combined static map: node positions + walls + rooms, serializable to disk."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from senseye.mapping.static.topology import Connection, Room, RoomGraph
from senseye.mapping.static.walls import WallSegment


DEFAULT_PATH = Path.home() / ".senseye" / "floorplan.json"


@dataclass
class FloorPlan:
    node_positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    wall_segments: list[WallSegment] = field(default_factory=list)
    rooms: RoomGraph = field(default_factory=RoomGraph)
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    labels: dict[str, str] = field(default_factory=dict)
    calibrated_at: float = field(default_factory=time.time)


def _wall_to_dict(w: WallSegment) -> dict:
    return {
        "start": list(w.start),
        "end": list(w.end),
        "attenuation_db": w.attenuation_db,
        "material": w.material,
    }


def _wall_from_dict(d: dict) -> WallSegment:
    return WallSegment(
        start=tuple(d["start"]),
        end=tuple(d["end"]),
        attenuation_db=d["attenuation_db"],
        material=d["material"],
    )


def _room_to_dict(r: Room) -> dict:
    return {
        "name": r.name,
        "center": list(r.center) if r.center is not None else None,
        "node_ids": r.node_ids,
    }


def _room_from_dict(d: dict) -> Room:
    return Room(
        name=d["name"],
        center=tuple(d["center"]) if d["center"] is not None else None,
        node_ids=d["node_ids"],
    )


def _connection_to_dict(c: Connection) -> dict:
    return {
        "room_a": c.room_a,
        "room_b": c.room_b,
        "doorway_position": list(c.doorway_position) if c.doorway_position is not None else None,
    }


def _connection_from_dict(d: dict) -> Connection:
    return Connection(
        room_a=d["room_a"],
        room_b=d["room_b"],
        doorway_position=tuple(d["doorway_position"]) if d["doorway_position"] is not None else None,
    )


def _plan_to_dict(plan: FloorPlan) -> dict:
    return {
        "node_positions": {k: list(v) for k, v in plan.node_positions.items()},
        "wall_segments": [_wall_to_dict(w) for w in plan.wall_segments],
        "rooms": {
            "rooms": [_room_to_dict(r) for r in plan.rooms.rooms],
            "connections": [_connection_to_dict(c) for c in plan.rooms.connections],
        },
        "bounds": list(plan.bounds),
        "labels": plan.labels,
        "calibrated_at": plan.calibrated_at,
    }


def _plan_from_dict(d: dict) -> FloorPlan:
    rooms_data = d.get("rooms", {"rooms": [], "connections": []})
    return FloorPlan(
        node_positions={k: tuple(v) for k, v in d["node_positions"].items()},
        wall_segments=[_wall_from_dict(w) for w in d.get("wall_segments", [])],
        rooms=RoomGraph(
            rooms=[_room_from_dict(r) for r in rooms_data.get("rooms", [])],
            connections=[_connection_from_dict(c) for c in rooms_data.get("connections", [])],
        ),
        bounds=tuple(d.get("bounds", [0.0, 0.0, 0.0, 0.0])),
        labels=d.get("labels", {}),
        calibrated_at=d.get("calibrated_at", 0.0),
    )


def save(plan: FloorPlan, path: Path = DEFAULT_PATH) -> None:
    """Serialize FloorPlan to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _plan_to_dict(plan)
    path.write_text(json.dumps(data, indent=2) + "\n")


def load(path: Path = DEFAULT_PATH) -> FloorPlan:
    """Deserialize FloorPlan from JSON."""
    data = json.loads(path.read_text())
    return _plan_from_dict(data)


def needs_update(
    plan: FloorPlan,
    current_distances: np.ndarray,
    threshold: float = 2.0,
) -> bool:
    """Check if any pairwise distance has shifted beyond threshold.

    current_distances: NxN distance matrix, node order matches
    sorted(plan.node_positions.keys()).
    """
    node_ids = sorted(plan.node_positions.keys())
    n = len(node_ids)

    if n < 2:
        return False

    if current_distances.shape != (n, n):
        return True  # shape mismatch means topology changed

    for i in range(n):
        for j in range(i + 1, n):
            pi = np.array(plan.node_positions[node_ids[i]])
            pj = np.array(plan.node_positions[node_ids[j]])
            plan_dist = float(np.linalg.norm(pi - pj))
            current_dist = current_distances[i, j]
            if abs(plan_dist - current_dist) > threshold:
                return True

    return False
