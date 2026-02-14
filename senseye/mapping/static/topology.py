"""Room connectivity from motion path traces and wall segments."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from senseye.mapping.static.walls import WallSegment


@dataclass
class Room:
    name: str
    center: tuple[float, float] | None = None
    node_ids: list[str] = field(default_factory=list)


@dataclass
class Connection:
    room_a: str
    room_b: str
    doorway_position: tuple[float, float] | None = None


@dataclass
class RoomGraph:
    rooms: list[Room] = field(default_factory=list)
    connections: list[Connection] = field(default_factory=list)


def _segments_intersect(
    p1: np.ndarray, p2: np.ndarray,
    p3: np.ndarray, p4: np.ndarray,
) -> bool:
    """Check if line segment p1-p2 intersects segment p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12:
        return False  # parallel
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


def _wall_between(
    pos_a: tuple[float, float],
    pos_b: tuple[float, float],
    wall_segments: list[WallSegment],
) -> bool:
    """Check if any wall segment intersects the line between two positions."""
    pa = np.array(pos_a)
    pb = np.array(pos_b)
    for wall in wall_segments:
        ws = np.array(wall.start)
        we = np.array(wall.end)
        if _segments_intersect(pa, pb, ws, we):
            return True
    return False


def infer_rooms_from_nodes(
    node_positions: dict[str, tuple[float, float]],
    wall_segments: list[WallSegment],
) -> RoomGraph:
    """Cluster nodes into rooms based on wall segments.

    Nodes with no wall between them are placed in the same room.
    Uses union-find for clustering.
    """
    node_ids = sorted(node_positions.keys())
    n = len(node_ids)

    if n == 0:
        return RoomGraph()

    # Union-find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # Merge nodes that have no wall between them
    for i in range(n):
        for j in range(i + 1, n):
            pos_i = node_positions[node_ids[i]]
            pos_j = node_positions[node_ids[j]]
            if not _wall_between(pos_i, pos_j, wall_segments):
                union(i, j)

    # Group by root
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    rooms: list[Room] = []
    for idx, (_, members) in enumerate(sorted(clusters.items())):
        member_ids = [node_ids[i] for i in members]
        positions = np.array([node_positions[nid] for nid in member_ids])
        center = positions.mean(axis=0)
        rooms.append(Room(
            name=f"room_{idx}",
            center=(float(center[0]), float(center[1])),
            node_ids=member_ids,
        ))

    # Connections: rooms that have adjacent nodes (with a wall between them
    # but close enough to suggest a doorway exists)
    connections: list[Connection] = []
    room_of_node: dict[str, str] = {}
    for room in rooms:
        for nid in room.node_ids:
            room_of_node[nid] = room.name

    # Find the closest cross-room node pair for each room pair
    best_pair: dict[tuple[str, str], tuple[float, int, int]] = {}  # pair -> (dist, i, j)
    for i in range(n):
        for j in range(i + 1, n):
            ra = room_of_node[node_ids[i]]
            rb = room_of_node[node_ids[j]]
            if ra == rb:
                continue
            pair = (min(ra, rb), max(ra, rb))
            pa = np.array(node_positions[node_ids[i]])
            pb = np.array(node_positions[node_ids[j]])
            dist = float(np.linalg.norm(pa - pb))
            if pair not in best_pair or dist < best_pair[pair][0]:
                best_pair[pair] = (dist, i, j)

    for pair, (_dist, i, j) in best_pair.items():
        pa = np.array(node_positions[node_ids[i]])
        pb = np.array(node_positions[node_ids[j]])
        midpoint = (pa + pb) / 2.0
        connections.append(Connection(
            room_a=pair[0],
            room_b=pair[1],
            doorway_position=(float(midpoint[0]), float(midpoint[1])),
        ))

    return RoomGraph(rooms=rooms, connections=connections)


# Minimum traversals to consider a zone pair as connected by a doorway
_DOORWAY_THRESHOLD = 3


def update_topology(
    graph: RoomGraph,
    motion_events: list[tuple[str, str, float]],
) -> RoomGraph:
    """Update room connectivity based on observed motion events.

    motion_events: list of (from_zone, to_zone, timestamp).
    Frequently traversed zone pairs get a doorway connection added.
    """
    # Count traversals between zone pairs
    traversal_counts: dict[tuple[str, str], int] = {}
    for from_zone, to_zone, _ts in motion_events:
        if from_zone == to_zone:
            continue
        pair = (min(from_zone, to_zone), max(from_zone, to_zone))
        traversal_counts[pair] = traversal_counts.get(pair, 0) + 1

    # Existing connections as a set for fast lookup
    existing: set[tuple[str, str]] = set()
    for conn in graph.connections:
        pair = (min(conn.room_a, conn.room_b), max(conn.room_a, conn.room_b))
        existing.add(pair)

    # Known room names
    room_names = {r.name for r in graph.rooms}

    new_connections = list(graph.connections)
    for pair, count in traversal_counts.items():
        if count >= _DOORWAY_THRESHOLD and pair not in existing:
            # Only add connections between known rooms
            if pair[0] in room_names and pair[1] in room_names:
                new_connections.append(Connection(
                    room_a=pair[0],
                    room_b=pair[1],
                    doorway_position=None,
                ))
                existing.add(pair)

    return RoomGraph(rooms=list(graph.rooms), connections=new_connections)
