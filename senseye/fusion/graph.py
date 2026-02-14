"""Signal graph: vertices (nodes/devices), edges (observations)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Vertex:
    id: str
    position: tuple[float, float] | None = None
    role: str = "fixed"  # "fixed" or "mobile"


@dataclass
class Edge:
    source: str
    target: str
    rssi: float
    attenuation: float = 0.0
    motion: bool = False
    timestamp: float = 0.0


class SignalGraph:
    def __init__(self) -> None:
        self._vertices: dict[str, Vertex] = {}
        self._edges: dict[tuple[str, str], Edge] = {}

    def add_vertex(self, vertex: Vertex) -> None:
        self._vertices[vertex.id] = vertex

    def get_vertex(self, vertex_id: str) -> Vertex | None:
        return self._vertices.get(vertex_id)

    def add_edge(self, edge: Edge) -> None:
        self._edges[(edge.source, edge.target)] = edge

    def update_edge(self, source: str, target: str, **kwargs) -> None:
        key = (source, target)
        edge = self._edges.get(key)
        if edge is None:
            return
        for k, v in kwargs.items():
            if hasattr(edge, k):
                setattr(edge, k, v)

    def get_edge(self, source: str, target: str) -> Edge | None:
        return self._edges.get((source, target))

    def get_neighbors(self, vertex_id: str) -> list[str]:
        neighbors: set[str] = set()
        for src, tgt in self._edges:
            if src == vertex_id:
                neighbors.add(tgt)
            elif tgt == vertex_id:
                neighbors.add(src)
        return list(neighbors)

    def get_edges_for(self, vertex_id: str) -> list[Edge]:
        return [
            e for (src, tgt), e in self._edges.items()
            if src == vertex_id or tgt == vertex_id
        ]

    def get_all_edges(self) -> list[Edge]:
        return list(self._edges.values())

    def get_all_vertices(self) -> list[Vertex]:
        return list(self._vertices.values())
