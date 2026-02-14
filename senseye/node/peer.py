from __future__ import annotations

import asyncio
import logging
import socket
import uuid
from collections.abc import Callable
from typing import Any

from zeroconf import IPVersion, ServiceInfo, ServiceStateChange, Zeroconf
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

from senseye.node.belief import Belief
from senseye.protocol import MessageReader, MessageWriter

log = logging.getLogger(__name__)

SERVICE_TYPE = "_senseye._tcp.local."
RECONNECT_BASE = 1.0
RECONNECT_MAX = 30.0


class PeerMesh:
    def __init__(self, node_id: str, port: int = 5483) -> None:
        self._node_id = node_id
        self._port = port

        # peer_id -> (reader, writer, MessageReader, MessageWriter)
        self._peers: dict[str, tuple[asyncio.StreamReader, asyncio.StreamWriter,
                                     MessageReader, MessageWriter]] = {}
        # Discovered peers: peer_id -> (host, port)
        self._discovered: dict[str, tuple[str, int]] = {}
        # Callbacks for received beliefs
        self._belief_callbacks: list[Callable[[Belief], Any]] = []
        # Callback invoked for incoming acoustic ping requests.
        self._acoustic_ping_callbacks: list[Callable[[str, dict[str, Any]], Any]] = []
        # request_id -> (peer_id, Future awaiting acoustic_pong response)
        self._pending_acoustic: dict[
            str,
            tuple[str, asyncio.Future[dict[str, Any]]],
        ] = {}
        # Tasks for per-peer read loops and reconnect loops
        self._peer_tasks: dict[str, asyncio.Task] = {}
        self._reconnect_tasks: dict[str, asyncio.Task] = {}
        # Lock to protect _peers mutations
        self._lock = asyncio.Lock()

        self._server: asyncio.Server | None = None
        self._azc: AsyncZeroconf | None = None
        self._browser: AsyncServiceBrowser | None = None
        self._running = False

    # -- Public API --

    async def start(self) -> None:
        self._running = True
        self._server = await asyncio.start_server(
            self._handle_incoming, host="0.0.0.0", port=self._port,
        )
        log.info("TCP server listening on port %d", self._port)

        self._azc = AsyncZeroconf(ip_version=IPVersion.V4Only)
        await self._register_service()
        self._browser = AsyncServiceBrowser(
            self._azc.zeroconf, SERVICE_TYPE, handlers=[self._on_service_change],
        )
        log.info("mDNS registered and browsing for peers")

    async def stop(self) -> None:
        self._running = False

        # Cancel reconnect tasks
        for task in self._reconnect_tasks.values():
            task.cancel()
        for task in self._reconnect_tasks.values():
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._reconnect_tasks.clear()

        # Close all peer connections
        async with self._lock:
            for peer_id in list(self._peers):
                await self._close_peer(peer_id)

        # Cancel read loop tasks
        for task in self._peer_tasks.values():
            task.cancel()
        for task in self._peer_tasks.values():
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._peer_tasks.clear()

        # Shut down mDNS
        if self._browser:
            await self._browser.async_cancel()
        if self._azc:
            await self._azc.async_unregister_all_services()
            await self._azc.async_close()

        # Shut down TCP server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        for _request_id, (_peer_id, future) in list(self._pending_acoustic.items()):
            if not future.done():
                future.cancel()
        self._pending_acoustic.clear()

        log.info("PeerMesh stopped")

    def on_belief(self, callback: Callable[[Belief], Any]) -> None:
        self._belief_callbacks.append(callback)

    def on_acoustic_ping(self, callback: Callable[[str, dict[str, Any]], Any]) -> None:
        self._acoustic_ping_callbacks.append(callback)

    async def broadcast_belief(self, belief: Belief) -> None:
        msg = {"type": "belief", **belief.to_dict()}
        async with self._lock:
            peers = list(self._peers.items())
        failed: list[str] = []
        for peer_id, (_, _, _, mw) in peers:
            try:
                await mw.write_message(msg)
            except (ConnectionError, OSError):
                log.debug("failed to send belief to %s", peer_id)
                failed.append(peer_id)
        for peer_id in failed:
            await self._disconnect_peer(peer_id)

    def get_peers(self) -> list[str]:
        return list(self._peers.keys())

    async def request_acoustic_ping(
        self,
        peer_id: str,
        *,
        delay_s: float = 0.2,
        sample_rate: int = 48_000,
        freq_start: int = 18_000,
        freq_end: int = 22_000,
        chirp_duration: float = 0.01,
        timeout: float = 3.0,
    ) -> dict[str, Any] | None:
        """Request a peer to emit an acoustic chirp and await acknowledgement."""
        request_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_acoustic[request_id] = (peer_id, future)

        sent = await self._send_to_peer(
            peer_id,
            {
                "type": "acoustic_ping",
                "request_id": request_id,
                "delay_s": delay_s,
                "sample_rate": sample_rate,
                "freq_start": freq_start,
                "freq_end": freq_end,
                "chirp_duration": chirp_duration,
            },
        )
        if not sent:
            self._pending_acoustic.pop(request_id, None)
            return None

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self._pending_acoustic.pop(request_id, None)

    # -- mDNS --

    async def _register_service(self) -> None:
        info = ServiceInfo(
            SERVICE_TYPE,
            f"{self._node_id}.{SERVICE_TYPE}",
            port=self._port,
            properties={"node_id": self._node_id},
            server=f"{self._node_id}.local.",
            addresses=[socket.inet_aton(self._get_local_ip())],
        )
        await self._azc.async_register_service(info)

    def _get_local_ip(self) -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"
        finally:
            s.close()

    def _on_service_change(self, **kwargs) -> None:
        zc = kwargs.get("zeroconf")
        service_type = kwargs.get("service_type", "")
        name = kwargs.get("name", "")
        state_change = kwargs.get("state_change")
        if state_change in (ServiceStateChange.Added, ServiceStateChange.Updated):
            asyncio.ensure_future(self._handle_discovered(zc, service_type, name))
        elif state_change == ServiceStateChange.Removed:
            # Extract node_id from service name: "{node_id}._senseye._tcp.local."
            peer_id = name.removesuffix(f".{SERVICE_TYPE}").removesuffix(".")
            if peer_id in self._discovered:
                del self._discovered[peer_id]
                log.info("peer %s removed from mDNS", peer_id)

    async def _handle_discovered(self, zc: Zeroconf, service_type: str, name: str) -> None:
        info = AsyncServiceInfo(service_type, name)
        await info.async_request(zc, 3000)

        if not info.addresses or not info.port:
            return

        peer_id = info.properties.get(b"node_id", b"").decode()
        if not peer_id or peer_id == self._node_id:
            return

        addresses = info.parsed_addresses(IPVersion.V4Only)
        host = addresses[0] if addresses else None
        if not host:
            return

        self._discovered[peer_id] = (host, info.port)
        log.info("discovered peer %s at %s:%d", peer_id, host, info.port)

        # Only connect if we are the client (lower node_id)
        if self._node_id < peer_id and peer_id not in self._peers:
            ok = await self._connect_to_peer(peer_id, host, info.port)
            if not ok:
                self._schedule_reconnect(peer_id, host, info.port)

    # -- TCP server (incoming connections) --

    async def _handle_incoming(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        mr = MessageReader(reader)
        # First message must be an announce so we learn the peer's node_id
        msg = await mr.read_message()
        if msg is None or msg.get("type") != "announce":
            writer.close()
            await writer.wait_closed()
            return

        peer_id = msg.get("node_id", "")
        if not peer_id or peer_id == self._node_id:
            writer.close()
            await writer.wait_closed()
            return

        # Validate dedup rule: the peer connecting to us should have a lower node_id
        if peer_id >= self._node_id:
            log.debug("rejecting incoming connection from %s (dedup rule)", peer_id)
            writer.close()
            await writer.wait_closed()
            return

        async with self._lock:
            if peer_id in self._peers:
                log.debug("already connected to %s, rejecting duplicate", peer_id)
                writer.close()
                await writer.wait_closed()
                return
            mw = MessageWriter(writer)
            self._peers[peer_id] = (reader, writer, mr, mw)

        log.info("accepted connection from %s", peer_id)
        self._peer_tasks[peer_id] = asyncio.ensure_future(self._read_loop(peer_id, mr))

    # -- TCP client (outgoing connections) --

    async def _connect_to_peer(self, peer_id: str, host: str, port: int) -> bool:
        """Attempt to connect to a peer. Returns True on success."""
        if peer_id in self._peers:
            return True

        try:
            reader, writer = await asyncio.open_connection(host, port)
        except OSError as e:
            log.debug("failed to connect to %s: %s", peer_id, e)
            return False

        mr = MessageReader(reader)
        mw = MessageWriter(writer)

        # Send announce message so the server knows who we are
        await mw.write_message({"type": "announce", "node_id": self._node_id})

        async with self._lock:
            if peer_id in self._peers:
                writer.close()
                await writer.wait_closed()
                return True
            self._peers[peer_id] = (reader, writer, mr, mw)

        log.info("connected to %s at %s:%d", peer_id, host, port)
        self._peer_tasks[peer_id] = asyncio.ensure_future(self._read_loop(peer_id, mr))
        return True

    # -- Read loop --

    async def _read_loop(self, peer_id: str, mr: MessageReader) -> None:
        try:
            async for msg in mr:
                msg_type = msg.get("type")
                if msg_type == "belief":
                    try:
                        belief = Belief.from_dict(msg)
                    except (KeyError, TypeError, ValueError):
                        log.debug("malformed belief from %s", peer_id)
                        continue
                    for cb in self._belief_callbacks:
                        try:
                            result = cb(belief)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception:
                            log.exception("belief callback error")
                elif msg_type == "acoustic_ping":
                    await self._handle_acoustic_ping(peer_id, msg)
                elif msg_type == "acoustic_pong":
                    self._resolve_acoustic_pong(peer_id, msg)
        except (ConnectionError, OSError):
            log.debug("connection error reading from %s", peer_id)
        finally:
            await self._disconnect_peer(peer_id)

    # -- Connection lifecycle --

    async def _disconnect_peer(self, peer_id: str) -> None:
        async with self._lock:
            if peer_id not in self._peers:
                return
            await self._close_peer(peer_id)

        for request_id, (pending_peer, future) in list(self._pending_acoustic.items()):
            if pending_peer != peer_id:
                continue
            if not future.done():
                future.set_exception(ConnectionError(f"peer {peer_id} disconnected"))
            self._pending_acoustic.pop(request_id, None)

        log.info("disconnected from %s", peer_id)

        # If we are the client side, schedule reconnect
        if self._running and self._node_id < peer_id and peer_id in self._discovered:
            host, port = self._discovered[peer_id]
            self._schedule_reconnect(peer_id, host, port)

    async def _close_peer(self, peer_id: str) -> None:
        """Close a peer connection. Caller must hold self._lock."""
        if peer_id not in self._peers:
            return
        _, writer, _, _ = self._peers.pop(peer_id)
        try:
            writer.close()
            await writer.wait_closed()
        except OSError:
            pass

    # -- Reconnection --

    def _schedule_reconnect(self, peer_id: str, host: str, port: int) -> None:
        if peer_id in self._reconnect_tasks and not self._reconnect_tasks[peer_id].done():
            return
        self._reconnect_tasks[peer_id] = asyncio.ensure_future(
            self._reconnect_loop(peer_id, host, port),
        )

    async def _reconnect_loop(self, peer_id: str, host: str, port: int) -> None:
        delay = RECONNECT_BASE
        while self._running and peer_id not in self._peers:
            log.debug("reconnecting to %s in %.1fs", peer_id, delay)
            await asyncio.sleep(delay)
            if not self._running or peer_id in self._peers:
                break
            if peer_id in self._discovered:
                host, port = self._discovered[peer_id]
            if await self._connect_to_peer(peer_id, host, port):
                break
            delay = min(delay * 2, RECONNECT_MAX)

    async def _send_to_peer(self, peer_id: str, message: dict[str, Any]) -> bool:
        async with self._lock:
            peer = self._peers.get(peer_id)
        if peer is None:
            return False
        _, _, _, writer = peer
        try:
            await writer.write_message(message)
            return True
        except (ConnectionError, OSError):
            await self._disconnect_peer(peer_id)
            return False

    async def _handle_acoustic_ping(self, peer_id: str, msg: dict[str, Any]) -> None:
        request_id = msg.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return

        ok = False
        error: str | None = None
        for callback in self._acoustic_ping_callbacks:
            try:
                result = callback(peer_id, msg)
                if asyncio.iscoroutine(result):
                    result = await result
                if bool(result):
                    ok = True
            except Exception as exc:
                error = str(exc)
                log.exception("acoustic ping callback error")

        response = {
            "type": "acoustic_pong",
            "request_id": request_id,
            "ok": ok,
            "error": error or "",
        }
        await self._send_to_peer(peer_id, response)

    def _resolve_acoustic_pong(self, peer_id: str, msg: dict[str, Any]) -> None:
        request_id = msg.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return
        pending = self._pending_acoustic.get(request_id)
        if pending is None:
            return
        pending_peer, future = pending
        if pending_peer != peer_id:
            return
        if not future.done():
            future.set_result(msg)
