from __future__ import annotations

import asyncio
import json
import logging

log = logging.getLogger(__name__)

NEWLINE = b"\n"


def encode(message: dict) -> bytes:
    return json.dumps(message, separators=(",", ":")).encode() + NEWLINE


def decode(data: bytes) -> dict:
    return json.loads(data)


class MessageReader:
    """Reads newline-delimited JSON from an asyncio.StreamReader."""

    def __init__(self, reader: asyncio.StreamReader) -> None:
        self._reader = reader

    async def read_message(self) -> dict | None:
        """Read and parse one message. Returns None on EOF."""
        while True:
            try:
                line = await self._reader.readuntil(NEWLINE)
            except asyncio.IncompleteReadError as e:
                if e.partial:
                    log.debug("connection closed with partial data: %d bytes", len(e.partial))
                return None
            except asyncio.LimitOverrunError:
                log.warning("message exceeded buffer limit, discarding")
                try:
                    await self._reader.readuntil(NEWLINE)
                except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
                    pass
                continue

            line = line.strip()
            if not line:
                continue

            try:
                return json.loads(line)
            except json.JSONDecodeError:
                log.warning("malformed JSON, discarding line")
                continue

    async def __aiter__(self):
        while True:
            msg = await self.read_message()
            if msg is None:
                break
            yield msg


class MessageWriter:
    """Writes newline-delimited JSON to an asyncio.StreamWriter."""

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer

    async def write_message(self, message: dict) -> None:
        self._writer.write(encode(message))
        await self._writer.drain()

    def close(self) -> None:
        self._writer.close()

    async def wait_closed(self) -> None:
        await self._writer.wait_closed()
