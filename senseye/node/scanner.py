"""WiFi + BLE RSSI scanning."""

from __future__ import annotations

import asyncio
import platform
import re
import time
from dataclasses import dataclass, field
from enum import Enum


class SignalType(Enum):
    WIFI = "wifi"
    BLE = "ble"
    ACOUSTIC = "acoustic"


@dataclass(frozen=True, slots=True)
class Observation:
    device_id: str
    rssi: float
    timestamp: float
    signal_type: SignalType
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# WiFi scanning
# ---------------------------------------------------------------------------

_SYSTEM = platform.system()


def _parse_airport_output(raw: str) -> list[Observation]:
    """Parse macOS airport -s output into observations."""
    lines = raw.strip().splitlines()
    if len(lines) < 2:
        return []
    results: list[Observation] = []
    now = time.time()
    # Header line followed by data lines. airport -s output is fixed-width:
    # the SSID is right-justified in the first 33 chars, then whitespace-separated fields.
    for line in lines[1:]:
        # Match: SSID (may have spaces), BSSID, RSSI, CHANNEL, HT, CC, SECURITY
        m = re.match(
            r"\s*(.+?)\s+([0-9a-fA-F:]{17})\s+(-?\d+)\s+",
            line,
        )
        if m:
            ssid, bssid, rssi_str = m.group(1).strip(), m.group(2), m.group(3)
            results.append(Observation(
                device_id=bssid.lower(),
                rssi=float(rssi_str),
                timestamp=now,
                signal_type=SignalType.WIFI,
                metadata={"ssid": ssid},
            ))
    return results


def _parse_iwlist_output(raw: str) -> list[Observation]:
    """Parse Linux iwlist scan output into observations."""
    results: list[Observation] = []
    now = time.time()
    current_bssid: str | None = None
    current_rssi: float | None = None
    current_ssid: str | None = None

    for line in raw.splitlines():
        line = line.strip()
        bssid_match = re.match(r"Cell \d+ - Address: ([0-9A-Fa-f:]{17})", line)
        if bssid_match:
            # Emit previous cell
            if current_bssid is not None and current_rssi is not None:
                results.append(Observation(
                    device_id=current_bssid.lower(),
                    rssi=current_rssi,
                    timestamp=now,
                    signal_type=SignalType.WIFI,
                    metadata={"ssid": current_ssid or ""},
                ))
            current_bssid = bssid_match.group(1)
            current_rssi = None
            current_ssid = None
            continue

        signal_match = re.match(r"Signal level[=:](-?\d+)", line)
        if signal_match:
            current_rssi = float(signal_match.group(1))
            continue

        ssid_match = re.match(r'ESSID:"(.+)"', line)
        if ssid_match:
            current_ssid = ssid_match.group(1)

    # Emit last cell
    if current_bssid is not None and current_rssi is not None:
        results.append(Observation(
            device_id=current_bssid.lower(),
            rssi=current_rssi,
            timestamp=now,
            signal_type=SignalType.WIFI,
            metadata={"ssid": current_ssid or ""},
        ))
    return results


def _scan_wifi_corewlan() -> list[Observation]:
    """Scan WiFi using macOS CoreWLAN framework (replaces deprecated airport command).

    Without Location Services permission, BSSID/SSID are redacted. We still
    get RSSI + channel for every nearby AP, which is enough for signal-based
    sensing.  We use channel as a stable-ish identifier when BSSID is null
    and always include the currently-connected network (whose info is available
    regardless of permissions).
    """
    import CoreWLAN

    iface = CoreWLAN.CWWiFiClient.sharedWiFiClient().interface()
    if iface is None:
        return []

    now = time.time()
    results: list[Observation] = []

    # Current network info is always available
    cur_bssid = iface.bssid()
    cur_ssid = iface.ssid()
    cur_rssi = iface.rssiValue()
    if cur_bssid:
        results.append(Observation(
            device_id=cur_bssid.lower(),
            rssi=float(cur_rssi),
            timestamp=now,
            signal_type=SignalType.WIFI,
            metadata={"ssid": cur_ssid or "", "connected": True},
        ))

    networks, error = iface.scanForNetworksWithName_error_(None, None)
    if error or not networks:
        return results

    # Track per-channel count to differentiate APs on same channel
    ch_count: dict[int, int] = {}
    for net in networks:
        bssid = net.bssid()
        ssid = net.ssid()
        channel = net.wlanChannel()
        ch_num = channel.channelNumber() if channel else 0

        if bssid:
            dev_id = bssid.lower()
        elif ch_num:
            # No BSSID due to privacy — use channel+index as pseudo-identifier
            idx = ch_count.get(ch_num, 0)
            ch_count[ch_num] = idx + 1
            dev_id = f"wifi-ch{ch_num}-{idx}"
        else:
            continue

        # Skip if this is the already-added connected network
        if bssid and cur_bssid and bssid.lower() == cur_bssid.lower():
            continue

        results.append(Observation(
            device_id=dev_id,
            rssi=float(net.rssiValue()),
            timestamp=now,
            signal_type=SignalType.WIFI,
            metadata={
                "ssid": ssid or "",
                "channel": ch_num,
            },
        ))
    return results


async def scan_wifi() -> list[Observation]:
    if _SYSTEM == "Darwin":
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _scan_wifi_corewlan)
    else:
        # Linux: try iwlist
        proc = await asyncio.create_subprocess_exec(
            "iwlist", "wlan0", "scan",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        return _parse_iwlist_output(stdout.decode(errors="replace"))


# ---------------------------------------------------------------------------
# BLE scanning
# ---------------------------------------------------------------------------

async def scan_ble(duration: float = 2.0) -> list[Observation]:
    from bleak import BleakScanner

    devices = await BleakScanner.discover(timeout=duration, return_adv=True)
    now = time.time()
    results: list[Observation] = []
    for _addr, (device, adv_data) in devices.items():
        meta: dict = {}
        if adv_data.manufacturer_data:
            # Store first manufacturer id + data as hex
            for mfr_id, mfr_bytes in adv_data.manufacturer_data.items():
                meta["manufacturer_id"] = mfr_id
                meta["manufacturer_data"] = mfr_bytes.hex()
                break
        if adv_data.service_uuids:
            meta["service_uuids"] = adv_data.service_uuids
        if adv_data.local_name:
            meta["name"] = adv_data.local_name

        results.append(Observation(
            device_id=device.address.lower(),
            rssi=float(adv_data.rssi),
            timestamp=now,
            signal_type=SignalType.BLE,
            metadata=meta,
        ))
    return results


# ---------------------------------------------------------------------------
# Combined scanner
# ---------------------------------------------------------------------------

async def scan_all(
    wifi: bool = True,
    ble: bool = True,
    ble_duration: float = 2.0,
) -> list[Observation]:
    tasks = []
    if wifi:
        tasks.append(scan_wifi())
    if ble:
        tasks.append(scan_ble(ble_duration))
    if not tasks:
        return []
    results_lists = await asyncio.gather(*tasks, return_exceptions=True)
    combined: list[Observation] = []
    for result in results_lists:
        if isinstance(result, list):
            combined.extend(result)
        # Silently skip exceptions — scanner unavailability is expected
    return combined
