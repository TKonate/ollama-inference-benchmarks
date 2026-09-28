"""System metrics collection for benchmark runs."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import psutil as _psutil

    HAS_PSUTIL = True
except ImportError:
    _psutil = None  # type: ignore[assignment]
    HAS_PSUTIL = False
    logger.debug("psutil not installed — memory metrics will be unavailable")


@dataclass
class MemorySnapshot:
    """A point-in-time memory observation."""

    rss_mib: float = 0.0
    available_mib: float = 0.0
    timestamp: float = 0.0


class MemoryTracker:
    """Track process memory during a benchmark run.

    Usage:
        tracker = MemoryTracker()
        tracker.start()
        # ... run benchmark ...
        snapshot = tracker.stop()
        print(snapshot.rss_mib)
    """

    def __init__(self) -> None:
        self._peak_rss_mib: float = 0.0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._interval: float = 0.5  # poll every 500ms

    def _poll(self) -> None:
        """Continuously sample RSS until stopped."""
        if not HAS_PSUTIL or _psutil is None:
            return

        process = _psutil.Process()
        while not self._stop_event.is_set():
            try:
                mem = process.memory_info()
                rss_mib = mem.rss / (1024 * 1024)
                if rss_mib > self._peak_rss_mib:
                    self._peak_rss_mib = rss_mib
            except Exception:
                pass
            self._stop_event.wait(self._interval)

    def start(self) -> None:
        """Start background memory tracking."""
        self._peak_rss_mib = 0.0
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> MemorySnapshot:
        """Stop tracking and return the peak memory snapshot."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

        available_mib = 0.0
        if HAS_PSUTIL and _psutil is not None:
            try:
                vm = _psutil.virtual_memory()
                available_mib = vm.available / (1024 * 1024)
            except Exception:
                pass

        return MemorySnapshot(
            rss_mib=round(self._peak_rss_mib, 2),
            available_mib=round(available_mib, 2),
            timestamp=time.time(),
        )


def get_system_info() -> dict[str, str | int | float]:
    """Return a summary of system hardware for benchmark metadata."""
    info: dict[str, str | int | float] = {}

    if not HAS_PSUTIL or _psutil is None:
        info["error"] = "psutil not installed"
        return info

    try:
        vm = _psutil.virtual_memory()
        info["total_ram_gib"] = round(vm.total / (1024**3), 1)
        info["available_ram_gib"] = round(vm.available / (1024**3), 1)
    except Exception:
        pass

    try:
        cpu_count = _psutil.cpu_count(logical=True)
        if cpu_count is not None:
            info["cpu_count"] = cpu_count
        freq = _psutil.cpu_freq()
        if freq is not None:
            info["cpu_freq_mhz"] = freq.current
    except Exception:
        pass

    return info
