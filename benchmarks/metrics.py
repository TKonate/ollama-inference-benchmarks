"""Host and process metrics that qualify a benchmark run.

Two kinds of number come from here: the environment fingerprint written into
every dataset header, and memory samples taken while a run is in flight. It
reads /proc and psutil, nothing else. It does not time inference, and what it
reports is what the kernel says, so RSS figures are resident pages rather than
private allocations.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

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
    """Track the benchmark client's own memory during a run.

    This watches the process running the benchmark, which is useful for sizing
    the harness and misleading for sizing a model: the weights live in the
    backend. Use :class:`ServerMemoryTracker` for that.

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
        # 500 ms: long enough to cost nothing, short enough to catch a load spike.
        self._interval: float = 0.5

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
                # A failing sample must not kill a run that is already measuring:
                # psutil raises on conditions that come and go, and one missed
                # tick costs less than the whole run.
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


def read_cpu_model() -> str:
    """Return the CPU model string, from /proc/cpuinfo when available."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


@dataclass(frozen=True)
class Environment:
    """Machine fingerprint attached to a benchmark dataset.

    A result without its environment is not reproducible: the same model can
    differ by 10x between a 2-vCPU VPS and a modern desktop. Every dataset this
    project publishes carries this block.
    """

    captured_at: str
    cpu_model: str
    cpu_count: int
    total_ram_gib: float
    available_ram_gib: float
    kernel: str
    os_name: str
    python_version: str
    ollama_version: str | None = None
    load_average: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, object]:
        """Flatten to a JSON-serialisable mapping."""
        return {
            "captured_at": self.captured_at,
            "cpu_model": self.cpu_model,
            "cpu_count": self.cpu_count,
            "total_ram_gib": self.total_ram_gib,
            "available_ram_gib": self.available_ram_gib,
            "kernel": self.kernel,
            "os_name": self.os_name,
            "python_version": self.python_version,
            "ollama_version": self.ollama_version,
            "load_average": list(self.load_average) if self.load_average else None,
        }


def get_environment(ollama_version: str | None = None) -> Environment:
    """Capture the environment a benchmark campaign runs in."""
    total_ram_gib = 0.0
    available_ram_gib = 0.0
    cpu_count = os.cpu_count() or 0

    if HAS_PSUTIL and _psutil is not None:
        try:
            vm = _psutil.virtual_memory()
            total_ram_gib = round(vm.total / (1024**3), 1)
            available_ram_gib = round(vm.available / (1024**3), 1)
            cpu_count = _psutil.cpu_count(logical=True) or cpu_count
        except Exception:
            pass

    try:
        load_average: tuple[float, float, float] | None = os.getloadavg()
    except OSError:
        load_average = None

    return Environment(
        captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        cpu_model=read_cpu_model(),
        cpu_count=cpu_count,
        total_ram_gib=total_ram_gib,
        available_ram_gib=available_ram_gib,
        kernel=platform.release(),
        os_name=platform.platform(terse=True),
        python_version=platform.python_version(),
        ollama_version=ollama_version,
        load_average=load_average,
    )


def _read_vm_rss_mib(pid: int) -> float | None:
    """Read current RSS of a process from /proc/<pid>/status."""
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    kib = float(line.split()[1])  # the kernel reports KiB here
                    return kib / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def _child_pids(pid: int) -> list[int]:
    """Return the direct children of a process using the kernel's children file."""
    try:
        with open(f"/proc/{pid}/task/{pid}/children", encoding="utf-8") as handle:
            return [int(value) for value in handle.read().split() if value.isdigit()]
    except (OSError, ValueError):
        return []


def _tree_rss_mib(pid: int) -> float | None:
    """Sum RSS over a process and its descendants.

    Kept for servers that do fork their workers (vLLM does); Ollama detaches
    its runner, which is why campaigns also pass a ``pid_provider``.
    """
    total = _read_vm_rss_mib(pid) or 0.0
    seen = {pid}
    stack = [pid]
    while stack:
        current = stack.pop()
        for child in _child_pids(current):
            if child in seen:
                continue
            seen.add(child)
            total += _read_vm_rss_mib(child) or 0.0
            stack.append(child)
    return total or None


class ServerMemoryTracker:
    """Sample the RSS of the inference server process while a run is in flight.

    ``MemoryTracker`` watches the client process, which is useless for
    inference: the weights live in the backend process (the Ollama daemon).
    This tracker follows the server PID instead, so the reported figure is the
    memory the model actually occupies: weights plus KV cache.

    Usage:
        tracker = ServerMemoryTracker(pid)
        tracker.start()
        result = client.generate(request)
        peak = tracker.stop()  # MiB, None when the PID was unknown or gone
    """

    def __init__(
        self,
        pid: int | None = None,
        *,
        pid_provider: Callable[[], list[int]] | None = None,
        interval: float = 0.05,
    ) -> None:
        self.pid = pid
        self._pid_provider = pid_provider
        self._interval = interval
        self._peak_mib: float = 0.0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _pids(self) -> list[int]:
        """Resolve the processes to sample, preferring the runner finder.

        The provider is called on every tick because a runner is a short-lived
        process: it is replaced when the model is evicted or reloaded.
        """
        if self._pid_provider is not None:
            found = self._pid_provider()
            if found:
                return found
        if self.pid:
            return [self.pid, *_child_pids(self.pid)]
        return []

    def _poll(self) -> None:
        if not self._pids():
            return
        while not self._stop_event.is_set():
            current = sum(_read_vm_rss_mib(pid) or 0.0 for pid in self._pids())
            if current > self._peak_mib:
                self._peak_mib = current
            self._stop_event.wait(self._interval)

    def start(self) -> None:
        """Begin sampling."""
        self._peak_mib = 0.0
        self._stop_event.clear()
        if not self.pid and self._pid_provider is None:
            return
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        """Stop sampling and return the peak RSS in MiB (None when unavailable)."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        # One last read after the thread is gone: the final poll can sit up to
        # one interval behind a load or unload that happened at the tail end.
        current = sum(_read_vm_rss_mib(pid) or 0.0 for pid in self._pids())
        if current > self._peak_mib:
            self._peak_mib = current
        return round(self._peak_mib, 1) if self._peak_mib > 0 else None


def find_server_pid(port: int = 11434, name_hint: str = "ollama") -> int | None:
    """Locate the process listening on ``port`` (the inference server).

    Falls back to matching a process name when socket introspection is
    unavailable, which happens in some containers.
    """
    if not HAS_PSUTIL or _psutil is None:
        return None

    with contextlib.suppress(Exception):
        for conn in _psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.pid:
                return int(conn.pid)

    with contextlib.suppress(Exception):
        for proc in _psutil.process_iter(["name"]):
            proc_name = (proc.info.get("name") or "").lower()
            if name_hint in proc_name and proc.pid != os.getpid():
                return int(proc.pid)

    return None


def runner_pids(name_hint: str = "llama-server") -> list[int]:
    """Return the PIDs of the backend's model runners.

    The runner is *not* a child of the server: Ollama detaches it, so walking
    the server's process tree finds nothing and memory reports come out two
    orders of magnitude too small. Matching the process command line is the
    only reliable way to find it without privileges.
    """
    matches: list[int] = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return matches

    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                cmdline = handle.read().decode("utf-8", "replace")
        except OSError:
            continue
        if name_hint in cmdline:
            matches.append(pid)
    return matches
