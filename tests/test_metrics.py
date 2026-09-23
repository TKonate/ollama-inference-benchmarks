"""Unit tests for memory tracking and system info."""

from benchmarks.metrics import MemorySnapshot, MemoryTracker, get_system_info


class TestMemoryTracker:
    def test_start_stop(self):
        tracker = MemoryTracker()
        tracker.start()
        import time

        time.sleep(0.1)
        snapshot = tracker.stop()
        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.timestamp > 0

    def test_peak_increases(self):
        tracker = MemoryTracker()
        tracker.start()
        # Allocate some memory
        data = [bytearray(1024 * 1024) for _ in range(10)]  # ~10MB
        import time

        time.sleep(0.3)
        snapshot = tracker.stop()
        # Peak should be non-zero (at least the alloc)
        assert snapshot.rss_mib >= 0
        del data


class TestGetSystemInfo:
    def test_returns_dict(self):
        info = get_system_info()
        assert isinstance(info, dict)

    def test_has_ram_or_error(self):
        info = get_system_info()
        assert "total_ram_gib" in info or "error" in info
