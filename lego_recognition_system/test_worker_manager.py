#!/usr/bin/env python3
"""
test_worker_manager.py — Unit tests for BlenderWorkerManager
=============================================================
Tests the dynamic semaphore logic, pressure classification,
and hysteresis without requiring psutil or a real Blender install.

Run:
    cd lego_recognition_system
    python -m pytest test_worker_manager.py -v
    # or directly:
    python test_worker_manager.py
"""

import sys
import os
import time
import threading
import unittest

# Allow running from docs_root or lego_recognition_system/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.blender_worker_manager import (
    BlenderWorkerManager,
    PressureLevel,
    MINIMUM_WORKERS,
    _classify_pressure,
    _sample_resources,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

class _PatchedManager(BlenderWorkerManager):
    """
    Subclass that allows injecting fake CPU/RAM values
    and skips the background poll sleep for fast testing.
    """
    def __init__(self, max_workers, fake_cpu=20.0, fake_ram=30.0):
        super().__init__(max_workers=max_workers, log_path=None, verbose=False)
        self.fake_cpu = fake_cpu
        self.fake_ram = fake_ram

    def inject_pressure(self, cpu: float, ram: float) -> None:
        """Simulate one monitoring cycle with the given metrics."""
        self.fake_cpu = cpu
        self.fake_ram = ram
        new_level = _classify_pressure(cpu, ram)

        # Re-run hysteresis logic from the monitor loop (condensed)
        if new_level == self._current_level:
            self._pending_level = None
            self._pending_count = 0
        else:
            if new_level == self._pending_level:
                self._pending_count += 1
            else:
                self._pending_level = new_level
                self._pending_count = 1

            if self._pending_count >= 2:  # HYSTERESIS_SAMPLES
                self._current_level = new_level
                new_target = self._compute_target(new_level)
                self._adjust_semaphore(new_target)
                self._pending_level = None
                self._pending_count = 0


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPressureClassification(unittest.TestCase):
    """Tests for _classify_pressure()"""

    def test_calm(self):
        self.assertEqual(_classify_pressure(20.0, 40.0), PressureLevel.CALM)
        self.assertEqual(_classify_pressure(49.9, 59.9), PressureLevel.CALM)

    def test_moderate_by_cpu(self):
        self.assertEqual(_classify_pressure(55.0, 40.0), PressureLevel.MODERATE)

    def test_moderate_by_ram(self):
        self.assertEqual(_classify_pressure(30.0, 65.0), PressureLevel.MODERATE)

    def test_high_by_cpu(self):
        self.assertEqual(_classify_pressure(75.0, 50.0), PressureLevel.HIGH)

    def test_high_by_ram(self):
        self.assertEqual(_classify_pressure(40.0, 80.0), PressureLevel.HIGH)

    def test_critical_by_cpu(self):
        self.assertEqual(_classify_pressure(90.0, 50.0), PressureLevel.CRITICAL)

    def test_critical_by_ram(self):
        self.assertEqual(_classify_pressure(40.0, 92.0), PressureLevel.CRITICAL)

    def test_both_critical(self):
        self.assertEqual(_classify_pressure(95.0, 95.0), PressureLevel.CRITICAL)


class TestTargetComputation(unittest.TestCase):
    """Tests for _compute_target() at different pressure levels"""

    def setUp(self):
        self.mgr = _PatchedManager(max_workers=8)

    def test_calm_full_capacity(self):
        target = self.mgr._compute_target(PressureLevel.CALM)
        self.assertEqual(target, 8)

    def test_moderate_75_percent(self):
        # 8 * 0.75 = 6
        target = self.mgr._compute_target(PressureLevel.MODERATE)
        self.assertEqual(target, 6)

    def test_high_50_percent(self):
        # 8 * 0.50 = 4
        target = self.mgr._compute_target(PressureLevel.HIGH)
        self.assertEqual(target, 4)

    def test_critical_minimum(self):
        target = self.mgr._compute_target(PressureLevel.CRITICAL)
        self.assertEqual(target, MINIMUM_WORKERS)

    def test_minimum_floor_on_small_pool(self):
        """Even tiny pools never go below MINIMUM_WORKERS"""
        small = _PatchedManager(max_workers=2)
        for level in PressureLevel:
            target = small._compute_target(level)
            self.assertGreaterEqual(target, MINIMUM_WORKERS,
                                    f"Level {level} produced target below minimum")


class TestHysteresis(unittest.TestCase):
    """Tests that level transitions require 2 consecutive samples"""

    def setUp(self):
        self.mgr = _PatchedManager(max_workers=8)

    def test_single_sample_does_not_change_level(self):
        """One HIGH sample should NOT change level from CALM."""
        self.assertEqual(self.mgr._current_level, PressureLevel.CALM)
        self.mgr.inject_pressure(cpu=80.0, ram=50.0)  # → HIGH candidate
        # Still CALM after just 1 sample
        self.assertEqual(self.mgr._current_level, PressureLevel.CALM)
        self.assertEqual(self.mgr._pending_count, 1)

    def test_two_samples_commit(self):
        """Two consecutive HIGH samples should commit the level change."""
        self.mgr.inject_pressure(cpu=80.0, ram=50.0)  # pending=1
        self.mgr.inject_pressure(cpu=80.0, ram=50.0)  # pending=2 → commit
        self.assertEqual(self.mgr._current_level, PressureLevel.HIGH)

    def test_interrupted_hysteresis_resets(self):
        """A different level interspersed resets the hysteresis counter."""
        self.mgr.inject_pressure(cpu=80.0, ram=50.0)  # HIGH pending=1
        self.mgr.inject_pressure(cpu=95.0, ram=50.0)  # CRITICAL → new candidate, count=1
        # Level is still CALM (neither reached 2 consecutive)
        self.assertEqual(self.mgr._current_level, PressureLevel.CALM)
        self.mgr.inject_pressure(cpu=95.0, ram=50.0)  # CRITICAL pending=2 → commit
        self.assertEqual(self.mgr._current_level, PressureLevel.CRITICAL)


class TestSemaphoreAdjustment(unittest.TestCase):
    """Tests for _adjust_semaphore() thread-safety and correctness"""

    def setUp(self):
        self.mgr = _PatchedManager(max_workers=8)

    def _count_semaphore(self) -> int:
        """Count current semaphore value by draining and refilling."""
        count = 0
        while self.mgr._semaphore.acquire(blocking=False):
            count += 1
        for _ in range(count):
            self.mgr._semaphore.release()
        return count

    def test_initial_semaphore_at_max(self):
        # After init, semaphore should be at max_workers
        value = self._count_semaphore()
        self.assertEqual(value, 8)

    def test_reduce_semaphore(self):
        self.mgr._adjust_semaphore(4)
        value = self._count_semaphore()
        self.assertEqual(value, 4)

    def test_increase_semaphore(self):
        # First reduce, then increase
        self.mgr._adjust_semaphore(4)
        self.mgr._adjust_semaphore(6)
        value = self._count_semaphore()
        self.assertEqual(value, 6)

    def test_semaphore_idempotent(self):
        """Adjusting to same value should be a no-op"""
        self.mgr._adjust_semaphore(8)
        self.mgr._adjust_semaphore(8)
        value = self._count_semaphore()
        self.assertEqual(value, 8)


class TestDynamicCycleIntegration(unittest.TestCase):
    """End-to-end test: simulate a full CALM→HIGH→CALM cycle"""

    def test_full_cycle(self):
        mgr = _PatchedManager(max_workers=10)

        # Start CALM — target = 10
        self.assertEqual(mgr._current_level, PressureLevel.CALM)
        self.assertEqual(mgr._current_target, 10)

        # Transition to HIGH (needs 2 samples)
        mgr.inject_pressure(cpu=78.0, ram=50.0)
        mgr.inject_pressure(cpu=78.0, ram=50.0)
        self.assertEqual(mgr._current_level, PressureLevel.HIGH)
        self.assertEqual(mgr._current_target, 5)   # 10 * 0.5

        # Back to CALM (needs 2 samples)
        mgr.inject_pressure(cpu=25.0, ram=40.0)
        mgr.inject_pressure(cpu=25.0, ram=40.0)
        self.assertEqual(mgr._current_level, PressureLevel.CALM)
        self.assertEqual(mgr._current_target, 10)

    def test_fake_worker_fn_runs_with_manager(self):
        """run_with_dynamic_workers executes all jobs correctly"""
        results_store = []
        lock = threading.Lock()

        def fake_worker(worker_id, piece_id, chunks, mode):
            time.sleep(0.05)  # Simulate short render
            with lock:
                results_store.append((worker_id, piece_id))
            return worker_id

        mgr = BlenderWorkerManager(max_workers=4, log_path=None, verbose=False)
        jobs = [(str(i), f"part_{i}", [], 'images_mix') for i in range(8)]
        results = mgr.run_with_dynamic_workers(jobs=jobs, worker_fn=fake_worker)

        self.assertEqual(len(results), 8)
        self.assertEqual(len(results_store), 8)


class TestFallbackWithoutPsutil(unittest.TestCase):
    """Verify that _sample_resources() returns sane values even without psutil"""

    def test_sample_returns_tuple_of_floats(self):
        cpu, ram = _sample_resources()
        self.assertIsInstance(cpu, float)
        self.assertIsInstance(ram, float)
        self.assertGreaterEqual(cpu, 0.0)
        self.assertLessEqual(cpu, 100.0)
        self.assertGreaterEqual(ram, 0.0)
        self.assertLessEqual(ram, 100.0)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("BlenderWorkerManager — Unit Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
