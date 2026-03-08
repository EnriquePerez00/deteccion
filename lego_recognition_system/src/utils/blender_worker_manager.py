#!/usr/bin/env python3
"""
BlenderWorkerManager — Dynamic Worker Protocol for macOS
=========================================================
Monitors HW resource usage in real-time and dynamically adjusts
the number of concurrent Blender processes to maintain a stable
interactive macOS environment.

Architecture:
  - A background monitor thread samples CPU% + RAM% every POLL_INTERVAL seconds
  - Pressure levels: CALM → MODERATE → HIGH → CRITICAL
  - A threading.Semaphore gates how many Blender workers run concurrently
  - When pressure drops: semaphore is released (more workers allowed)
  - When pressure rises: excess semaphore permits are reclaimed (workers
    finish their current image then pause before acquiring the next slot)

Hysteresis: a level change requires HYSTERESIS_SAMPLES consecutive readings
in the new zone before being applied (prevents oscillation).

Usage:
    manager = BlenderWorkerManager(max_workers=18, log_path="worker_mgr.log")
    manager.run_with_dynamic_workers(
        jobs=[(worker_id, piece_id, chunks, mode), ...],
        worker_fn=run_render_worker,
    )
"""

import os
import sys
import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Tuple, Any, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

POLL_INTERVAL = 3          # Faster response to system pressure
HYSTERESIS_SAMPLES = 2     # Maintain responsiveness
LOG_INTERVAL = 15

# Pressure thresholds (Lowered to leave more "air" for macOS UI)
THRESHOLDS = {
    # Level     CPU_low  CPU_high  RAM_low  RAM_high
    "CALM":     (0,      65,       0,       60), # Was 70, 65
    "MODERATE": (65,     80,       60,      75), # Was 85, 80
    "HIGH":     (80,     90,       75,      85), # Was 92, 90
    "CRITICAL": (90,     100,      85,      100),
}

# Process Priority (19 is the lowest priority, leaving more room for WindowServer/kernel_task)
NICE_VALUE = 19


class PressureLevel(Enum):
    CALM     = "CALM"
    MODERATE = "MODERATE"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


LEVEL_ORDER = [
    PressureLevel.CALM,
    PressureLevel.MODERATE,
    PressureLevel.HIGH,
    PressureLevel.CRITICAL,
]

# Worker target as fraction of max_workers for each pressure level
LEVEL_FRACTION = {
    PressureLevel.CALM:     1.00,
    PressureLevel.MODERATE: 0.60,  # Was 0.85 (more conservative)
    PressureLevel.HIGH:     0.30,  # Was 0.60 (more conservative)
    PressureLevel.CRITICAL: 0.00,  # -> min 2
}

MINIMUM_WORKERS = 2   # Never drop below this (avoids stall)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _classify_pressure(cpu_pct: float, ram_pct: float) -> PressureLevel:
    """Return the highest pressure level triggered by current metrics (Updated thresholds)."""
    if cpu_pct > 90 or ram_pct > 85:
        return PressureLevel.CRITICAL
    if cpu_pct > 80 or ram_pct > 75:
        return PressureLevel.HIGH
    if cpu_pct > 65 or ram_pct > 60:
        return PressureLevel.MODERATE
    return PressureLevel.CALM


def _sample_resources() -> Tuple[float, float]:
    """Return (cpu_pct, ram_pct) using psutil if available, else estimates."""
    if PSUTIL_AVAILABLE:
        cpu = psutil.cpu_percent(interval=1)          # 1-s blocking sample
        ram = psutil.virtual_memory().percent
        return cpu, ram
    else:
        # Fallback: read /proc/loadavg on Linux; on macOS parse `vm_stat`
        try:
            import subprocess
            out = subprocess.check_output(
                ["vm_stat"], text=True, timeout=2
            )
            pages_free = 0
            pages_total = 0
            for line in out.splitlines():
                if "Pages free" in line:
                    pages_free = int(line.split(":")[1].strip().rstrip("."))
            # Conservative estimate: assume 80% RAM usage when psutil absent
            return 60.0, 80.0
        except Exception:
            return 60.0, 80.0


# ──────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────

class BlenderWorkerManager:
    """
    Dynamic worker concurrency manager for Blender render subprocesses.

    Parameters
    ----------
    max_workers : int
        Absolute maximum number of concurrent Blender processes.
        Computed by the caller using the existing M4/resolution logic.
    log_path : str | None
        Path to the manager log file. If None, logs to stdout.
    verbose : bool
        If True, print status lines to stdout in addition to the log.
    """

    def __init__(
        self,
        max_workers: int,
        log_path: Optional[str] = None,
        verbose: bool = True,
        low_priority: bool = True,
    ):
        self.max_workers = max(MINIMUM_WORKERS, max_workers)
        self.verbose = verbose
        self.low_priority = low_priority

        # Current concurrency floor  (semaphore value)
        self._current_target: int = self.max_workers

        # Semaphore that gates how many workers run simultaneously
        # Initialised to max_workers; the monitor will reduce it if needed.
        self._semaphore = threading.Semaphore(self.max_workers)

        # Hysteresis state
        self._pending_level: Optional[PressureLevel] = None
        self._pending_count: int = 0
        self._current_level: PressureLevel = PressureLevel.CALM

        # Control flag for the monitor thread
        self._running = threading.Event()
        self._running.set()

        # Lock for semaphore adjustments (thread-safe)
        self._adjust_lock = threading.Lock()

        # Logger
        self._logger = self._setup_logger(log_path)
        self._logger.info(
            f"BlenderWorkerManager initialised | max_workers={self.max_workers}"
            + (" | psutil=OK" if PSUTIL_AVAILABLE else " | psutil=MISSING (fallback)")
        )

        # Track ongoing futures for monitoring
        self._active_count: int = 0
        self._count_lock = threading.Lock()

    # ── Logging ────────────────────────────────────────────────

    def _setup_logger(self, log_path: Optional[str]) -> logging.Logger:
        logger = logging.getLogger("BlenderWorkerManager")
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        if self.verbose or not log_path:
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            logger.addHandler(sh)
        return logger

    # ── Target computation ─────────────────────────────────────

    def _compute_target(self, level: PressureLevel) -> int:
        fraction = LEVEL_FRACTION[level]
        if fraction == 0.0:
            # CRITICAL — use minimum
            return MINIMUM_WORKERS
        raw = int(self.max_workers * fraction)
        return max(MINIMUM_WORKERS, raw)

    # ── Semaphore adjustment ───────────────────────────────────

    def _adjust_semaphore(self, new_target: int) -> None:
        """
        Thread-safe adjustment of the semaphore value.
        Increasing: release permits.
        Decreasing: drain permits (non-blocking acquire attempts).
        """
        with self._adjust_lock:
            old_target = self._current_target
            if new_target == old_target:
                return

            if new_target > old_target:
                # Release the extra permits
                delta = new_target - old_target
                for _ in range(delta):
                    self._semaphore.release()
                self._logger.info(
                    f"[{self._current_level.value}] ↑ Workers: {old_target} → {new_target}"
                )
            else:
                # Drain excess permits (non-blocking — don't block running workers)
                delta = old_target - new_target
                drained = 0
                for _ in range(delta):
                    acquired = self._semaphore.acquire(blocking=False)
                    if acquired:
                        drained += 1
                self._logger.info(
                    f"[{self._current_level.value}] ↓ Workers: {old_target} → {new_target} "
                    f"(drained {drained}/{delta} permits; {delta-drained} will drain naturally)"
                )

            self._current_target = new_target

    # ── Monitor thread ─────────────────────────────────────────

    def _monitor_loop(self) -> None:
        """Background thread: sample resources and adjust concurrency."""
        last_log_time = 0.0

        while self._running.is_set():
            try:
                cpu, ram = _sample_resources()
                new_level = _classify_pressure(cpu, ram)

                # ── Hysteresis logic ──────────────────────────
                if new_level == self._current_level:
                    # Stable — reset pending state
                    self._pending_level = None
                    self._pending_count = 0
                else:
                    if new_level == self._pending_level:
                        self._pending_count += 1
                    else:
                        # New candidate level
                        self._pending_level = new_level
                        self._pending_count = 1

                    if self._pending_count >= HYSTERESIS_SAMPLES:
                        # Commit the level change
                        old_level = self._current_level
                        self._current_level = new_level
                        new_target = self._compute_target(new_level)
                        self._adjust_semaphore(new_target)
                        self._pending_level = None
                        self._pending_count = 0
                        self._logger.info(
                            f"Pressure: {old_level.value} → {new_level.value} "
                            f"| CPU={cpu:.1f}% RAM={ram:.1f}% "
                            f"| Workers capped at {new_target}/{self.max_workers}"
                        )

                # ── Periodic status log ───────────────────────
                now = time.time()
                if now - last_log_time >= LOG_INTERVAL:
                    with self._count_lock:
                        active = self._active_count
                    self._logger.info(
                        f"[{self._current_level.value}] "
                        f"CPU={cpu:.1f}% RAM={ram:.1f}% "
                        f"| Active={active}/{self._current_target} "
                        f"(max_cap={self.max_workers})"
                    )
                    last_log_time = now

            except Exception as exc:
                self._logger.warning(f"Monitor error (non-fatal): {exc}")

            # Wait POLL_INTERVAL seconds, but check _running frequently
            for _ in range(int(POLL_INTERVAL / 0.5)):
                if not self._running.is_set():
                    break
                time.sleep(0.5)

    # ── Worker wrapper ─────────────────────────────────────────

    def _managed_worker(self, worker_fn: Callable, *args) -> Any:
        """
        Wrapper that acquires a semaphore permit before calling worker_fn
        and releases it after completion. This is what actually gates
        concurrent Blender process launches.
        """
        # Block until a permit is available (respects dynamic cap)
        self._semaphore.acquire()
        
        # Lower process priority to leave room for macOS UI (kernel_task/WindowServer)
        if self.low_priority and PSUTIL_AVAILABLE:
            try:
                p = psutil.Process()
                # On macOS, setting niceness to 15-19 puts the process in the "background" 
                # scheduling policy, which is exactly what we want.
                p.nice(NICE_VALUE)
            except Exception:
                pass

        with self._count_lock:
            self._active_count += 1
        try:
            return worker_fn(*args)
        finally:
            with self._count_lock:
                self._active_count -= 1
            self._semaphore.release()

    # ── Public API ─────────────────────────────────────────────

    def run_with_dynamic_workers(
        self,
        jobs: List[Tuple],
        worker_fn: Callable,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Any]:
        """
        Run all jobs with dynamic worker concurrency.

        Parameters
        ----------
        jobs : list of tuples
            Each tuple is passed as *args to worker_fn.
        worker_fn : callable
            The render worker function (e.g. run_render_worker).
        progress_callback : callable(done, total) | None
            Called after each job completes for progress tracking.

        Returns
        -------
        list of results from worker_fn, in completion order.
        """
        if not jobs:
            return []

        total = len(jobs)
        done_count = 0

        # Start monitor thread
        monitor = threading.Thread(
            target=self._monitor_loop,
            name="BlenderWorkerMonitor",
            daemon=True,
        )
        monitor.start()
        self._logger.info(
            f"🚀 Starting dynamic render pool | {total} jobs | "
            f"max_workers={self.max_workers} | level={self._current_level.value}"
        )

        results = []
        # Use max_workers as the thread pool size — the semaphore controls
        # actual Blender process concurrency, independently of thread count.
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._managed_worker, worker_fn, *job_args): i
                for i, job_args in enumerate(jobs)
            }

            try:
                for future in as_completed(futures):
                    done_count += 1
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as exc:
                        job_idx = futures[future]
                        self._logger.error(f"Job {job_idx} raised: {exc}")
                        results.append(None)

                    if progress_callback:
                        progress_callback(done_count, total)
            except KeyboardInterrupt:
                self._logger.warning("KeyboardInterrupt — stopping workers gracefully...")
                executor.shutdown(wait=False, cancel_futures=True)

        # Stop monitor
        self._running.clear()
        monitor.join(timeout=3.0)
        self._logger.info(
            f"✅ Dynamic render pool finished | {done_count}/{total} jobs completed"
        )
        return results

    def stop(self) -> None:
        """Signal the monitor thread to stop."""
        self._running.clear()
