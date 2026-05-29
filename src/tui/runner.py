"""
runner.py — Async subprocess launcher for dial-sft training jobs.

Manages a sequential job queue, redirects stdout/stderr to log files, and
provides an async generator for live log tailing (used by MonitorScreen).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import AsyncGenerator

# Repo root: src/tui/runner.py → parents[2] is repo root
_REPO_ROOT = Path(__file__).parents[2]
_LOGS_DIR = _REPO_ROOT / "logs"

# Maximum iterations waiting for log file to appear (each iteration = 0.1 s → 5 s total)
_LOG_FILE_WAIT_ITERATIONS = 50


class JobState(Enum):
    PENDING = auto()
    RUNNING = auto()
    DONE = auto()
    FAILED = auto()
    CANCELLED = auto()


def _make_run_id(override_string: str) -> str:
    """Generate a unique run_id from a timestamp and a slug of the overrides."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = override_string.replace(" ", "_").replace("=", "-").replace("/", "-")
    slug = slug[:60] if len(slug) > 60 else slug
    return f"{timestamp}__{slug}" if slug else timestamp


@dataclass
class Job:
    run_id: str
    override_string: str
    log_path: Path
    state: JobState = JobState.PENDING
    proc: asyncio.subprocess.Process | None = field(default=None, repr=False)


class Runner:
    """Manages a sequential queue of training jobs."""

    def __init__(self) -> None:
        _LOGS_DIR.mkdir(exist_ok=True)
        self._queue: list[Job] = []
        self._active_job: Job | None = None
        self._queue_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def submit(self, override_string: str) -> Job:
        """Add a single job to the queue and ensure the queue is running."""
        run_id = _make_run_id(override_string)
        log_path = _LOGS_DIR / f"{run_id}.log"
        job = Job(run_id=run_id, override_string=override_string, log_path=log_path)
        self._queue.append(job)
        self._ensure_queue_running()
        return job

    async def submit_queue(self, override_strings: list[str]) -> list[Job]:
        """Add multiple jobs to the queue (sequential multirun sweep)."""
        jobs = []
        for override_string in override_strings:
            job = await self.submit(override_string)
            jobs.append(job)
        return jobs

    async def cancel(self) -> None:
        """Cancel the currently running job and clear remaining pending jobs."""
        # Mark all pending jobs as cancelled
        for job in self._queue:
            if job.state == JobState.PENDING:
                job.state = JobState.CANCELLED

        # Kill active process
        if self._active_job and self._active_job.proc is not None:
            try:
                self._active_job.proc.terminate()
                try:
                    await asyncio.wait_for(self._active_job.proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._active_job.proc.kill()
            except ProcessLookupError:
                pass
            self._active_job.state = JobState.CANCELLED

        # Cancel the queue processing task
        if self._queue_task and not self._queue_task.done():
            self._queue_task.cancel()
            try:
                await self._queue_task
            except asyncio.CancelledError:
                pass
            self._queue_task = None

    async def tail(self, job: Job) -> AsyncGenerator[str, None]:
        """Async generator that yields new log lines as they arrive.

        Tails the log file for a running job, or reads a completed log file
        in full.
        """
        log_path = job.log_path
        # Wait for the log file to be created (with timeout)
        for _ in range(_LOG_FILE_WAIT_ITERATIONS):
            if log_path.exists():
                break
            await asyncio.sleep(0.1)

        if not log_path.exists():
            yield f"[Log file not found: {log_path}]\n"
            return

        with log_path.open("r", errors="replace") as fh:
            # Yield all existing content first
            while True:
                line = fh.readline()
                if line:
                    yield line
                else:
                    break

            # While the job is still running, keep polling for new lines
            while job.state in (JobState.PENDING, JobState.RUNNING):
                line = fh.readline()
                if line:
                    yield line
                else:
                    await asyncio.sleep(0.25)

            # Drain any remaining lines after job completes
            while True:
                line = fh.readline()
                if line:
                    yield line
                else:
                    break

    @property
    def active_job(self) -> Job | None:
        return self._active_job

    @property
    def queue(self) -> list[Job]:
        return list(self._queue)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_queue_running(self) -> None:
        """Start the queue processing task if it is not already running."""
        if self._queue_task is None or self._queue_task.done():
            self._queue_task = asyncio.create_task(self._process_queue())

    async def _process_queue(self) -> None:
        """Process jobs from the queue sequentially."""
        while True:
            # Find the next pending job
            next_job = next(
                (j for j in self._queue if j.state == JobState.PENDING), None
            )
            if next_job is None:
                break
            await self._run_job(next_job)

    async def _run_job(self, job: Job) -> None:
        """Execute a single training job as an async subprocess."""
        job.state = JobState.RUNNING
        self._active_job = job

        # Ensure log directory exists
        job.log_path.parent.mkdir(exist_ok=True)

        # Build subprocess environment — suppress ANSI progress bars in log
        env = os.environ.copy()
        env["TQDM_DISABLE"] = "1"
        env["HF_DATASETS_DISABLE_PROGRESS_BAR"] = "1"

        # Split override_string into individual args for create_subprocess_exec
        override_args = job.override_string.split() if job.override_string.strip() else []

        cmd = ["uv", "run", "python", "hydra_mdlm_sft.py"] + override_args

        with job.log_path.open("w") as log_fh:
            log_fh.write(f"[dial-sft] run_id: {job.run_id}\n")
            log_fh.write(f"[dial-sft] command: {' '.join(cmd)}\n")
            log_fh.write(f"[dial-sft] overrides: {job.override_string}\n")
            log_fh.write("[dial-sft] ---\n")
            log_fh.flush()

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=log_fh,
                    stderr=log_fh,
                    env=env,
                    cwd=str(_REPO_ROOT),
                )
                job.proc = proc
                return_code = await proc.wait()
                job.state = JobState.DONE if return_code == 0 else JobState.FAILED
            except asyncio.CancelledError:
                if job.proc is not None:
                    try:
                        job.proc.terminate()
                    except ProcessLookupError:
                        pass
                job.state = JobState.CANCELLED
                raise
            except Exception as exc:
                with job.log_path.open("a") as err_fh:
                    err_fh.write(f"[dial-sft] ERROR launching job: {exc}\n")
                job.state = JobState.FAILED
        self._active_job = None
