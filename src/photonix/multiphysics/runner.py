"""Safe subprocess runner for prepared external multiphysics jobs."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from .adapters import SolverAdapter
from .contracts import MultiphysicsJob, SimulationResult

__all__ = ["SolverUnavailableError", "SolverExecutionError", "run_job"]


class SolverUnavailableError(RuntimeError):
    """Raised before execution when a solver capability probe fails."""


class SolverExecutionError(RuntimeError):
    """A process failed, timed out, or did not produce required output files."""

    def __init__(self, result: SimulationResult):
        self.result = result
        if result.timed_out:
            detail = "timed out"
        elif result.returncode != 0:
            detail = f"exited with code {result.returncode}"
        else:
            detail = f"did not produce {len(result.missing_outputs)} required output(s)"
        super().__init__(f"solver {result.solver!r} job {result.job_name!r} {detail}")


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def run_job(
    adapter: SolverAdapter,
    job: MultiphysicsJob,
    *,
    check: bool = True,
    require_available: bool = True,
    environment: dict[str, str] | None = None,
) -> SimulationResult:
    """Execute one prepared job as an argv sequence with ``shell=False``.

    The runner validates paths and declared outputs but never parses fields or
    fabricates physical observables.  Use a dedicated, empty work directory when
    output freshness matters; presence checks cannot distinguish stale files.
    """
    assert job.workdir is not None  # normalized by MultiphysicsJob.__post_init__
    workdir = Path(job.workdir)
    if not workdir.is_dir():
        raise FileNotFoundError(f"job workdir does not exist or is not a directory: {workdir}")
    if not job.input_file.is_file():
        raise FileNotFoundError(f"job input_file does not exist or is not a file: {job.input_file}")

    process_environment = dict(os.environ)
    if environment is not None:
        process_environment.update(environment)
    process_environment.update(job.environment)
    capability = adapter.capability(process_environment)
    if require_available and not capability.available:
        raise SolverUnavailableError(capability.reason or f"solver {adapter.name!r} is unavailable")

    command = tuple(str(value) for value in adapter.build_command(job))
    if not command or any(not value for value in command):
        raise ValueError("adapter returned an empty command or argv element")
    started = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(workdir),
            env=process_environment,
            capture_output=True,
            text=True,
            timeout=job.timeout_s,
            check=False,
            shell=False,
        )
        returncode = int(completed.returncode)
        stdout = _text(completed.stdout)
        stderr = _text(completed.stderr)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        stdout = _text(exc.stdout)
        stderr = _text(exc.stderr)
    outputs = tuple(job.required_outputs)
    missing = tuple(path for path in outputs if not path.is_file())
    result = SimulationResult(
        solver=adapter.name,
        job_name=job.name,
        command=command,
        cwd=workdir,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        elapsed_s=time.perf_counter() - started,
        outputs=outputs,
        missing_outputs=missing,
        timed_out=timed_out,
    )
    if check and not result.succeeded:
        raise SolverExecutionError(result)
    return result
