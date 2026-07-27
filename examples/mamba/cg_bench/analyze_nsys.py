#!/usr/bin/env python3
"""Analyze Nsight Systems CUDA-graph traces overall and per training step.

The benchmark writes an SQLite export beside each ``.nsys-rep``. This script
uses the explicit ``train_step_<N>`` NVTX ranges emitted by Megatron to report:

* cudaLaunchKernel and cudaGraphLaunch API call count and CPU duration
* all CUDA Runtime API call count and CPU duration
* GPU kernel count and summed kernel duration
* CPU gaps between launch API calls
* GPU idle gaps between merged kernel intervals on each device

Outputs:
  nsys_steps.csv       one row per NVTX-marked training step
  nsys_summary.csv     overall totals and per-step averages for each run
  nsys_comparison.csv  paired CUDA-graph-off/on averages and deltas
  nsys_results.json    all of the above plus trace warnings
  nsys_summary.md      concise, manager-readable profiling summary

If only an ``.nsys-rep`` exists, the script can run ``nsys export`` when the
Nsight CLI is available. Traces made before the step NVTX marker was added get
one capture-level row; their per-step breakdown is intentionally unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


STEP_RE = re.compile(r"(?:^|[^A-Za-z0-9_])train_step_(\d+)(?:$|[^0-9])")
CUDA_KERNEL_LAUNCH_RE = re.compile(r"^cudaLaunchKernel")
CU_KERNEL_LAUNCH_RE = re.compile(r"^cuLaunchKernel")
CUDA_GRAPH_LAUNCH_RE = re.compile(r"^cudaGraphLaunch")
CU_GRAPH_LAUNCH_RE = re.compile(r"^cuGraphLaunch")
NS_PER_MS = 1_000_000.0
NS_PER_US = 1_000.0


@dataclass(frozen=True)
class Event:
    start: int
    end: int
    name: str = ""
    tid: int = 0
    device: int = 0
    stream: int = 0


@dataclass(frozen=True)
class Window:
    step: int | None
    start: int
    end: int
    source: str


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linearly interpolated percentile, with q in [0, 1]."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    fraction = position - low
    return float(ordered[low] * (1.0 - fraction) + ordered[high] * fraction)


def safe_mean(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def duration_ns(event: Event) -> int:
    return max(0, event.end - event.start)


class NsysSQLite:
    """Schema-tolerant reader for the CUDA/NVTX subset of an nsys export."""

    def __init__(self, path: Path):
        self.path = path
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        self.tables = {
            row["name"].lower(): row["name"]
            for row in self.db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        self._columns: dict[str, dict[str, str]] = {}
        self.strings = self._load_strings()

    def close(self) -> None:
        self.db.close()

    def columns(self, table: str) -> dict[str, str]:
        if table not in self._columns:
            rows = self.db.execute(
                f"PRAGMA table_info({quote_ident(table)})"
            ).fetchall()
            self._columns[table] = {row["name"].lower(): row["name"] for row in rows}
        return self._columns[table]

    def _load_strings(self) -> dict[int, str]:
        table = self.tables.get("stringids")
        if table is None:
            for candidate in self.tables.values():
                cols = self.columns(candidate)
                if "id" in cols and "value" in cols and "string" in candidate.lower():
                    table = candidate
                    break
        if table is None:
            return {}
        cols = self.columns(table)
        id_col, value_col = cols.get("id"), cols.get("value")
        if id_col is None or value_col is None:
            return {}
        query = (
            f"SELECT {quote_ident(id_col)} AS id, "
            f"{quote_ident(value_col)} AS value FROM {quote_ident(table)}"
        )
        result = {}
        for row in self.db.execute(query):
            if row["id"] is not None and row["value"] is not None:
                result[int(row["id"])] = str(row["value"])
        return result

    def _tables_with_time(self, predicate) -> list[str]:
        result = []
        for table in self.tables.values():
            if not predicate(table.lower()):
                continue
            cols = self.columns(table)
            if "start" in cols and "end" in cols:
                result.append(table)
        return result

    def runtime_tables(self) -> list[str]:
        exact = self.tables.get("cupti_activity_kind_runtime")
        if exact and "start" in self.columns(exact) and "end" in self.columns(exact):
            return [exact]
        return self._tables_with_time(
            lambda name: "cupti_activity" in name and "runtime" in name
        )

    def kernel_tables(self) -> list[str]:
        return self._tables_with_time(
            lambda name: "cupti_activity" in name and "kernel" in name
        )

    def graph_trace_tables(self) -> list[str]:
        exact = self.tables.get("cupti_activity_kind_graph_trace")
        if exact and "start" in self.columns(exact) and "end" in self.columns(exact):
            return [exact]
        return self._tables_with_time(
            lambda name: "cupti_activity" in name and "graph_trace" in name
        )

    def nvtx_tables(self) -> list[str]:
        exact = self.tables.get("nvtx_events")
        if exact and "start" in self.columns(exact) and "end" in self.columns(exact):
            return [exact]
        return self._tables_with_time(lambda name: "nvtx" in name and "event" in name)

    def _resolve_name(
        self, row: sqlite3.Row, columns: dict[str, str], candidates: Sequence[str]
    ) -> str:
        keys = set(row.keys())
        for candidate in candidates:
            actual = columns.get(candidate.lower())
            if actual is None or actual not in keys:
                continue
            value = row[actual]
            if value is None or value == "":
                continue
            if isinstance(value, str):
                return value
            try:
                string_value = self.strings.get(int(value))
            except (TypeError, ValueError):
                string_value = None
            if string_value is not None:
                return string_value
        return ""

    def events(
        self,
        tables: Sequence[str],
        start: int | None = None,
        end: int | None = None,
        name_candidates: Sequence[str] = (),
    ) -> list[Event]:
        events: list[Event] = []
        for table in tables:
            cols = self.columns(table)
            start_col, end_col = cols["start"], cols["end"]
            selected = [start_col, end_col]
            for candidate in name_candidates:
                actual = cols.get(candidate.lower())
                if actual and actual not in selected:
                    selected.append(actual)
            for optional in (
                "globaltid",
                "threadid",
                "tid",
                "deviceid",
                "device",
                "streamid",
                "stream",
            ):
                actual = cols.get(optional)
                if actual and actual not in selected:
                    selected.append(actual)

            select_sql = ", ".join(quote_ident(column) for column in selected)
            clauses = [f"{quote_ident(end_col)} IS NOT NULL"]
            params: list[int] = []
            if start is not None:
                clauses.append(f"{quote_ident(end_col)} > ?")
                params.append(start)
            if end is not None:
                clauses.append(f"{quote_ident(start_col)} < ?")
                params.append(end)
            query = (
                f"SELECT {select_sql} FROM {quote_ident(table)} "
                f"WHERE {' AND '.join(clauses)}"
            )
            for row in self.db.execute(query, params):
                event_start = int(row[start_col])
                event_end = int(row[end_col])
                if event_end <= event_start:
                    continue
                name = self._resolve_name(row, cols, name_candidates)
                tid = self._integer_value(row, cols, ("globaltid", "threadid", "tid"))
                device = self._integer_value(row, cols, ("deviceid", "device"))
                stream = self._integer_value(row, cols, ("streamid", "stream"))
                events.append(
                    Event(event_start, event_end, name, tid, device, stream)
                )
        events.sort(key=lambda event: (event.start, event.end))
        return events

    @staticmethod
    def _integer_value(
        row: sqlite3.Row, columns: dict[str, str], candidates: Sequence[str]
    ) -> int:
        keys = set(row.keys())
        for candidate in candidates:
            actual = columns.get(candidate)
            if actual is not None and actual in keys and row[actual] is not None:
                try:
                    return int(row[actual])
                except (TypeError, ValueError):
                    pass
        return 0

    def step_windows(self) -> list[Window]:
        nvtx_events = self.events(
            self.nvtx_tables(),
            name_candidates=("text", "textId", "message", "nameId", "name"),
        )
        windows = []
        seen = set()
        for event in nvtx_events:
            match = STEP_RE.search(event.name)
            if not match:
                continue
            key = (int(match.group(1)), event.start, event.end)
            if key in seen:
                continue
            seen.add(key)
            windows.append(Window(key[0], key[1], key[2], "nvtx_step"))
        return sorted(windows, key=lambda window: (window.start, window.step or -1))

    def capture_window(self) -> Window | None:
        bounds = []
        for table in self.runtime_tables() + self.kernel_tables():
            cols = self.columns(table)
            row = self.db.execute(
                f"SELECT MIN({quote_ident(cols['start'])}) AS lo, "
                f"MAX({quote_ident(cols['end'])}) AS hi FROM {quote_ident(table)}"
            ).fetchone()
            if row and row["lo"] is not None and row["hi"] is not None:
                bounds.append((int(row["lo"]), int(row["hi"])))
        if not bounds:
            return None
        return Window(None, min(lo for lo, _ in bounds), max(hi for _, hi in bounds), "capture")


def clipped(events: Iterable[Event], window: Window) -> list[Event]:
    result = []
    for event in events:
        if event.end <= window.start or event.start >= window.end:
            continue
        result.append(
            Event(
                max(event.start, window.start),
                min(event.end, window.end),
                event.name,
                event.tid,
                event.device,
                event.stream,
            )
        )
    return result


def positive_gaps_by_group(
    events: Sequence[Event], group_value
) -> list[int]:
    groups: dict[Any, list[Event]] = defaultdict(list)
    for event in events:
        groups[group_value(event)].append(event)
    gaps = []
    for group_events in groups.values():
        previous_end = None
        for event in sorted(group_events, key=lambda item: (item.start, item.end)):
            if previous_end is not None and event.start > previous_end:
                gaps.append(event.start - previous_end)
            previous_end = (
                event.end if previous_end is None else max(previous_end, event.end)
            )
    return gaps


def merge_kernel_intervals(
    kernels: Sequence[Event],
) -> tuple[int, int, list[int]]:
    """Return device-summed busy ns, device-summed span ns, and idle gaps."""
    by_device: dict[int, list[Event]] = defaultdict(list)
    for kernel in kernels:
        by_device[kernel.device].append(kernel)

    total_busy = 0
    total_span = 0
    all_gaps: list[int] = []
    for device_kernels in by_device.values():
        ordered = sorted(device_kernels, key=lambda event: (event.start, event.end))
        if not ordered:
            continue
        merged: list[tuple[int, int]] = []
        current_start, current_end = ordered[0].start, ordered[0].end
        for kernel in ordered[1:]:
            if kernel.start <= current_end:
                current_end = max(current_end, kernel.end)
            else:
                merged.append((current_start, current_end))
                all_gaps.append(kernel.start - current_end)
                current_start, current_end = kernel.start, kernel.end
        merged.append((current_start, current_end))
        total_busy += sum(end - start for start, end in merged)
        total_span += merged[-1][1] - merged[0][0]
    return total_busy, total_span, all_gaps


def gap_metrics(prefix: str, gaps_ns: Sequence[int]) -> dict[str, Any]:
    gaps_us = [gap / NS_PER_US for gap in gaps_ns]
    return {
        f"{prefix}_count": len(gaps_us),
        f"{prefix}_total_ms": sum(gaps_ns) / NS_PER_MS,
        f"{prefix}_mean_us": safe_mean(gaps_us),
        f"{prefix}_p50_us": percentile(gaps_us, 0.50),
        f"{prefix}_p95_us": percentile(gaps_us, 0.95),
        f"{prefix}_max_us": max(gaps_us) if gaps_us else None,
    }


def event_duration_metrics(prefix: str, events: Sequence[Event]) -> dict[str, Any]:
    durations_ns = [duration_ns(event) for event in events]
    durations_us = [duration / NS_PER_US for duration in durations_ns]
    return {
        f"{prefix}_count": len(events),
        f"{prefix}_cpu_ms": sum(durations_ns) / NS_PER_MS,
        f"{prefix}_cpu_mean_us": safe_mean(durations_us),
        f"{prefix}_cpu_p50_us": percentile(durations_us, 0.50),
        f"{prefix}_cpu_p95_us": percentile(durations_us, 0.95),
        f"{prefix}_cpu_max_us": max(durations_us) if durations_us else None,
    }


def analyze_window(
    window: Window,
    runtime_events: Sequence[Event],
    kernel_events: Sequence[Event],
    graph_gpu_events: Sequence[Event],
) -> dict[str, Any]:
    runtime = clipped(runtime_events, window)
    kernels = clipped(kernel_events, window)
    graph_gpu = clipped(graph_gpu_events, window)
    cuda_kernel_launches = [
        event for event in runtime if CUDA_KERNEL_LAUNCH_RE.match(event.name)
    ]
    cu_kernel_launches = [
        event for event in runtime if CU_KERNEL_LAUNCH_RE.match(event.name)
    ]
    cuda_graph_launches = [
        event for event in runtime if CUDA_GRAPH_LAUNCH_RE.match(event.name)
    ]
    cu_graph_launches = [
        event for event in runtime if CU_GRAPH_LAUNCH_RE.match(event.name)
    ]
    launch_events = (
        cuda_kernel_launches
        + cu_kernel_launches
        + cuda_graph_launches
        + cu_graph_launches
    )
    cpu_gaps = positive_gaps_by_group(launch_events, lambda event: event.tid)
    # In whole-graph trace mode, replayed nodes are omitted and each graph is
    # represented by one GRAPH_TRACE interval. Including both kinds here keeps
    # device busy/idle time meaningful even though kernel counts are incomplete.
    gpu_busy_ns, gpu_span_ns, gpu_gaps = merge_kernel_intervals(
        kernels + graph_gpu
    )

    row = {
        "scope": window.source,
        "step": window.step,
        "window_start_ns": window.start,
        "window_end_ns": window.end,
        "profiled_step_wall_ms": (window.end - window.start) / NS_PER_MS,
        "cuda_runtime_api_count": len(runtime),
        "cuda_runtime_api_cpu_ms": sum(map(duration_ns, runtime)) / NS_PER_MS,
        "gpu_kernel_count": len(kernels),
        "gpu_kernel_duration_ms": sum(map(duration_ns, kernels)) / NS_PER_MS,
        "gpu_kernel_duration_mean_us": (
            safe_mean([duration_ns(kernel) / NS_PER_US for kernel in kernels])
        ),
        "gpu_kernel_duration_p95_us": percentile(
            [duration_ns(kernel) / NS_PER_US for kernel in kernels], 0.95
        ),
        "gpu_kernel_duration_max_us": (
            max(duration_ns(kernel) / NS_PER_US for kernel in kernels)
            if kernels
            else None
        ),
        "gpu_graph_execution_count": len(graph_gpu),
        "gpu_graph_execution_duration_ms": (
            sum(map(duration_ns, graph_gpu)) / NS_PER_MS
        ),
        "gpu_kernel_metrics_complete": int(not graph_gpu),
        "gpu_busy_device_ms": gpu_busy_ns / NS_PER_MS,
        "gpu_active_span_device_ms": gpu_span_ns / NS_PER_MS,
    }
    row.update(event_duration_metrics("cuda_launch_kernel", cuda_kernel_launches))
    row.update(event_duration_metrics("cu_launch_kernel", cu_kernel_launches))
    row.update(event_duration_metrics("cuda_graph_launch", cuda_graph_launches))
    row.update(event_duration_metrics("cu_graph_launch", cu_graph_launches))
    row.update(gap_metrics("cpu_launch_gap", cpu_gaps))
    row.update(gap_metrics("gpu_idle_gap", gpu_gaps))
    return row


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def export_trace(report: Path, sqlite_path: Path, nsys_bin: str) -> tuple[bool, str]:
    command = [
        nsys_bin,
        "export",
        "--type=sqlite",
        "--force-overwrite=true",
        f"--output={sqlite_path}",
        str(report),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return False, f"nsys export failed for {report}: {detail}"
    return sqlite_path.exists(), (
        "" if sqlite_path.exists() else f"nsys did not create {sqlite_path}"
    )


def discover_sqlite(
    run_dir: Path, nsys_bin: str | None
) -> tuple[list[Path], list[str]]:
    warnings = []
    sqlite_paths = sorted(run_dir.glob("*.sqlite"))
    known = {path.stem for path in sqlite_paths}
    for report in sorted(run_dir.glob("*.nsys-rep")):
        sqlite_path = report.with_suffix(".sqlite")
        if sqlite_path.stem in known:
            continue
        if nsys_bin is None:
            warnings.append(
                f"{report}: no SQLite export and `nsys` is unavailable; "
                "run `nsys export --type=sqlite <trace>.nsys-rep` on a node with Nsight"
            )
            continue
        ok, warning = export_trace(report, sqlite_path, nsys_bin)
        if ok:
            sqlite_paths.append(sqlite_path)
            known.add(sqlite_path.stem)
        elif warning:
            warnings.append(warning)
    return sorted(set(sqlite_paths)), warnings


def analyze_trace(
    sqlite_path: Path, run_dir: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings = []
    reader = NsysSQLite(sqlite_path)
    try:
        runtime_tables = reader.runtime_tables()
        kernel_tables = reader.kernel_tables()
        graph_trace_tables = reader.graph_trace_tables()
        if not runtime_tables:
            warnings.append(f"{sqlite_path}: CUDA Runtime activity table not found")
        if not kernel_tables:
            warnings.append(f"{sqlite_path}: CUDA kernel activity table not found")

        windows = reader.step_windows()
        if not windows:
            capture = reader.capture_window()
            if capture is None:
                return [], warnings + [f"{sqlite_path}: no timed CUDA activity found"]
            windows = [capture]
            warnings.append(
                f"{sqlite_path}: no train_step_<N> NVTX ranges; reporting capture "
                "overall only (rerun with the updated Megatron training loop for per-step data)"
            )

        first, last = min(window.start for window in windows), max(
            window.end for window in windows
        )
        runtime_events = reader.events(
            runtime_tables,
            first,
            last,
            name_candidates=("nameId", "name", "apiName", "textId"),
        )
        kernel_events = reader.events(
            kernel_tables,
            first,
            last,
            name_candidates=(
                "shortName",
                "demangledName",
                "mangledName",
                "nameId",
                "name",
            ),
        )
        graph_gpu_events = reader.events(
            graph_trace_tables,
            first,
            last,
            name_candidates=("nameId", "name", "textId"),
        )
        if graph_gpu_events:
            warnings.append(
                f"{sqlite_path}: whole-graph CUDA trace events are present; "
                "gpu_kernel_count/duration exclude replayed graph nodes. "
                "gpu_busy/gap metrics include whole-graph intervals. Rerun with "
                "--cuda-graph-trace=node for comparable kernel counts."
            )

        base = {
            "run_tag": manifest.get("run_tag", run_dir.name),
            "trace": str(sqlite_path),
            "data_mode": manifest.get("data_mode", ""),
            "model_scale": manifest.get("model_scale", ""),
            "use_cuda_graph": int(bool(manifest.get("use_cuda_graph", 0))),
            "tensor_model_parallel_size": manifest.get(
                "tensor_model_parallel_size", ""
            ),
            "pipeline_model_parallel_size": manifest.get(
                "pipeline_model_parallel_size", ""
            ),
            "context_parallel_size": manifest.get(
                "context_parallel_size", ""
            ),
            "seq_length": manifest.get("seq_length", ""),
            "micro_batch_size": manifest.get("micro_batch_size", ""),
            "global_batch_size": manifest.get("global_batch_size", ""),
            "gpus_per_node": manifest.get("gpus_per_node", ""),
            "nnodes": manifest.get("nnodes", ""),
            "cuda_graph_max_packed_seqs": manifest.get(
                "cuda_graph_max_packed_seqs", ""
            ),
            "slurm_job_id": manifest.get("slurm_job_id", ""),
            "nsys_graph_trace": manifest.get("nsys_graph_trace", ""),
        }
        rows = []
        for window in windows:
            row = dict(base)
            row.update(
                analyze_window(
                    window, runtime_events, kernel_events, graph_gpu_events
                )
            )
            rows.append(row)
        return rows, warnings
    finally:
        reader.close()


TOTAL_FIELDS = (
    "cuda_launch_kernel_count",
    "cuda_launch_kernel_cpu_ms",
    "cu_launch_kernel_count",
    "cu_launch_kernel_cpu_ms",
    "cuda_graph_launch_count",
    "cuda_graph_launch_cpu_ms",
    "cu_graph_launch_count",
    "cu_graph_launch_cpu_ms",
    "cuda_runtime_api_count",
    "cuda_runtime_api_cpu_ms",
    "gpu_kernel_count",
    "gpu_kernel_duration_ms",
    "gpu_graph_execution_count",
    "gpu_graph_execution_duration_ms",
    "gpu_busy_device_ms",
    "gpu_active_span_device_ms",
    "cpu_launch_gap_count",
    "cpu_launch_gap_total_ms",
    "gpu_idle_gap_count",
    "gpu_idle_gap_total_ms",
)

MEAN_FIELDS = (
    "profiled_step_wall_ms",
    "cuda_launch_kernel_count",
    "cuda_launch_kernel_cpu_ms",
    "cuda_launch_kernel_cpu_mean_us",
    "cuda_launch_kernel_cpu_p50_us",
    "cuda_launch_kernel_cpu_p95_us",
    "cuda_launch_kernel_cpu_max_us",
    "cu_launch_kernel_count",
    "cu_launch_kernel_cpu_ms",
    "cu_launch_kernel_cpu_mean_us",
    "cu_launch_kernel_cpu_p50_us",
    "cu_launch_kernel_cpu_p95_us",
    "cu_launch_kernel_cpu_max_us",
    "cuda_graph_launch_count",
    "cuda_graph_launch_cpu_ms",
    "cuda_graph_launch_cpu_mean_us",
    "cuda_graph_launch_cpu_p50_us",
    "cuda_graph_launch_cpu_p95_us",
    "cuda_graph_launch_cpu_max_us",
    "cu_graph_launch_count",
    "cu_graph_launch_cpu_ms",
    "cu_graph_launch_cpu_mean_us",
    "cu_graph_launch_cpu_p50_us",
    "cu_graph_launch_cpu_p95_us",
    "cu_graph_launch_cpu_max_us",
    "cuda_runtime_api_count",
    "cuda_runtime_api_cpu_ms",
    "gpu_kernel_count",
    "gpu_kernel_duration_ms",
    "gpu_kernel_duration_mean_us",
    "gpu_kernel_duration_p95_us",
    "gpu_kernel_duration_max_us",
    "gpu_graph_execution_count",
    "gpu_graph_execution_duration_ms",
    "gpu_busy_device_ms",
    "gpu_active_span_device_ms",
    "cpu_launch_gap_mean_us",
    "cpu_launch_gap_p95_us",
    "cpu_launch_gap_max_us",
    "gpu_idle_gap_mean_us",
    "gpu_idle_gap_p95_us",
    "gpu_idle_gap_max_us",
)

PER_GPU_FIELDS = (
    "cuda_launch_kernel_count",
    "cuda_launch_kernel_cpu_ms",
    "cu_launch_kernel_count",
    "cu_launch_kernel_cpu_ms",
    "cuda_graph_launch_count",
    "cuda_graph_launch_cpu_ms",
    "cu_graph_launch_count",
    "cu_graph_launch_cpu_ms",
    "cuda_runtime_api_count",
    "cuda_runtime_api_cpu_ms",
    "gpu_kernel_count",
    "gpu_kernel_duration_ms",
    "gpu_graph_execution_count",
    "gpu_graph_execution_duration_ms",
    "gpu_busy_device_ms",
    "gpu_active_span_device_ms",
)


def summarize_runs(step_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    all_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in step_rows:
        all_groups[row["run_tag"]].append(row)

    summaries = []
    for run_tag, all_rows in sorted(all_groups.items()):
        first = all_rows[0]
        rows = [row for row in all_rows if row.get("included_in_summary", 1)]
        if not rows:
            rows = all_rows
        all_step_scopes = [
            row for row in all_rows if row["scope"] == "nvtx_step"
        ]
        step_scopes = [row for row in rows if row["scope"] == "nvtx_step"]
        summary = {
            key: first[key]
            for key in (
                "run_tag",
                "data_mode",
                "model_scale",
                "use_cuda_graph",
                "tensor_model_parallel_size",
                "pipeline_model_parallel_size",
                "context_parallel_size",
                "seq_length",
                "micro_batch_size",
                "global_batch_size",
                "gpus_per_node",
                "nnodes",
                "cuda_graph_max_packed_seqs",
                "slurm_job_id",
                "nsys_graph_trace",
            )
        }
        summary["trace_count"] = len({row["trace"] for row in all_rows})
        summary["analysis_windows_total"] = len(all_rows)
        summary["analysis_windows"] = len(rows)
        summary["profiled_steps_total"] = len(all_step_scopes)
        summary["profiled_steps"] = len(step_scopes)
        summary["discarded_trace_start_steps"] = len(all_step_scopes) - len(
            step_scopes
        )
        summary["step_breakdown_available"] = int(bool(all_step_scopes))
        summary["gpu_kernel_metrics_complete"] = int(
            all(bool(row.get("gpu_kernel_metrics_complete")) for row in rows)
        )
        for field in TOTAL_FIELDS:
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            summary[f"{field}_total"] = sum(values)
        for field in MEAN_FIELDS:
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            summary[f"{field}_per_step_mean"] = safe_mean(values)
        try:
            world_size = int(summary["gpus_per_node"]) * int(summary["nnodes"])
        except (TypeError, ValueError):
            world_size = 1
        summary["world_size"] = max(1, world_size)
        for field in PER_GPU_FIELDS:
            value = summary.get(f"{field}_per_step_mean")
            summary[f"{field}_per_gpu_per_step_mean"] = (
                float(value) / summary["world_size"] if value is not None else None
            )
        wall_values = [
            float(row["profiled_step_wall_ms"])
            for row in rows
            if row.get("profiled_step_wall_ms") is not None
        ]
        summary["profiled_step_wall_ms_median"] = percentile(wall_values, 0.50)
        summary["profiled_step_wall_ms_p90"] = percentile(wall_values, 0.90)
        cpu_gap_count = summary["cpu_launch_gap_count_total"]
        gpu_gap_count = summary["gpu_idle_gap_count_total"]
        summary["cpu_launch_gap_pooled_mean_us"] = (
            summary["cpu_launch_gap_total_ms_total"] * 1000.0 / cpu_gap_count
            if cpu_gap_count
            else None
        )
        kernel_launch_count = summary["cuda_launch_kernel_count_total"]
        cu_kernel_launch_count = summary["cu_launch_kernel_count_total"]
        graph_launch_count = summary["cuda_graph_launch_count_total"]
        cu_graph_launch_count = summary["cu_graph_launch_count_total"]
        gpu_kernel_count = summary["gpu_kernel_count_total"]
        summary["cuda_launch_kernel_cpu_pooled_mean_us"] = (
            summary["cuda_launch_kernel_cpu_ms_total"] * 1000.0
            / kernel_launch_count
            if kernel_launch_count
            else None
        )
        summary["cuda_graph_launch_cpu_pooled_mean_us"] = (
            summary["cuda_graph_launch_cpu_ms_total"] * 1000.0
            / graph_launch_count
            if graph_launch_count
            else None
        )
        summary["cu_launch_kernel_cpu_pooled_mean_us"] = (
            summary["cu_launch_kernel_cpu_ms_total"] * 1000.0
            / cu_kernel_launch_count
            if cu_kernel_launch_count
            else None
        )
        summary["cu_graph_launch_cpu_pooled_mean_us"] = (
            summary["cu_graph_launch_cpu_ms_total"] * 1000.0
            / cu_graph_launch_count
            if cu_graph_launch_count
            else None
        )
        summary["gpu_kernel_duration_pooled_mean_us"] = (
            summary["gpu_kernel_duration_ms_total"] * 1000.0 / gpu_kernel_count
            if gpu_kernel_count
            else None
        )
        summary["gpu_idle_gap_pooled_mean_us"] = (
            summary["gpu_idle_gap_total_ms_total"] * 1000.0 / gpu_gap_count
            if gpu_gap_count
            else None
        )
        summaries.append(summary)
    return summaries


def config_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        row.get(field)
        for field in (
            "data_mode",
            "model_scale",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "seq_length",
            "micro_batch_size",
            "global_batch_size",
            "cuda_graph_max_packed_seqs",
            "gpus_per_node",
            "nnodes",
        )
    )


def ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def compare_runs(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = defaultdict(dict)
    for summary in summaries:
        pairs[config_key(summary)][int(summary["use_cuda_graph"])] = summary

    comparisons = []
    for key, states in sorted(pairs.items(), key=lambda item: str(item[0])):
        if 0 not in states or 1 not in states:
            continue
        off, on = states[0], states[1]
        off_wall = off.get("profiled_step_wall_ms_per_step_mean")
        on_wall = on.get("profiled_step_wall_ms_per_step_mean")
        row = {
            "data_mode": key[0],
            "model_scale": key[1],
            "tensor_model_parallel_size": key[2],
            "pipeline_model_parallel_size": key[3],
            "context_parallel_size": key[4],
            "seq_length": key[5],
            "micro_batch_size": key[6],
            "global_batch_size": key[7],
            "cuda_graph_max_packed_seqs": key[8],
            "graph_off_run_tag": off["run_tag"],
            "graph_on_run_tag": on["run_tag"],
            "profiled_step_ms_off": off_wall,
            "profiled_step_ms_on": on_wall,
            "profiled_step_speedup": ratio(off_wall, on_wall),
            "profiled_step_delta_ms": (
                float(off_wall) - float(on_wall)
                if off_wall is not None and on_wall is not None
                else None
            ),
            "gpu_kernel_metrics_comparable": int(
                bool(off.get("gpu_kernel_metrics_complete"))
                and bool(on.get("gpu_kernel_metrics_complete"))
            ),
        }
        metrics = (
            "cuda_launch_kernel_count",
            "cuda_launch_kernel_cpu_ms",
            "cuda_launch_kernel_cpu_mean_us",
            "cu_launch_kernel_count",
            "cu_launch_kernel_cpu_ms",
            "cu_launch_kernel_cpu_mean_us",
            "cuda_graph_launch_count",
            "cuda_graph_launch_cpu_ms",
            "cuda_graph_launch_cpu_mean_us",
            "cu_graph_launch_count",
            "cu_graph_launch_cpu_ms",
            "cu_graph_launch_cpu_mean_us",
            "cuda_runtime_api_count",
            "cuda_runtime_api_cpu_ms",
            "gpu_kernel_count",
            "gpu_kernel_duration_ms",
            "gpu_kernel_duration_mean_us",
            "gpu_graph_execution_count",
            "gpu_graph_execution_duration_ms",
            "cpu_launch_gap_mean_us",
            "gpu_idle_gap_mean_us",
        )
        for metric in metrics:
            field = f"{metric}_per_step_mean"
            row[f"{metric}_off"] = off.get(field)
            row[f"{metric}_on"] = on.get(field)
            kernel_metric_incomplete = metric.startswith("gpu_kernel_") and not row[
                "gpu_kernel_metrics_comparable"
            ]
            if (
                not kernel_metric_incomplete
                and off.get(field) is not None
                and on.get(field) is not None
            ):
                row[f"{metric}_delta_off_minus_on"] = float(off[field]) - float(
                    on[field]
                )
            else:
                row[f"{metric}_delta_off_minus_on"] = None
        comparisons.append(row)
    return comparisons


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_summary(
    summaries: Sequence[dict[str, Any]], comparisons: Sequence[dict[str, Any]]
) -> None:
    print("NSIGHT PROFILE AVERAGES (profiled/traced latency; not production latency)")
    print("Additive call/kernel counts and durations span all traced GPU workers.")
    header = (
        f"{'run':<43} {'steps':>5} {'step ms':>9} {'kern calls':>10} "
        f"{'kern API us':>11} {'graph calls':>11} {'graph API us':>12} "
        f"{'kernels':>9} {'GPU ms':>9} {'GPU gap us':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in summaries:
        print(
            f"{row['run_tag']:<43} "
            f"{row['profiled_steps']:>5} "
            f"{fmt(row.get('profiled_step_wall_ms_per_step_mean')):>9} "
            f"{fmt(row.get('cuda_launch_kernel_count_per_step_mean')):>10} "
            f"{fmt(row.get('cuda_launch_kernel_cpu_pooled_mean_us')):>11} "
            f"{fmt(row.get('cuda_graph_launch_count_per_step_mean')):>11} "
            f"{fmt(row.get('cuda_graph_launch_cpu_pooled_mean_us')):>12} "
            f"{fmt(row.get('gpu_kernel_count_per_step_mean')):>9} "
            f"{fmt(row.get('gpu_kernel_duration_ms_per_step_mean')):>9} "
            f"{fmt(row.get('gpu_idle_gap_pooled_mean_us')):>10}"
        )

    if comparisons:
        print("\nCUDA GRAPH OFF/ON COMPARISON")
        header = (
            f"{'config':<31} {'off ms':>9} {'on ms':>9} {'speedup':>8} "
            f"{'off kern launches':>17} {'on graph launches':>17}"
        )
        print(header)
        print("-" * len(header))
        for row in comparisons:
            config = (
                f"{row['data_mode']}/{row['model_scale']}/"
                f"tp{row['tensor_model_parallel_size']}/s{row['seq_length']}"
            )
            speedup = row.get("profiled_step_speedup")
            print(
                f"{config:<31} "
                f"{fmt(row.get('profiled_step_ms_off')):>9} "
                f"{fmt(row.get('profiled_step_ms_on')):>9} "
                f"{(fmt(speedup, 3) + 'x') if speedup is not None else '-':>8} "
                f"{fmt(row.get('cuda_launch_kernel_count_off')):>17} "
                f"{fmt(row.get('cuda_graph_launch_count_on')):>17}"
            )


def markdown_summary(
    summaries: Sequence[dict[str, Any]],
    comparisons: Sequence[dict[str, Any]],
    warnings: Sequence[str],
) -> str:
    lines = [
        "# Nsight CUDA-graph diagnostic summary",
        "",
        "> Profiling-only data. CUDA tracing perturbs step latency; use uninstrumented "
        "`analyze.py` results for production latency and speedup claims.",
        "",
        "## Per-run averages",
        "",
        "| Run | Steps used/total | Profiled step ms | cudaLaunchKernel / GPU / step "
        "| cudaLaunchKernel CPU µs/call | cudaGraphLaunch / GPU / step "
        "| cudaGraphLaunch CPU µs/call | Visible GPU kernels / GPU / step "
        "| CPU launch gap µs | GPU idle gap µs | Kernel metrics complete |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['run_tag']} "
            f"| {row['profiled_steps']}/{row['profiled_steps_total']} "
            f"| {fmt(row.get('profiled_step_wall_ms_per_step_mean'))} "
            f"| {fmt(row.get('cuda_launch_kernel_count_per_gpu_per_step_mean'))} "
            f"| {fmt(row.get('cuda_launch_kernel_cpu_pooled_mean_us'))} "
            f"| {fmt(row.get('cuda_graph_launch_count_per_gpu_per_step_mean'))} "
            f"| {fmt(row.get('cuda_graph_launch_cpu_pooled_mean_us'))} "
            f"| {fmt(row.get('gpu_kernel_count_per_gpu_per_step_mean'))} "
            f"| {fmt(row.get('cpu_launch_gap_pooled_mean_us'))} "
            f"| {fmt(row.get('gpu_idle_gap_pooled_mean_us'))} "
            f"| {'yes' if row.get('gpu_kernel_metrics_complete') else 'no'} |"
        )

    if comparisons:
        lines.extend(
            [
                "",
                "## Graph-off/on diagnostic comparison",
                "",
                "| Config | Profiled off ms | Profiled on ms | Profiled speedup "
                "| CUDA Runtime API CPU ms/step off → on | GPU kernels comparable |",
                "|---|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in comparisons:
            config = (
                f"{row['data_mode']}/{row['model_scale']}/"
                f"tp{row['tensor_model_parallel_size']}/s{row['seq_length']}"
            )
            runtime_off = row.get("cuda_runtime_api_cpu_ms_off")
            runtime_on = row.get("cuda_runtime_api_cpu_ms_on")
            runtime_pair = f"{fmt(runtime_off)} → {fmt(runtime_on)}"
            lines.append(
                f"| {config} "
                f"| {fmt(row.get('profiled_step_ms_off'))} "
                f"| {fmt(row.get('profiled_step_ms_on'))} "
                f"| {fmt(row.get('profiled_step_speedup'), 3)}x "
                f"| {runtime_pair} "
                f"| {'yes' if row.get('gpu_kernel_metrics_comparable') else 'no'} |"
            )

    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(
        [
            "",
            "The step-wise source data, including discarded trace-start steps, is in "
            "`nsys_steps.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--bench-root",
        required=True,
        type=Path,
        help="Directory containing benchmark run subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: BENCH_ROOT/nsys_analysis).",
    )
    parser.add_argument(
        "--nsys-bin",
        default="nsys",
        help="Nsight CLI used to export missing SQLite files (default: nsys).",
    )
    parser.add_argument(
        "--discard-first",
        type=int,
        default=1,
        help="Keep but exclude this many trace-start NVTX steps per trace from "
        "aggregate averages (default: 1).",
    )
    args = parser.parse_args()

    if not args.bench_root.is_dir():
        print(f"error: {args.bench_root} is not a directory", file=sys.stderr)
        return 2
    if args.discard_first < 0:
        print("error: --discard-first must be non-negative", file=sys.stderr)
        return 2

    output_dir = args.output_dir or args.bench_root / "nsys_analysis"
    nsys_bin = shutil.which(args.nsys_bin)
    warnings = []
    step_rows = []
    run_dirs = [
        path
        for path in sorted(args.bench_root.iterdir())
        if path.is_dir() and (path / "manifest.json").exists()
    ]
    for run_dir in run_dirs:
        manifest = load_manifest(run_dir)
        sqlite_paths, discover_warnings = discover_sqlite(run_dir, nsys_bin)
        warnings.extend(discover_warnings)
        for sqlite_path in sqlite_paths:
            try:
                rows, trace_warnings = analyze_trace(sqlite_path, run_dir, manifest)
                step_positions = [
                    index
                    for index, row in enumerate(rows)
                    if row["scope"] == "nvtx_step"
                ]
                discarded_positions = set(step_positions[: args.discard_first])
                for index, row in enumerate(rows):
                    row["included_in_summary"] = int(
                        row["scope"] != "nvtx_step"
                        or index not in discarded_positions
                    )
                step_rows.extend(rows)
                warnings.extend(trace_warnings)
            except (OSError, sqlite3.DatabaseError, KeyError, ValueError) as error:
                warnings.append(f"{sqlite_path}: analysis failed: {error}")

    if not step_rows:
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
        print(
            f"error: no analyzable Nsight SQLite traces under {args.bench_root}",
            file=sys.stderr,
        )
        return 1

    summaries = summarize_runs(step_rows)
    comparisons = compare_runs(summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "nsys_steps.csv", step_rows)
    write_csv(output_dir / "nsys_summary.csv", summaries)
    write_csv(output_dir / "nsys_comparison.csv", comparisons)
    (output_dir / "nsys_results.json").write_text(
        json.dumps(
            {
                "bench_root": str(args.bench_root),
                "profiling_only": True,
                "discard_first_steps_per_trace": args.discard_first,
                "step_rows": step_rows,
                "summaries": summaries,
                "comparisons": comparisons,
                "warnings": warnings,
            },
            indent=2,
        )
        + "\n"
    )
    (output_dir / "nsys_summary.md").write_text(
        markdown_summary(summaries, comparisons, warnings)
    )

    print_summary(summaries, comparisons)
    print(f"\nWrote analysis to {output_dir}")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
