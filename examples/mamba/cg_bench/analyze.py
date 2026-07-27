#!/usr/bin/env python3
"""Turn CUDA-graph benchmark runs into a latency comparison.

Reads the run directories produced by bench.sub and answers three questions:

  1. How much wall-clock latency do CUDA graphs remove, per model?
  2. How much of that is eliminated CPU kernel-launch overhead?
  3. How many steps must a job run before graph capture pays for itself?

For mock-data runs, CUDA graphs do not change the amount of GPU work, so the
steady-state difference estimates exposed CPU launch overhead. For packed-SFT
runs, graph replay also pads cu_seqlens to a fixed bucket. Their difference is
reported as net end-to-end speedup instead of being attributed entirely to
launch overhead.

Usage:
    python analyze.py --bench-root /lustre/.../cg_bench
    python analyze.py --bench-root ... --csv results.csv --json results.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

# " [ts] iteration  42/ 120 | ... | elapsed time per iteration (ms): 1234.5 | ..."
RE_ITER = re.compile(r"iteration\s+(\d+)\s*/")
RE_ELAPSED = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.]+)")
RE_TFLOPS = re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)")
RE_LM_LOSS = re.compile(r"lm loss:\s*([0-9.eE+-]+)")
# "    forward-backward .....: (1234.56, 1240.00)"   (--timing-log-option minmax)
RE_TIMER_MINMAX = re.compile(r"^\s*([a-zA-Z0-9\-_]+)\s*\.+:?\s*\(([0-9.]+),\s*([0-9.]+)\)")
# "    forward-backward .....: 1234.56"              (--timing-log-option max)
RE_TIMER_SINGLE = re.compile(r"^\s*([a-zA-Z0-9\-_]+)\s*\.+:?\s*([0-9.]+)\s*$")
RE_CAPTURE = re.compile(r"Time spent in CUDA Graphs capture on rank \d+:\s*([0-9.]+)")

TIMERS_OF_INTEREST = (
    "forward-backward",
    "forward-compute",
    "backward-compute",
    "optimizer",
)


@dataclass
class Run:
    tag: str
    manifest: dict
    iters: list[int] = field(default_factory=list)
    step_ms: list[float] = field(default_factory=list)
    tflops: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    timers: dict[str, list[float]] = field(default_factory=dict)
    capture_s: float | None = None

    @property
    def model(self) -> str:
        return self.manifest.get("model_scale", "?")

    @property
    def data_mode(self) -> str:
        return self.manifest.get("data_mode", "mock")

    @property
    def max_packed_seqs(self) -> int | None:
        if self.data_mode != "packed":
            return None
        value = self.manifest.get("cuda_graph_max_packed_seqs")
        return int(value) if value is not None else None

    @property
    def cg(self) -> bool:
        return bool(self.manifest.get("use_cuda_graph", 0))

    @property
    def tp(self) -> int:
        return int(self.manifest.get("tensor_model_parallel_size", 1))

    def steady(self, warmup: int) -> list[float]:
        """Step times after warmup, so capture and autotuning are excluded."""
        return [t for i, t in zip(self.iters, self.step_ms) if i > warmup]

    def steady_mean(self, warmup: int) -> float | None:
        vals = self.steady(warmup)
        return statistics.mean(vals) if vals else None

    def steady_median(self, warmup: int) -> float | None:
        vals = self.steady(warmup)
        return statistics.median(vals) if vals else None

    def timer_mean(self, name: str, warmup: int) -> float | None:
        vals = self.timers.get(name)
        if not vals:
            return None
        tail = vals[min(warmup, len(vals) - 1):] or vals
        return statistics.mean(tail)

    def cumulative_ms(self, upto: int) -> float | None:
        """Wall time to complete `upto` steps, including capture in early steps."""
        total = 0.0
        seen = 0
        for i, t in zip(self.iters, self.step_ms):
            if i <= upto:
                total += t
                seen += 1
        return total if seen else None


def parse_run(run_dir: Path) -> Run | None:
    manifest_path = run_dir / "manifest.json"
    log_path = run_dir / "stdout.log"
    if not log_path.exists():
        return None

    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            pass

    run = Run(tag=manifest.get("run_tag", run_dir.name), manifest=manifest)

    for line in log_path.read_text(errors="replace").splitlines():
        m_cap = RE_CAPTURE.search(line)
        if m_cap and run.capture_s is None:
            run.capture_s = float(m_cap.group(1))

        m_iter = RE_ITER.search(line)
        m_elapsed = RE_ELAPSED.search(line)
        if m_iter and m_elapsed:
            run.iters.append(int(m_iter.group(1)))
            run.step_ms.append(float(m_elapsed.group(1)))
            m_tf = RE_TFLOPS.search(line)
            if m_tf:
                run.tflops.append(float(m_tf.group(1)))
            m_loss = RE_LM_LOSS.search(line)
            if m_loss:
                try:
                    run.losses.append(float(m_loss.group(1)))
                except ValueError:
                    pass
            continue

        # Timer lines carry no "iteration" token, so check them separately.
        m_t = RE_TIMER_MINMAX.match(line)
        if m_t:
            name = m_t.group(1)
            if name in TIMERS_OF_INTEREST:
                run.timers.setdefault(name, []).append(float(m_t.group(3)))
            continue
        m_t = RE_TIMER_SINGLE.match(line)
        if m_t:
            name = m_t.group(1)
            if name in TIMERS_OF_INTEREST:
                run.timers.setdefault(name, []).append(float(m_t.group(2)))

    return run if run.step_ms else None


def breakeven_step(off: Run, on: Run) -> int | None:
    """Smallest step after which the CG run stays ahead cumulatively.

    Capture happens once, so CG-on normally starts behind and catches up.
    Requiring it to remain ahead avoids reporting a transient early crossing
    before the graph-capture iteration.
    """
    common = sorted(set(off.iters) & set(on.iters))
    deltas: list[tuple[int, float]] = []
    for k in common:
        c_off, c_on = off.cumulative_ms(k), on.cumulative_ms(k)
        if c_off is None or c_on is None:
            continue
        deltas.append((k, c_on - c_off))
    for idx, (k, delta) in enumerate(deltas):
        if delta < 0 and all(later_delta < 0 for _, later_delta in deltas[idx:]):
            return k
    return None


def projected_breakeven_step(off: Run, on: Run, steady_saving_ms: float) -> int | None:
    """Project break-even past the measured run using steady-state savings."""
    if steady_saving_ms <= 0:
        return None
    common = sorted(set(off.iters) & set(on.iters))
    if not common:
        return None
    last = common[-1]
    remaining_ms = (on.cumulative_ms(last) or 0.0) - (off.cumulative_ms(last) or 0.0)
    if remaining_ms <= 0:
        return last
    return last + math.ceil(remaining_ms / steady_saving_ms)


def fmt(v, spec=".1f", dash="-"):
    return dash if v is None else format(v, spec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-root", required=True, type=Path,
                    help="Directory containing the per-run subdirectories.")
    ap.add_argument("--warmup", type=int, default=20,
                    help="Iterations to discard before measuring steady state "
                         "(must exceed capture + cuda_graph_warmup_steps).")
    ap.add_argument("--csv", type=Path, help="Write per-run rows here.")
    ap.add_argument("--json", type=Path, help="Write full results here.")
    args = ap.parse_args()

    if not args.bench_root.is_dir():
        print(f"error: {args.bench_root} is not a directory", file=sys.stderr)
        return 1

    runs = [r for r in (parse_run(d) for d in sorted(args.bench_root.iterdir()) if d.is_dir()) if r]
    if not runs:
        print(f"error: no parseable runs under {args.bench_root}\n"
              f"       expected <run>/stdout.log with 'elapsed time per iteration' lines",
              file=sys.stderr)
        return 1

    print(f"Parsed {len(runs)} run(s) from {args.bench_root}")
    print(f"Steady state = iterations > {args.warmup}\n")

    print("PER-RUN")
    hdr = f"{'run':<41} {'iters':>6} {'step ms':>9} {'median':>9} {'TFLOP/s':>9} {'capture s':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(runs, key=lambda r: (r.data_mode, r.model, r.cg)):
        tf = statistics.mean(r.tflops[args.warmup:]) if len(r.tflops) > args.warmup else None
        print(f"{r.tag:<41} {len(r.step_ms):>6} "
              f"{fmt(r.steady_mean(args.warmup)):>9} "
              f"{fmt(r.steady_median(args.warmup)):>9} "
              f"{fmt(tf):>9} "
              f"{fmt(r.capture_s, '.2f'):>10}")

    # Pair each model's CG-off and CG-on run.
    by_model: dict[tuple[str, str, int, int | None], dict[bool, Run]] = {}
    for r in runs:
        by_model.setdefault((r.data_mode, r.model, r.tp, r.max_packed_seqs), {})[r.cg] = r

    print("\nCUDA GRAPHS ON vs OFF")
    hdr2 = (f"{'mode':<8} {'model':<12} {'tp':>3} {'bucket':>7} "
            f"{'off ms':>9} {'on ms':>9} {'saved ms':>9} {'speedup':>8} {'loss rel':>10} "
            f"{'launch-bound':>13} {'breakeven':>10}")
    print(hdr2)
    print("-" * len(hdr2))

    summary = []
    for (data_mode, model, tp, max_packed_seqs), pair in sorted(
        by_model.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2],
                          -1 if item[0][3] is None else item[0][3]),
    ):
        off, on = pair.get(False), pair.get(True)
        bucket_text = str(max_packed_seqs) if max_packed_seqs is not None else "-"
        if not off or not on:
            have = "CG-on only" if on else "CG-off only"
            print(f"{data_mode:<8} {model:<12} {tp:>3} {bucket_text:>7}   "
                  f"incomplete pair ({have}) - skipping")
            continue

        t_off = off.steady_mean(args.warmup)
        t_on = on.steady_mean(args.warmup)
        if t_off is None or t_on is None:
            continue

        saved = t_off - t_on
        speedup = t_off / t_on if t_on else None
        # GPU work is identical only in mock mode. Packed mode adds fixed-bucket
        # cu_seqlens padding, so report its speedup as a net end-to-end result.
        launch_frac = saved / t_off if data_mode == "mock" and t_off else None
        be = breakeven_step(off, on)
        be_projected = None if be is not None else projected_breakeven_step(off, on, saved)
        be_text = str(be) if be is not None else (f"~{be_projected}" if be_projected else "never")
        launch_text = f"{launch_frac * 100:.1f}%" if launch_frac is not None else "-"
        common_losses = min(len(off.losses), len(on.losses))
        if common_losses:
            loss_rel = max(
                abs(a - b) / max(abs(a), abs(b), 1e-12)
                for a, b in zip(off.losses[:common_losses], on.losses[:common_losses])
            )
        else:
            loss_rel = None

        print(f"{data_mode:<8} {model:<12} {tp:>3} {bucket_text:>7} "
              f"{fmt(t_off):>9} {fmt(t_on):>9} "
              f"{fmt(saved):>9} {fmt(speedup, '.3f'):>8} "
              f"{fmt(loss_rel, '.2e'):>10} {launch_text:>13} "
              f"{be_text:>10}")

        summary.append({
            "data_mode": data_mode, "model": model, "tp": tp,
            "cuda_graph_max_packed_seqs": max_packed_seqs,
            "step_ms_cg_off": t_off, "step_ms_cg_on": t_on,
            "saved_ms": saved, "speedup": speedup,
            "launch_bound_fraction": launch_frac,
            "max_relative_loss_delta": loss_rel,
            "capture_s": on.capture_s,
            "breakeven_step": be,
            "projected_breakeven_step": be_projected,
            "timers_cg_off": {t: off.timer_mean(t, args.warmup) for t in TIMERS_OF_INTEREST},
            "timers_cg_on": {t: on.timer_mean(t, args.warmup) for t in TIMERS_OF_INTEREST},
        })

    if any(s["timers_cg_off"].get("forward-backward") for s in summary):
        print("\nTIMER BREAKDOWN (ms, mean of steady state)")
        hdr3 = f"{'mode':<8} {'model':<12} {'timer':<20} {'off':>9} {'on':>9} {'delta':>9}"
        print(hdr3)
        print("-" * len(hdr3))
        for s in summary:
            for t in TIMERS_OF_INTEREST:
                a, b = s["timers_cg_off"].get(t), s["timers_cg_on"].get(t)
                if a is None and b is None:
                    continue
                d = (a - b) if (a is not None and b is not None) else None
                print(f"{s['data_mode']:<8} {s['model']:<12} {t:<20} "
                      f"{fmt(a):>9} {fmt(b):>9} {fmt(d):>9}")

    print("\nHOW TO READ THIS")
    print("  launch-bound  mock-data estimate of exposed CPU launch overhead.")
    print("                Packed-SFT shows '-' because fixed-bucket padding also")
    print("                changes GPU work; its speedup is the measured net result.")
    print("  breakeven     steps needed before one-time graph capture pays off.")
    print("                '~N' extrapolates beyond the measured run using the")
    print("                steady-state mean; 'never' means no positive saving.")

    if args.csv:
        import csv
        with args.csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run", "data_mode", "model", "tp", "cuda_graph", "iters",
                        "steady_step_ms", "median_step_ms", "tflops_per_gpu", "capture_s"])
            for r in sorted(runs, key=lambda r: (r.data_mode, r.model, r.cg)):
                tf = statistics.mean(r.tflops[args.warmup:]) if len(r.tflops) > args.warmup else ""
                w.writerow([r.tag, r.data_mode, r.model, r.tp, int(r.cg), len(r.step_ms),
                            r.steady_mean(args.warmup), r.steady_median(args.warmup),
                            tf, r.capture_s])
        print(f"\nwrote {args.csv}")

    if args.json:
        payload = {
            "warmup": args.warmup,
            "comparisons": summary,
            "runs": [{
                "tag": r.tag, "data_mode": r.data_mode, "model": r.model,
                "tp": r.tp, "cuda_graph": r.cg,
                "iters": r.iters, "step_ms": r.step_ms, "tflops": r.tflops,
                "losses": r.losses, "capture_s": r.capture_s,
                "timers": r.timers,
            } for r in runs],
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
