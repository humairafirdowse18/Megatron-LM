# Nsight CUDA-graph analysis

Use Nsight runs only to explain *why* latency changes. CUDA tracing perturbs
step time, so use `analyze.py` on non-Nsight runs for the latency number quoted
to management.

## Collect paired traces

Use a separate root from the normal latency sweep:

```bash
cd /home/humairafirdo/Megatron-LM/examples/mamba/cg_bench

export BENCH_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/humairafirdo/cg_bench_nsys
PACKED_SFT=1 MODELS=8B ITERS=50 SEQ_LEN=4096 GBS=8 MAX_SEQS=64 \
  NSYS=1 NSYS_START=25 NSYS_END=30 ./sweep.sh
```

This submits graph-off and graph-on runs. Each compute-node run writes both
`*.nsys-rep` and an automatically exported `*.sqlite`. The five traced training
steps are numbered 26 through 30 because Megatron's profile start is a
zero-based pre-iteration index. The harness selects
`--cuda-graph-trace=node`; without it, Nsight's default whole-graph mode omits
replayed node activities, so graph-on kernel counts and durations are not
comparable to graph-off.

## Analyze overall and step by step

The analyzer works on a login node without the Nsight CLI because it reads the
SQLite exports:

```bash
python analyze_nsys.py --bench-root "${BENCH_ROOT}"
```

The first trace-start step is retained in the step CSV but excluded from
aggregate averages because profiler startup commonly perturbs it. Override this
with `--discard-first 0` or another count.

Outputs are under `${BENCH_ROOT}/nsys_analysis/`:

- `nsys_steps.csv`: every profiled step
- `nsys_summary.csv`: totals, mean/median/p90 step latency, and per-step means
- `nsys_comparison.csv`: paired graph-off/on values and deltas
- `nsys_results.json`: machine-readable union, including warnings
- `nsys_summary.md`: concise manager-readable profiling table and caveats

The CPU gap columns measure gaps between launch API calls on each host thread.
The GPU idle gap columns merge overlapping kernels across streams separately
for each GPU, then measure positive gaps between busy intervals. These gaps can
include synchronization or communication waits; they are not automatically
attributable to Python or CUDA launch latency.

Newer Nsight exports put Runtime API names (`cuda*`) and Driver API names
(`cu*`) in the same activity table. The analyzer reports them separately to
avoid silently treating a nested driver call as a second Runtime API launch.
Counts and durations are available both as process-tree aggregates and divided
by the manifest world size in `*_per_gpu_per_step_mean` columns.

For traces created before `train_step_<N>` NVTX ranges were added, the analyzer
can still report one capture-level aggregate row. A new trace is required for a
reliable per-step breakdown.
