# Packed-sequence CUDA graph benchmark — 2026-07-26

## Executive result

On real variable-length packed SFT data, CUDA graphs reduced steady-state mean
training-step latency by **21.4% to 34.9%** across the 800M, 2B, and 8B hybrid
models (**1.27x to 1.54x speedup**).

For the target 8B/TP4 configuration, mean step latency fell from **783.0 ms to
545.3 ms**: **30.4% lower latency / 1.44x speedup**. Mean throughput increased
from **227.6 to 328.2 TFLOP/s/GPU** (**+44.2%**).

This is a preliminary single job per configuration. All six comparison jobs
completed successfully with no skipped or NaN iterations.

## Configuration

- Branch/commit: `sj/cudagraph-packedseq` at `f12d22108`
- Hardware: 1 node, 8x H100 per run
- Models: 800M/TP1, 2B/TP1, and 8B/TP4, BF16
- Sequence/batch: sequence length 4096, MBS=1, GBS=8
- Data: `blend_jan21.shuf.packed.10k` through the real `SFTDataset` packed path
- CUDA graph scope: `mamba attn`
- Fixed cu-seqlens bucket: 64 packed sequences
- Measurement: 40 iterations; iterations 1–20 discarded
- Jobs:
  - 800M: CG off `14411565`; CG on `14411566`
  - 2B: CG off `14411651`; CG on `14411652`
  - 8B: CG off `14411653`; CG on `14411654`
  - 8B frozen-weight correctness: CG off `14411725`; CG on `14411726`

## Steady-state measurements

| Model | TP | Off mean | On mean | Mean latency change | Speedup | Throughput off → on |
|---|---:|---:|---:|---:|---:|---:|
| 800M | 1 | 166.98 ms | 108.73 ms | -34.9% | 1.536x | 81.4 → 134.0 TFLOP/s/GPU |
| 2B | 1 | 232.47 ms | 182.62 ms | -21.4% | 1.273x | 225.8 → 287.2 TFLOP/s/GPU |
| 8B | 4 | 783.05 ms | 545.30 ms | -30.4% | 1.436x | 227.6 → 328.2 TFLOP/s/GPU |

| Model | Off median | On median | Off P90 | On P90 |
|---|---:|---:|---:|---:|
| 800M | 163.20 ms | 93.60 ms | 179.50 ms | 152.40 ms |
| 2B | 232.15 ms | 182.65 ms | 237.80 ms | 184.50 ms |
| 8B | 777.35 ms | 535.90 ms | 792.70 ms | 537.40 ms |

For 8B, the 237.8 ms saving was entirely in forward/backward: forward compute
improved by 109.1 ms and backward compute by 123.9 ms, while optimizer time was
unchanged at about 21.3 ms.

The 800M graph-on distribution was bimodal: 15 of 20 measured steps were at or
below 110 ms, while 4 were above 140 ms. The 8B run had one 731.6 ms outlier
against a roughly 536 ms graph-on mode. This is consistent with pack-dependent
graph replay/fallback behavior, but exact fallback coverage was not instrumented
in these runs.

## Numerical alignment

- In the performance runs, maximum relative per-step loss difference was:
  - 800M: `1.31e-4`
  - 2B: `1.84e-5`
  - 8B: `4.69e-3` (mean `5.80e-4`; first step `1.03e-5`)
- No run reported skipped or NaN iterations.

The 8B maximum occurs at one mid-training step after the two independently
evolving BF16 optimizer trajectories have diverged slightly. A separate
zero-learning-rate 8B pair froze the weights and tested four graph-replay steps:

- Maximum relative replay-step loss difference: `4.78e-6`
- Maximum relative replay-step grad-norm difference: `6.23e-5`
- No skipped or NaN iterations; both jobs exited 0

This supports numerical correctness of the packed CUDA-graph replay path within
normal BF16 tolerance.

## Startup amortization

Megatron reported 0.60–1.17 seconds in CUDA graph capture. Including early graph
setup/autotuning, estimated/observed end-to-end break-even was:

- 800M: about iteration 46
- 2B: about iteration 66
- 8B: iteration 11

## Failure diagnosis and fixes

- Job `14411286` failed after 11 seconds, before Python or CUDA launched.
  `BENCH_ROOT=/lustre/.../cg_bench_smoke2` used `...` literally and `mkdir`
  received permission denied.
- The benchmark launcher now rejects literal placeholder paths and reports a
  complete writable Lustre example.
- Earlier benchmark attempts also found that normal (non-packed) batches
  returned seven fields after packed training added `max_seqlen_tensor` as an
  eighth field. The current `pretrain_mamba.py` worktree fix supplies
  `max_seqlen_tensor=None` on the normal path. Corrected smoke job `14411517`
  completed all 15 iterations with exit code 0.

## Nsight launch-overhead diagnosis

A separate profiling-only 8B/TP4 pair used Nsight Systems node-level CUDA graph
tracing for steps 26–30 (jobs `14413465` off and `14413468` on). Aggregate
averages exclude the first trace-start step and use steps 27–30. These traced
step times are not production latency measurements.

Per GPU, per profiled step:

| Metric | Graph off | Graph on | Change |
|---|---:|---:|---:|
| `cudaLaunchKernel*` Runtime API calls | 7,019 | 2,399 | -65.8% |
| `cudaGraphLaunch*` Runtime API calls | 0 | 224 | +224 |
| All CUDA API calls | 64,922 | 35,114 | -45.9% |
| Summed CUDA API CPU time | 382.1 ms | 309.3 ms | -19.1% |
| Visible GPU kernel executions | 12,109.5 | 12,161.5 | +0.43% |
| Mean GPU inter-work gap | 20.53 us | 4.17 us | -79.7% |

The nearly unchanged GPU kernel count confirms that graph-on did not obtain its
speedup by dropping GPU work. Instead, it replaced about 4,396 net Runtime
launch calls per GPU per step and sharply reduced GPU idle gaps. A
`cudaLaunchKernel*` call averaged 4.62 us off and 4.64 us on; a node-traced
`cudaGraphLaunch*` averaged 110.8 us but amortized many kernel launches.

Nsight exaggerated the wall-time speedup (1,165.7 ms off versus 576.5 ms on,
2.02x) because tracing tens of thousands of explicit graph-off launches adds
much more overhead than tracing graph replay. The manager-facing latency result
therefore remains the uninstrumented 783.0 to 545.3 ms (1.44x).

## Recommended next measurement

Sweep `--cuda-graph-max-packed-seqs` over 64, 128, and 256 and record exact
graph-replay versus eager-fallback counts. A larger bucket should improve replay
coverage, but excessive zero-length cu-seqlens padding can reduce
FlashAttention efficiency. Repeat the winning setting three times and add the
production sequence length/batch configuration before treating the result as a
capacity-planning number.
