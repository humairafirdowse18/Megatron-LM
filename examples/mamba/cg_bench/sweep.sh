#!/bin/bash
#
# -----------------------------------------------------------------------------
# Submit the CUDA-graph benchmark matrix: {model scales} x {CG off, CG on}.
#
# The step-count axis is deliberately NOT a submission axis. Each job logs every
# iteration, so analyze.py reconstructs cumulative-time-vs-steps from one run
# per config. Submitting per step count would multiply GPU hours for data we
# already have.
#
# Usage:
#   ./sweep.sh                    # submit the default 4-model matrix (8 jobs)
#   DRY_RUN=1 ./sweep.sh          # print the sbatch commands, submit nothing
#   MODELS="800M 8B" ./sweep.sh   # restrict the model set
#   NSYS=1 ./sweep.sh             # add Nsight Systems capture to every job
# -----------------------------------------------------------------------------

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ACCOUNT="${ACCOUNT:-coreai_dlalgo_nemorl}"
PARTITION="${PARTITION:-batch}"
# 8B-tp8 is the same weights as 8B at higher tensor parallelism. Including both
# separates "bigger model" from "smaller per-GPU kernels" as causes of any
# change in CUDA-graph benefit.
MODELS="${MODELS:-800M 2B 8B 8B-tp8}"
ITERS="${ITERS:-120}"
SEQ_LEN="${SEQ_LEN:-4096}"
GBS="${GBS:-8}"
LR="${LR:-1e-4}"
MIN_LR="${MIN_LR:-1e-5}"
NSYS="${NSYS:-0}"
NSYS_START="${NSYS_START:-60}"
NSYS_END="${NSYS_END:-65}"
NSYS_GRAPH_TRACE="${NSYS_GRAPH_TRACE:-node}"
PACKED_SFT="${PACKED_SFT:-0}"
MAX_SEQS="${MAX_SEQS:-64}"
SFT_DATA_PATH="${SFT_DATA_PATH:-/lustre/fsw/portfolios/coreai/users/pmannan/nemotron_ultra_sft_data/blend_jan21.shuf.packed.10k}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/lustre/fsw/portfolios/coreai/users/pmannan/nemotron_ultra_sft_data/tokenizer}"
DRY_RUN="${DRY_RUN:-0}"
TIME_LIMIT="${TIME_LIMIT:-00:40:00}"

BENCH_ROOT="${BENCH_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/humairafirdo/cg_bench}"
JOBLIST="${BENCH_ROOT}/jobs_$(date +%Y%m%d_%H%M%S).txt"

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${BENCH_ROOT}"
  : > "${JOBLIST}"
fi

echo "Benchmark matrix"
echo "  models : ${MODELS}"
echo "  cg     : off on"
echo "  data   : $([[ "${PACKED_SFT}" == "1" ]] && echo "packed SFT (max_seqs=${MAX_SEQS})" || echo "mock")"
echo "  iters  : ${ITERS}   seq_len: ${SEQ_LEN}   gbs: ${GBS}   nsys: ${NSYS}"
if [[ "${NSYS}" == "1" ]]; then
  echo "  trace  : pre-iteration indices [${NSYS_START}, ${NSYS_END})"
  echo "  graphs : ${NSYS_GRAPH_TRACE}"
fi
echo "  output : ${BENCH_ROOT}"
echo

n=0
for model in ${MODELS}; do
  for cg in 0 1; do
    tag="${model}_cg${cg}"
    exports="ALL,MODEL_SCALE=${model},USE_CUDA_GRAPH=${cg},ITERS=${ITERS},SEQ_LEN=${SEQ_LEN},GBS=${GBS},LR=${LR},MIN_LR=${MIN_LR},NSYS=${NSYS},NSYS_START=${NSYS_START},NSYS_END=${NSYS_END},NSYS_GRAPH_TRACE=${NSYS_GRAPH_TRACE},PACKED_SFT=${PACKED_SFT},MAX_SEQS=${MAX_SEQS},SFT_DATA_PATH=${SFT_DATA_PATH},TOKENIZER_MODEL=${TOKENIZER_MODEL},BENCH_ROOT=${BENCH_ROOT}"

    cmd=(sbatch
         --account="${ACCOUNT}"
         --partition="${PARTITION}"
         --time="${TIME_LIMIT}"
         --job-name="cgbench-${tag}"
         --export="${exports}"
         "${HERE}/bench.sub")

    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '%q ' "${cmd[@]}"; echo
    else
      out="$("${cmd[@]}")"
      echo "${out}  [${tag}]"
      echo "${out} ${tag}" >> "${JOBLIST}"
    fi
    n=$((n + 1))
  done
done

echo
echo "${n} jobs $([[ "${DRY_RUN}" == "1" ]] && echo 'would be submitted' || echo submitted)"
if [[ "${DRY_RUN}" != "1" ]]; then
  echo "job list: ${JOBLIST}"
  echo
  echo "When they finish:"
  if [[ "${NSYS}" == "1" ]]; then
    echo "  python ${HERE}/analyze_nsys.py --bench-root ${BENCH_ROOT}"
  else
    echo "  python ${HERE}/analyze.py --bench-root ${BENCH_ROOT}"
  fi
fi
