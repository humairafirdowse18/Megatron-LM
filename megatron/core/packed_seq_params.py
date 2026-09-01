# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor


# Maximum number of packed sequences supported by CUDA graph capture.
# cu_seqlens tensors are padded to this length + 1 for fixed-shape graph inputs.
# Override at runtime with --cuda-graph-max-packed-seqs.
CUDA_GRAPH_MAX_PACKED_SEQS: int = 2048


# Module-level cache for shared CUDA graph buffer tensors.
# Key includes tag, token/sequence capacity, CP/capture mode, and device.
# Values are shared buffer dicts (or a tensor for seq_idx).
# All layers with the same key share the SAME dict and SAME underlying tensor objects.
# Updating these tensors once per micro-batch propagates to ALL layers' CUDA graphs.
_CG_SHARED_BUFFERS: dict = {}


@dataclass
class PackedSeqParams:
    '''
    Parameters for TEDotProductAttention and fused rope kernels for the
    `thd` (packed) sequence format.
    '''

    qkv_format: str = None
    cu_seqlens_q: Tensor = None
    cu_seqlens_kv: Tensor = None
    cu_seqlens_q_padded: Tensor = None
    cu_seqlens_kv_padded: Tensor = None
    max_seqlen_q: int = None
    max_seqlen_kv: int = None
    # Tensor versions of max_seqlen for CUDA graph buffer updates (avoids int->tensor inside CG).
    max_seqlen_q_tensor: Tensor = None
    max_seqlen_kv_tensor: Tensor = None
    local_cp_size: int = None
    cp_group: dist.ProcessGroup = None
    total_tokens: int = None
    # Pre-computed seq_idx for Mamba. When set, mamba_mixer reads it directly,
    # avoiding dynamic allocations that are forbidden inside CUDA graph capture.
    seq_idx: Optional[Tensor] = None
    tokens_per_sample: int = None
    pad_between_seqs: bool = None
    cp_scatter_cache: object = None

    def __post_init__(self):
        """Pre-compute seq_idx for Mamba mixer.

        Converts cu_seqlens into a per-token sequence index tensor. For example,
        cu_seqlens=[0, 5, 7, 11] with total_tokens=16 produces:
        [0,0,0,0,0, 1,1, 2,2,2,2, 3,3,3,3,3]

        An extra sequence index is appended for tokens beyond the last cu_seqlens entry.
        """
        if self.seq_idx is not None:
            return  # Already set (e.g. CG dummy PSP with pre-allocated buffer)

        cu_seqlens = self.cu_seqlens_q
        if isinstance(cu_seqlens, Tensor) and self.total_tokens is not None:
            # Skip seq_idx computation when cu_seqlens has been CG-padded.
            # CG-padded cu_seqlens contain entries at the global seq_len
            # (e.g. 262144) while total_tokens is CP-local (e.g. 8192).
            # In CG mode, seq_idx is managed separately by mamba_layer.py's
            # _te_cuda_graph_replay via shared CG buffers.
            if cu_seqlens[-1] > self.total_tokens:
                return  # CG-padded: skip, let mamba_layer handle seq_idx

            total_tokens_tensor = torch.tensor(
                [self.total_tokens], dtype=cu_seqlens.dtype, device=cu_seqlens.device
            )
            cu_seqlens_with_max = torch.cat([cu_seqlens, total_tokens_tensor])
            seq_lengths = cu_seqlens_with_max[1:] - cu_seqlens_with_max[:-1]
            # Example: [5, 2, 4, 5] -> [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]
            self.seq_idx = (
                torch.repeat_interleave(
                    torch.arange(seq_lengths.numel(), device=cu_seqlens.device), seq_lengths, output_size=self.total_tokens
                )
                .to(torch.int32)
                .unsqueeze(0)  # Add a batch dimension
            )

    # ----------------------------------------------------------------
    # CUDA graph padding utilities
    # ----------------------------------------------------------------

    @staticmethod
    def pad_cu_seqlens(cu_seqlens: Tensor, target_len: int) -> Tensor:
        """Pad cu_seqlens to a fixed length using the last element as fill value.

        CUDA graphs require fixed-shape inputs. By padding cu_seqlens to a
        constant size (bucket_size + 1), the graph captures a single shape and
        replays it for all batches that fit within the bucket.
        """
        actual_len = cu_seqlens.shape[0]
        if actual_len >= target_len:
            return cu_seqlens[:target_len]
        padded = cu_seqlens.new_empty(target_len)
        padded[:actual_len] = cu_seqlens
        padded[actual_len:] = cu_seqlens[-1]
        return padded

    def ensure_cg_padded(self, target_len: int) -> None:
        """Lazily compute and cache padded cu_seqlens for CUDA graph replay.

        Called per-layer during CG replay but computes padding only once per
        micro-batch (the PSP object is reused across all layers in the same
        iteration). Subsequent calls are a no-op because the cache is stored
        on the PSP instance itself.
        """
        if getattr(self, '_cg_pad_target', None) == target_len:
            return  # Already cached for this target_len
        self._cg_pad_target = target_len
        self._cg_padded_q = PackedSeqParams.pad_cu_seqlens(self.cu_seqlens_q, target_len)
        self._cg_padded_kv = PackedSeqParams.pad_cu_seqlens(self.cu_seqlens_kv, target_len)
        # TE's THD CUDA-graph interface needs all four offsets to be tensor
        # inputs. When storage has no gaps, the physical offsets are identical
        # to the valid-token offsets; materialize that aliasing before capture
        # or replay instead of representing it with a graph-unsafe None.
        cu_seqlens_q_padded = (
            self.cu_seqlens_q_padded
            if self.cu_seqlens_q_padded is not None
            else self.cu_seqlens_q
        )
        cu_seqlens_kv_padded = (
            self.cu_seqlens_kv_padded
            if self.cu_seqlens_kv_padded is not None
            else self.cu_seqlens_kv
        )
        self._cg_padded_qp = PackedSeqParams.pad_cu_seqlens(cu_seqlens_q_padded, target_len)
        self._cg_padded_kvp = PackedSeqParams.pad_cu_seqlens(
            cu_seqlens_kv_padded, target_len
        )

    # ----------------------------------------------------------------
    # Shared CUDA graph buffer management
    # ----------------------------------------------------------------

    @classmethod
    def get_or_create_shared_cg_buffers(
        cls,
        seq_length: int,
        max_seqs: int,
        device: torch.device,
        *,
        context_parallel_size: int = 1,
        partition_for_attention: bool = False,
        tag: str = 'attn',
    ) -> Dict[str, Tensor]:
        """Return the shared PSP buffer dict for CUDA graph replay.

        Layers with the same tag, capacities, capture mode, and device share
        the SAME dict object and therefore the SAME underlying tensor objects.
        Updating the tensors once per micro-batch (via copy_()) propagates to
        all layers' captured CUDA graphs simultaneously.
        """
        key = (
            tag,
            seq_length,
            max_seqs,
            context_parallel_size,
            partition_for_attention,
            int(device.index or 0),
        )
        if key not in _CG_SHARED_BUFFERS:
            _, buffers = cls.create_dummy_for_cuda_graph(
                seq_length,
                max_seqs=max_seqs,
                context_parallel_size=context_parallel_size,
                partition_for_attention=partition_for_attention,
            )
            # Object-identity gate; None forces first update.
            buffers['_last_updated_psp'] = None
            _CG_SHARED_BUFFERS[key] = buffers
        return _CG_SHARED_BUFFERS[key]

    @classmethod
    def get_or_create_shared_seq_idx_buffer(
        cls, total_tokens: int, device: torch.device
    ) -> Tensor:
        """Return the shared seq_idx buffer tensor for Mamba CUDA graph replay."""
        key = ('seq_idx', total_tokens, int(device.index or 0))
        if key not in _CG_SHARED_BUFFERS:
            _CG_SHARED_BUFFERS[key] = torch.zeros(
                1, total_tokens, dtype=torch.int32, device=device
            )
        return _CG_SHARED_BUFFERS[key]

    @classmethod
    def create_dummy_for_cuda_graph(
        cls,
        seq_length: int,
        max_seqs: int = CUDA_GRAPH_MAX_PACKED_SEQS,
        context_parallel_size: int = 1,
        partition_for_attention: bool = False,
    ) -> Tuple[PackedSeqParams, Dict[str, Tensor]]:
        """Create a dummy PackedSeqParams for CUDA graph capture.

        Returns the dummy PSP and a dict of tensor buffer references that can
        be updated via copy_() during graph replay.
        """
        assert max_seqs > 0, "CUDA graph packed-sequence capacity must be positive"
        effective_max_seqs = max_seqs
        if partition_for_attention:
            cp_alignment = 2 * context_parallel_size
            assert seq_length % cp_alignment == 0, (
                f"Packed CUDA graph sequence length ({seq_length}) must be divisible by "
                f"2 * context_parallel_size ({cp_alignment})."
            )
            aligned_token_units = seq_length // cp_alignment
            # Every physical THD attention sequence must occupy at least one
            # CP-alignment unit, so a larger metadata capacity cannot describe
            # a real attention batch.
            effective_max_seqs = min(max_seqs, aligned_token_units)
        cu_seqlens_len = effective_max_seqs + 1
        device = torch.cuda.current_device()
        dtype = torch.int32

        if partition_for_attention:
            # The attention capture sample must itself be a valid THD batch.
            # Partition token capacity into positive, CP-aligned sequences
            # instead of a single maximum-length sequence followed by zeros.
            positive_sequences = effective_max_seqs
            base_units, extra_units = divmod(aligned_token_units, positive_sequences)
            dummy_lengths = torch.full(
                (positive_sequences,),
                base_units * cp_alignment,
                dtype=dtype,
                device=device,
            )
            if extra_units:
                dummy_lengths[:extra_units] += cp_alignment

            cu_seqlens_q = torch.full(
                (cu_seqlens_len,), seq_length, dtype=dtype, device=device
            )
            cu_seqlens_q[0] = 0
            cu_seqlens_q[1 : positive_sequences + 1] = torch.cumsum(dummy_lengths, dim=0)
            cu_seqlens_kv = cu_seqlens_q.clone()
            cu_seqlens_q_padded = cu_seqlens_q.clone()
            cu_seqlens_kv_padded = cu_seqlens_q.clone()
        else:
            # Preserve the established generic/Mamba capture contract.
            cu_seqlens_q = torch.zeros(cu_seqlens_len, dtype=dtype, device=device)
            cu_seqlens_q[1:] = seq_length
            cu_seqlens_kv = cu_seqlens_q.clone()
            cu_seqlens_q_padded = cu_seqlens_q.clone()
            cu_seqlens_kv_padded = cu_seqlens_q.clone()
        max_seqlen_q_tensor = torch.tensor([seq_length], dtype=dtype, device=device)
        max_seqlen_kv_tensor = torch.tensor([seq_length], dtype=dtype, device=device)

        psp = cls(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            max_seqlen_q=seq_length,
            max_seqlen_kv=seq_length,
            max_seqlen_q_tensor=max_seqlen_q_tensor,
            max_seqlen_kv_tensor=max_seqlen_kv_tensor,
            # Keep this graph-static. Letting TE infer it with torch.equal()
            # would synchronize CUDA tensors with the CPU during capture.
            pad_between_seqs=True,
        )
        buffers = {
            'cu_seqlens_q': cu_seqlens_q,
            'cu_seqlens_kv': cu_seqlens_kv,
            'cu_seqlens_q_padded': cu_seqlens_q_padded,
            'cu_seqlens_kv_padded': cu_seqlens_kv_padded,
            'max_seqlen_q_tensor': max_seqlen_q_tensor,
            'max_seqlen_kv_tensor': max_seqlen_kv_tensor,
        }
        return psp, buffers
