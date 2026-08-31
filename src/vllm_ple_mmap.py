"""vllm_ple_mmap — serve the Qwen3.8-Flash-Next N-gram (PLE) table from NVMe via mmap.

Why: the 51B-parameter n-gram table is 44 GiB in FP8 and vLLM keeps it resident
(GPU, or pinned host RAM with VLLM_PLE_CPU_OFFLOAD). On a DGX Spark / GX10 the
host and the GPU share one 121 GiB pool, so neither fits next to the 78 GiB main
model. But a token only ever touches 16 rows x 160 bytes of that table, so the
table can live on disk and be served through the page cache — exactly what
llama.cpp does with its GGUF mmap.

How: with VLLM_PLE_MMAP=1 this module patches ``Qwen3_8FlashNextNGramEmbedding``:
  * ``__init__`` swaps the 44/95 GiB ``VocabParallelEmbedding`` for a tiny
    placeholder whose ``forward(ids)`` gathers rows from ``np.memmap`` views of the
    checkpoint's ``model-plefp8-*.safetensors`` shards (zero-copy, page-cache backed);
  * ``load_weights`` drops the 128 shard tensors on the floor, keeps the global FP8
    ``weight_scale`` (as ``_offload_weight_scale``, which the untouched
    ``Qwen3_8FlashNextPLELayer._dequantize_embeddings`` already consumes) and opens
    the memmaps.
  * ``forward_impl`` (hashing + lookup) is wrapped in a custom op
    ``vllm::ple_mmap_lookup`` so that torch.compile treats it as opaque — the
    stock version trips an Inductor int64 indexing assert on sm_121. With
    ``VLLM_PLE_GPU_GATHER=1`` (the v2 recipes) the op must NOT be listed in
    ``-cc.splitting_ops``: decode-sized gathers run as a GPU (ATS) kernel
    inside the piecewise CUDA graphs — that is the point of v2 — and prefill
    runs outside capture as page-cache warm + the same GPU kernel. Only the
    legacy CPU-gather mode (GPU gather off) still needs the op listed as a
    splitting op, because there the gather is CPU work + a pageable H2D copy
    that cannot live inside a capture.
Nothing else in vLLM changes: the n-gram hashing, the short-conv, the dequant path
are the stock ones.

Fast gather hot path (CPU dedup -> persistent pinned staging buffer -> async H2D ->
GPU-side inverse expansion, plus a no-threadpool fast path for decode-sized
batches), bf16/f16 table support, VLLM_PLE_MMAP_DIR and the periodic stats line
were contributed by @Saren-Arterius (github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound).

Knobs (env):
  VLLM_PLE_MMAP=1            enable
  VLLM_PLE_MMAP_WORKERS=32   gather threads (page faults overlap across threads)
  VLLM_PLE_MMAP_CHUNK=2048   rows per gather task
  VLLM_PLE_MMAP_PREWARM=0    1 = stream the whole table once at load to fill the
                             page cache with whatever memory is free (harmless,
                             evictable; ~10 s at 4.7 GB/s)

Install: the Dockerfile copies this file next to vllm and appends
``_ple_mmap_apply(Qwen3_8FlashNextNGramEmbedding)`` to the end of
``vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py``. See the repo README.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import mmap
import os
import re
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("vllm.ple_mmap")

ENV_ENABLE = "VLLM_PLE_MMAP"
ENV_GPU_GATHER = "VLLM_PLE_GPU_GATHER"
ENV_GPU_SO = "VLLM_PLE_GPU_SO"  # path to ple_gpu_gather.so (default: baked into image)
_GPU_SO_DEFAULT = "/opt/ple_gpu_gather.so"
_FP8_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}
# 16-bit tables need no weight_scale: the stock _dequantize_embeddings is a
# no-op for non-FP8 rows.
_TABLE_DTYPES = {
    **_FP8_DTYPES,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
}


def enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "0").lower() in ("1", "true", "yes")


_GPU_GATHER_ON: list = [None]


def gpu_gather_enabled() -> bool:
    # Cached: read per token step on the hot path, and mid-run env mutation
    # was never a supported behavior.
    if _GPU_GATHER_ON[0] is None:
        _GPU_GATHER_ON[0] = os.environ.get(
            ENV_GPU_GATHER, "0"
        ).lower() in ("1", "true", "yes")
    return _GPU_GATHER_ON[0]


_GPU_LIB = None


def _gpu_lib():
    """ctypes handle to ple_gpu_gather.so (loaded once); None if unavailable.

    The .so is compiled at image build (Dockerfile) — nothing is built at
    runtime, so a fresh sparkrun launch needs no prep.
    """
    global _GPU_LIB
    if _GPU_LIB is not None:
        return _GPU_LIB or None
    import ctypes

    path = os.environ.get(ENV_GPU_SO, _GPU_SO_DEFAULT)
    try:
        lib = ctypes.CDLL(path)
        lib.ple_gpu_gather.restype = ctypes.c_int
        lib.ple_gpu_gather.argtypes = [
            ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong,
            ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p,
        ]
        _GPU_LIB = lib
        logger.info("PLE mmap: GPU gather library loaded from %s", path)
    except OSError as exc:
        _GPU_LIB = False
        logger.warning("PLE mmap: GPU gather requested but %s not loadable (%s); "
                       "falling back to CPU gather", path, exc)
    return _GPU_LIB or None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# safetensors header parsing (no dependency on the safetensors package: we need
# raw file offsets, which its Python API does not expose)
# --------------------------------------------------------------------------- #
def parse_safetensors_header(path: str) -> tuple[dict, int]:
    """Return (header_dict, data_start_offset) of a safetensors file."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header, 8 + header_len


class MmapPleTable:
    """Row gather over a table split into ``split_ngram_parts`` shard files.

    ``shards``: {shard_index: (path, absolute_byte_offset, rows)}. Shard ``i``
    holds global rows ``[i*shard_size, i*shard_size + rows)`` (vLLM's
    ``copy_ple_embedding_shard_`` layout).
    """

    def __init__(
        self,
        shards: dict[int, tuple[str, int, int]],
        shard_size: int,
        row_bytes: int,
        torch_dtype: torch.dtype,
        workers: int = 32,
        chunk: int = 2048,
    ) -> None:
        if not shards:
            raise ValueError("no PLE shards")
        self.shard_size = int(shard_size)
        self.row_bytes = int(row_bytes)
        self.torch_dtype = torch_dtype
        self.chunk = max(1, int(chunk))
        self.paths: list[str | None] = [None] * (max(shards) + 1)
        self.mm: list[np.memmap | None] = [None] * (max(shards) + 1)
        self.rows_total = 0
        for idx, (path, offset, rows) in shards.items():
            self.paths[idx] = path
            self.mm[idx] = np.memmap(
                path, dtype=np.uint8, mode="r", offset=offset, shape=(rows, row_bytes)
            )
            self.rows_total += rows
        self.workers = max(1, int(workers))
        self.pool = ThreadPoolExecutor(max_workers=self.workers)
        # Dedicated warm pool (default: same width as the gather pool) so
        # prefill gathers and decode warms never queue behind each other.
        self.warm_workers = max(1, _env_int("VLLM_PLE_WARM_WORKERS", self.workers))
        self.wpool = ThreadPoolExecutor(max_workers=self.warm_workers)
        self.fast_rows = _env_int("VLLM_PLE_MMAP_FAST_ROWS", 512)
        # Highest valid global row id + 1 (the kernel bounds-checks against it).
        self.vocab = max(
            idx * self.shard_size + (mm.shape[0] if mm is not None else 0)
            for idx, mm in enumerate(self.mm)
        )
        # GPU gather (ATS): host virtual addresses of each shard's row region.
        # On GB10 (Addressing Mode: ATS) a CUDA kernel dereferences these
        # pageable addresses directly through the CPU page tables — verified
        # on-box (byte-identical, graph-capturable). Missing shards -> 0; the
        # kernel treats a null base (and any id outside [0, vocab)) as invalid:
        # it zero-fills the row and counts it instead of dereferencing.
        self._base_addrs = [
            (mm.ctypes.data if mm is not None else 0) for mm in self.mm
        ]
        self._bases_dev: torch.Tensor | None = None
        self._counters: torch.Tensor | None = None  # pinned, 1 uint64 (oob)

    def init_gpu(self, device: torch.device) -> None:
        """Eagerly allocate everything gather_gpu needs and prove one launch.

        Runs at model load (outside any CUDA graph capture). The lazy versions
        of these allocations were capture-hostile: cudaHostAlloc and pageable
        H2D copies are illegal during stream capture, and only the incidental
        eager-prefill-first ordering ever hid that. The 1-row test gather also
        verifies at startup that the kernel can actually read the mmap through
        ATS on this device — failing the boot loudly instead of erroring (or
        faulting) per token at serve time on a non-ATS part.
        """
        if _gpu_lib() is None:
            # Fail the boot, not the first capture: with GPU gather requested
            # and no kernel, the CPU fallback would run a D2H sync inside the
            # piecewise CUDA graphs (the shipped recipe does not list the op
            # as a splitting op) and wedge capture at server start.
            raise RuntimeError(
                "PLE mmap: VLLM_PLE_GPU_GATHER=1 but the gather kernel is not "
                "loadable (see warning above); refusing to serve — unset "
                "VLLM_PLE_GPU_GATHER and add vllm::ple_mmap_lookup to "
                "splitting_ops for the CPU-gather mode"
            )
        device = torch.device(device)
        if device.index is None and device.type == "cuda":
            device = torch.device("cuda", torch.cuda.current_device())
        self._shard_bases_dev(device)
        if self._counters is None:
            try:
                self._counters = torch.zeros(1, dtype=torch.int64, pin_memory=True)
            except RuntimeError:
                # No pinning available: run without the counter rather than
                # hand the kernel a pageable pointer for atomicAdd.
                logger.warning(
                    "PLE mmap: pinned memory unavailable; GPU oob counter off"
                )
        if _OOB_COUNT[0] is None:
            # First table wins (single-PLE-layer assumption, same as the
            # warm hook); later layers keep their own counter but only the
            # first is surfaced by _stats_log.
            _OOB_COUNT[0] = self._counters
        first = next(i for i, mm in enumerate(self.mm) if mm is not None)
        test_id = first * self.shard_size
        self.warm(np.array([test_id], dtype=np.int64))
        test_ids = torch.full((1,), test_id, dtype=torch.int64, device=device)
        ref = torch.from_numpy(np.array(self.mm[first][0])).to(device)
        got = self.gather_gpu(test_ids)
        try:
            torch.cuda.synchronize(device)
        except RuntimeError as exc:
            # A non-ATS part dies here with a generic illegal-access error
            # (the kernel dereferenced an unmapped host address) — attach the
            # actionable diagnosis before the context is written off.
            raise RuntimeError(
                "PLE mmap: GPU test gather crashed — this device cannot "
                "dereference pageable host memory (no ATS?); unset "
                "VLLM_PLE_GPU_GATHER"
            ) from exc
        if not torch.equal(got[0], ref):
            raise RuntimeError(
                "PLE mmap: GPU gather test read of row 0 does not match the "
                "mmap contents — ATS access not working on this device; "
                "unset VLLM_PLE_GPU_GATHER"
            )
        logger.info(
            "PLE mmap: GPU gather ready (vocab %d, test row byte-exact)",
            self.vocab,
        )

    def _shard_bases_dev(self, device: torch.device) -> torch.Tensor:
        if self._bases_dev is None or self._bases_dev.device != device:
            self._bases_dev = torch.tensor(
                self._base_addrs, dtype=torch.int64, device=device
            )
        return self._bases_dev

    def gather_gpu(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: int64 CUDA tensor [N] -> uint8 CUDA tensor [N, row_bytes].

        Zero-copy: the kernel reads the mmap'd table through ATS on the current
        stream. No CPU sync, no dedup (torch.unique would break CUDA graph
        capture; duplicate rows are page-cache hits), no pinned staging.
        """
        lib = _gpu_lib()
        assert lib is not None, "GPU gather library not loaded"
        n = ids.numel()
        out = torch.empty((n, self.row_bytes), dtype=torch.uint8, device=ids.device)
        if n == 0:
            return out
        bases = self._shard_bases_dev(ids.device)
        stream = torch.cuda.current_stream(ids.device).cuda_stream
        counters_ptr = (
            self._counters.data_ptr() if self._counters is not None else None
        )
        rc = lib.ple_gpu_gather(
            bases.data_ptr(), self.shard_size, self.vocab, ids.data_ptr(),
            out.data_ptr(), self.row_bytes, n, stream, counters_ptr,
        )
        if rc != 0:
            # A failed launch (InvalidConfiguration/OutOfResources/...) aborts
            # only THIS kernel — the stream stays healthy, so nothing
            # downstream would fail loudly, and `out` was allocated with
            # torch.empty. Zero it (a normal stream op, capture-safe) so a
            # launch failure produces null embeddings instead of uninitialized
            # memory, count it, and log rate-limited — raising here inside a
            # graph capture would wedge the capture unrecoverably. rc is
            # attributable to this launch: the wrapper drains the pending
            # error state immediately before launching (sticky context errors
            # still surface here, and poison everything else too).
            out.zero_()
            n_err = _stats_add("launch_errors", 1)
            if n_err <= 5 or n_err % 1000 == 0:
                logger.warning(
                    "PLE mmap: ple_gpu_gather launch failed with CUDA error "
                    "%d (%d so far); returning zero rows for this launch", rc,
                    n_err,
                )
        return out

    def warm(self, ids: np.ndarray, use_gather_pool: bool = False) -> None:
        """Bring the pages for these rows into the page cache, fast.

        On GB10 the GPU services its own page faults at queue depth ~1
        (measured with iostat: 4096 cold rows = 1,416 ms in-kernel vs 70 ms
        via CPU threads at queue depth ~23 — same read count, same read size).
        So the GPU must never be the thing that triggers disk I/O. Two stages,
        both measured on-box (32k cold rows: 481 ms CPU gather -> 124 ms
        warm + 5 ms GPU gather):
          1. madvise(MADV_WILLNEED) per unique page: queues async kernel
             readahead for everything (~86 ms for 32k rows). Serial on
             purpose - concurrent madvise contends on the process mmap lock
             and measures slower.
          2. Threadpool touch of 2 bytes per unique row (first+last, covering
             a page straddle) as the completion barrier (~32 ms, mostly
             hitting I/O already in flight).
        """
        ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return
        uniq = np.unique(ids)
        # Warming is advisory: an out-of-range id (possible only from a
        # caller-side bug — production hash ids are in-vocab by construction)
        # must be skipped, not allowed to raise out of a warm thread.
        uniq = uniq[(uniq >= 0) & (uniq < self.vocab)]
        if uniq.size == 0:
            return
        shard = uniq // self.shard_size
        local = uniq - shard * self.shard_size

        page = mmap.PAGESIZE
        for si in np.unique(shard):
            mm = self.mm[int(si)]
            if mm is None:
                continue
            buf = mm._mmap
            adv = buf.madvise
            offs = (local[shard == si] * self.row_bytes) + (mm.offset % page)
            # Length covers the whole row even if it is wider than a page.
            adv_len = ((self.row_bytes // page) + 2) * page
            for st in np.unique((offs // page) * page).tolist():
                adv(mmap.MADV_WILLNEED, int(st), adv_len)
        bounds = np.flatnonzero(np.diff(shard)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [uniq.size]))
        # Two jobs share this method with different pools so they never queue
        # behind each other: prefill-path warms (called with use_gather_pool,
        # since the gather pool is otherwise idle in GPU-gather mode) and the
        # decode hook's warms (the dedicated warm pool).
        pool = self.pool if use_gather_pool else self.wpool
        n_workers = self.workers if use_gather_pool else self.warm_workers
        # Small chunks so the work spreads across every pool worker even for
        # decode-sized batches (self.chunk is tuned for gather's copy cost).
        wchunk = max(64, -(-uniq.size // (n_workers * 4)))
        tasks: list[tuple[int, int, int]] = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            si = int(shard[s])
            for c in range(s, e, wchunk):
                tasks.append((si, c, min(c + wchunk, e)))

        last = self.row_bytes - 1

        def run(task: tuple[int, int, int]) -> int:
            si, a, b = task
            mm = self.mm[si]
            if mm is None:
                raise IndexError(f"PLE shard {si} missing")
            sel = local[a:b]
            return int(mm[sel, 0].sum()) + int(mm[sel, last].sum())

        if len(tasks) == 1:
            run(tasks[0])
        else:
            for _ in pool.map(run, tasks):
                pass

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """ids: int64 [N] global row ids -> uint8 [N, row_bytes] (a fresh array)."""
        import time as _time

        t0 = _time.perf_counter()
        try:
            return self._gather(ids)
        finally:
            _stats_add("gather_ms", (_time.perf_counter() - t0) * 1e3)
            _stats_add("rows", int(np.asarray(ids).size))
            _stats_add("bytes", int(np.asarray(ids).size) * self.row_bytes)

    def _gather(self, ids: np.ndarray) -> np.ndarray:
        ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.empty((0, self.row_bytes), dtype=np.uint8)
        if ids.size <= self.fast_rows:
            # Decode-sized batches: thread-pool dispatch costs more than the
            # reads themselves (~50 tasks for ~65 rows). Gather inline instead.
            if ids.min() < 0 or ids.max() >= self.vocab:
                raise IndexError(
                    f"PLE row id out of range: [{ids.min()}, {ids.max()}] "
                    f"for vocab {self.vocab}"
                )
            shard = ids // self.shard_size
            local = ids - shard * self.shard_size
            out = np.empty((ids.size, self.row_bytes), dtype=np.uint8)
            for si in np.unique(shard):
                mask = shard == si
                out[mask] = self.mm[si][local[mask]]
            return out
        # Dedupe + sort: repeated n-grams are common, and sorted rows improve
        # locality inside a shard.
        uniq, inverse = np.unique(ids, return_inverse=True)
        if uniq[0] < 0 or uniq[-1] >= self.vocab:
            raise IndexError(
                f"PLE row id out of range: [{uniq[0]}, {uniq[-1]}] "
                f"for vocab {self.vocab}"
            )
        shard = uniq // self.shard_size
        local = uniq - shard * self.shard_size
        out = np.empty((uniq.size, self.row_bytes), dtype=np.uint8)

        bounds = np.flatnonzero(np.diff(shard)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [uniq.size]))
        tasks: list[tuple[int, int, int]] = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            si = int(shard[s])
            for c in range(s, e, self.chunk):
                tasks.append((si, c, min(c + self.chunk, e)))

        def run(task: tuple[int, int, int]) -> None:
            si, a, b = task
            mm = self.mm[si]
            if mm is None:
                raise IndexError(f"PLE shard {si} missing")
            # Fancy indexing on a memmap: page faults do the I/O; NumPy releases
            # the GIL for the copy, so tasks overlap across threads.
            out[a:b] = mm[local[a:b]]

        if len(tasks) == 1:
            run(tasks[0])
        else:
            for _ in self.pool.map(run, tasks):
                pass
        return out[inverse]

    def prewarm(self) -> None:
        """Stream every shard once so the page cache holds as much as it can."""
        block = 64 << 20
        buf = bytearray(block)
        view = memoryview(buf)
        for path, mm in zip(self.paths, self.mm):
            if path is None or mm is None:
                continue
            start = mm.offset
            end = start + mm.shape[0] * mm.shape[1]
            with open(path, "rb", buffering=0) as f:
                f.seek(start)
                pos = start
                while pos < end:
                    n = f.readinto(view[: min(block, end - pos)])
                    if not n:
                        break
                    pos += n


# --------------------------------------------------------------------------- #
# Placeholder that stands in for VocabParallelEmbedding
# --------------------------------------------------------------------------- #
class _MmapNgramEmbedding(nn.Module):
    """Duck-types the bits of VocabParallelEmbedding the PLE code reads.

    No ``weight`` attribute on purpose: ``Qwen3_8FlashNextPLELayer`` then falls
    back to ``ple_embedding._offload_weight_scale`` for the FP8 scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.org_vocab_size = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.table: MmapPleTable | None = None
        self._zeros_dtype = torch.bfloat16

    def _pinned_buf(self, rows: int, row_bytes: int) -> torch.Tensor | None:
        """Persistent pinned staging buffer for async H2D (grown as needed)."""
        buf = getattr(self, "_pinned", None)
        if buf is None or buf.shape[0] < rows or buf.shape[1] != row_bytes:
            try:
                cap = max(rows + rows // 2, 4096)
                buf = torch.empty((cap, row_bytes), dtype=torch.uint8, pin_memory=True)
            except RuntimeError:  # no CUDA (CPU tests) or pinning unavailable
                buf = None
            self._pinned = buf
        return buf

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        table = self.table
        if table is None:
            # Weights never loaded (e.g. --load-format dummy): keep the plumbing
            # alive with zeros so kernel tests can run without the 44 GiB table.
            return torch.zeros(
                (*ids.shape, self.embedding_dim),
                dtype=self._zeros_dtype,
                device=ids.device,
            )
        if (
            gpu_gather_enabled()
            and ids.device.type == "cuda"
            and _gpu_lib() is not None
        ):
            # Every read goes through the GPU (ATS) kernel; the branch is on
            # CAPTURE, not size. Inside CUDA graph capture no host sync is
            # possible, so the kernel reads directly — cold rows demand-fault
            # (slow but correct) and the decode-warm hook exists to make that
            # rare. Outside capture (prefill of any size, eager decode) the
            # page cache is warmed first at CPU-fault queue depth ~23, then
            # the same kernel reads warm pages. Measured on-box, 32k cold
            # rows: 481 ms CPU-threadpool gather vs 124 ms warm + 5 ms GPU
            # gather; byte-identical. The kernel bounds-checks every id
            # against [0, vocab) and null shard bases, zero-filling and
            # counting instead of dereferencing.
            if torch.cuda.is_current_stream_capturing():
                rows_dev = table.gather_gpu(ids.reshape(-1).to(torch.int64))
                out = rows_dev.view(table.torch_dtype)
                return out.reshape(*ids.shape, self.embedding_dim)
            import time as _time

            t0 = _time.perf_counter()
            ids_np = ids.detach().to("cpu", non_blocking=False).numpy().reshape(-1)
            # warm() clips out-of-range ids (advisory; must never raise from
            # here), while gather()/the kernel enforce bounds — asymmetric on
            # purpose. With CUDA graphs disabled this eager path also runs
            # for decode steps, adding a per-step D2H sync + a warm the hook
            # already did: correct but redundant — the shipping config
            # (piecewise graphs) never takes it for captured decode shapes.
            table.warm(ids_np, use_gather_pool=True)
            rows_dev = table.gather_gpu(ids.reshape(-1).to(torch.int64))
            _stats_add("gather_ms", (_time.perf_counter() - t0) * 1e3)
            _stats_add("rows", int(ids_np.size))
            _stats_add("bytes", int(ids_np.size) * table.row_bytes)
            out = rows_dev.view(table.torch_dtype)
            return out.reshape(*ids.shape, self.embedding_dim)
        ids_np = ids.detach().to("cpu", non_blocking=False).numpy().reshape(-1)
        # Dedup on CPU, gather only unique rows, expand on the GPU: fewer disk
        # reads AND fewer H2D bytes (repeated n-grams are the common case).
        uniq, inverse = np.unique(ids_np, return_inverse=True)
        rows = table.gather(uniq)  # uint8 [U, row_bytes], fresh & writable
        u = rows.shape[0]
        buf = self._pinned_buf(u, table.row_bytes) if ids.device.type == "cuda" else None
        if buf is not None:
            buf[:u].numpy()[:] = rows
            dev = buf[:u].to(ids.device, non_blocking=True)
        else:
            dev = torch.from_numpy(rows).to(ids.device)
        inv = torch.from_numpy(inverse.reshape(-1)).to(ids.device, non_blocking=True)
        out = dev.view(table.torch_dtype)[inv]
        return out.reshape(*ids.shape, self.embedding_dim)


# --------------------------------------------------------------------------- #
# Patch
# --------------------------------------------------------------------------- #
def host_ngram_ids(emb, ctx: np.ndarray, q_len: int) -> np.ndarray:
    """Host-side replica of ``forward_impl``'s n-gram id computation.

    ``ctx``: int64 [R, W] token rows — each row is the request's last
    ``ngram_size - 1`` committed tokens (EOS-padded on the left, exactly like
    the runner's ngram_context) followed by the tokens of the current step.
    Returns the table row ids for the last ``q_len`` columns: int64
    [R, q_len, ngram_heads].

    Used only to WARM pages ahead of the in-graph GPU gather. The gather
    itself always uses the model's in-graph ids, so a mismatch here can only
    cost speed, never correctness. Buffers (multipliers/sizes/offsets) are the
    checkpoint-loaded tensors off the live embedding, copied to host numpy
    ONCE at model load (``_setup_table``) — this function runs on the warm
    thread and must issue no CUDA calls: a background-thread ``.cpu()`` is a
    D2H copy + sync on that thread's default stream, which both serializes
    against the compute stream every decode step and can invalidate an
    in-flight CUDA graph capture.
    """
    eos, mult, sizes, offs, n, hpn = emb._ple_host_hash
    ctx = np.ascontiguousarray(ctx, dtype=np.int64)
    n_reqs, width = ctx.shape
    positions = np.arange(width, dtype=np.int64)
    eos_pos = np.where(ctx == eos, positions, -1)
    prev_incl = np.maximum.accumulate(eos_pos, axis=1)
    prev = np.concatenate(
        [np.full((n_reqs, 1), -1, np.int64), prev_incl[:, :-1]], axis=1
    )
    pos_in_seg = positions[None, :] - prev - 1
    shifted = [ctx]
    for s in range(1, n):
        src = positions - s
        gathered = ctx[:, np.clip(src, 0, None)]
        valid = (src[None, :] >= 0) & (pos_in_seg >= s)
        shifted.append(np.where(valid, gathered, eos))
    cols = np.arange(width - q_len, width)
    blocks = []
    # int64 multiply wraps silently in NumPy (no errstate involvement) —
    # identical wraparound to the model's int64 tensor math, which is the
    # point: the hash must overflow the same way on both sides.
    for ngram in range(2, n + 1):
        start = (ngram - 2) * hpn
        end = start + hpn
        mixed = shifted[0] * mult[0]
        for i in range(1, ngram):
            mixed = np.bitwise_xor(mixed, shifted[i] * mult[i])
        ids = np.remainder(mixed[:, cols, None], sizes[start:end])
        blocks.append(ids + offs[start:end])
    return np.concatenate(blocks, axis=-1)


def _bool_default(param) -> bool | None:
    """Default of an inspect.Parameter as a probe fallback; None = refuse.
    ANY truthy default — bool True included — would make the omitted-arg
    probe truthy and silently disable warming for the process, so the
    installer bails loudly instead, like the other signature mismatches.
    Any falsy default (False, None, 0, numpy bools) is accepted as False."""
    import inspect

    d = param.default
    if d is inspect.Parameter.empty:
        return False
    try:
        return None if d else False
    except Exception:
        # A default whose truth value raises (e.g. a numpy array) — refuse,
        # never take down model load from a defensive check.
        return None


_DECODE_WARM = {"installed": False}
_WARM_EXEC: list[ThreadPoolExecutor | None] = [None]
# Future of the most recently submitted warm; used for drop-if-busy
# backpressure and failure logging.
_WARM_PENDING: list = [None]


def _decode_warm_step_v2(runner, scheduler_output) -> None:
    """Decode warm for the V2 model runner (vllm.v1.worker.gpu.model_runner).

    Host state differs from V1: tokens live in req_states.all_token_ids, a
    UVA (pinned host) [max_reqs, max_model_len] buffer the GPU appends to in
    place, with num_computed_tokens_np the CPU mirror and req_id_to_index the
    row mapping. Known limit: draft tokens for the upcoming step live in a
    separate GPU tensor and are not hashed here, so their rows are not
    warmed — they demand-fault on first touch and are cached thereafter.
    """
    _stats_add("hook_calls", 1)
    if not _REGISTRY:
        return
    sched = scheduler_output.num_scheduled_tokens
    if not sched:
        return
    rs = runner.req_states
    idx = rs.req_id_to_index
    if not idx:
        # Batches whose request ids are not registered (nothing to warm) —
        # log the first one instead of silently eating it, since this guard
        # also hides hook liveness from the boot log if it always trips.
        if not _DECODE_WARM.get("unregistered_logged"):
            _DECODE_WARM["unregistered_logged"] = True
            logger.info(
                "PLE mmap: decode warm hook invoked with no registered "
                "request ids (%d scheduled) — warmup traffic; nothing warmed",
                len(sched),
            )
        return
    toks = rs.all_token_ids._uva_buf.np
    width = toks.shape[1]
    done = rs.num_computed_tokens_np
    if _DECODE_WARM.get("qmax") is None:
        _DECODE_WARM["qmax"] = _env_int("VLLM_PLE_DECODE_WARM_MAX_Q", 8)
    qmax = _DECODE_WARM["qmax"]
    # num_computed_tokens_np is an optimistic upper bound: it exceeds the
    # authoritative GPU value by up to num_speculative_steps when drafts are
    # rejected. Widening the warmed window left by that bound covers every
    # candidate frame, and the true frame's tokens all lie below the rollback
    # point where the buffer holds correct values.
    spec_k = int(getattr(rs, "num_speculative_steps", 0))
    if spec_k == 0 and not _DECODE_WARM.get("spec_k_checked"):
        _DECODE_WARM["spec_k_checked"] = True
        if not hasattr(rs, "num_speculative_steps"):
            logger.warning(
                "PLE mmap: req_states has no num_speculative_steps — warm "
                "windows are NOT widened for draft rollback (attribute "
                "renamed upstream?)"
            )
    # Single-PLE-layer assumption: windows are built with the first layer's
    # ngram_size/eos and would be WRONG for a second layer with different
    # hash geometry, so warm only the first and say so rather than silently
    # warming garbage for the rest (advisory path — correctness unaffected).
    owners = [
        o for o in _REGISTRY.values()
        if o.ngram_embedding.table is not None
    ]
    if not owners:
        return
    if len(owners) > 1 and not _DECODE_WARM.get("multi_owner_logged"):
        _DECODE_WARM["multi_owner_logged"] = True
        logger.warning(
            "PLE mmap: %d PLE layers registered; decode warming covers only "
            "the first (windows are built with its hash geometry)",
            len(owners),
        )
    owners = owners[:1]
    n_ctx = int(owners[0].ngram_size) - 1
    eos = int(owners[0].eos_token_id)
    groups: dict[int, list[np.ndarray]] = {}
    stale = _STALE_CHECK
    seen_rids = set()
    for rid, q in sched.items():
        i = idx.get(rid)
        q = int(q)
        if i is None or q <= 0 or q > qmax:
            continue
        seen_rids.add(rid)
        q_eff = q + spec_k
        # done[i] is optimistic; near max_model_len it can point past the
        # buffer, and a silent numpy slice-clamp then breaks the broadcast.
        # Clamp explicitly: warming is advisory, a truncated window only
        # costs warm coverage, never correctness.
        end = min(int(done[i]) + q, width)
        window = np.full(n_ctx + q_eff, eos, dtype=np.int64)
        lo = max(0, end - q_eff - n_ctx)
        window[n_ctx + q_eff - (end - lo):] = toks[i, lo:end]
        groups.setdefault(q_eff, []).append(window)
        # Staleness measurement (whether the GPU's token append has landed in
        # the host view by hook time): remember a token we read; next
        # invocation, if the buffer shows a different value at that position,
        # our earlier read was stale. The position is deliberately spec_k
        # below the optimistic count — a slot that cannot be rewritten by
        # draft rollback — so this measures pure host-visibility lag, not the
        # optimistic mirror's own churn.
        last_pos = end - q - 1 - spec_k
        if last_pos >= 0:
            prev = stale.get(rid)
            if prev is not None:
                p_pos, p_val = prev
                _stats_add("stale_checks", 1)
                if int(toks[i, p_pos]) != p_val:
                    _stats_add("stale_hits", 1)
            stale[rid] = (last_pos, int(toks[i, last_pos]))
    # Evict tracking for requests no longer scheduled (finished or preempted)
    # instead of wiping everything at an arbitrary size.
    for rid in [r for r in stale if r not in seen_rids]:
        del stale[rid]
    if _DECODE_WARM.get("first_pending"):
        _DECODE_WARM["first_pending"] = False
        logger.info(
            "PLE mmap: decode warm hook first invocation (V2 runner): "
            "%d scheduled reqs, %d warmable", len(sched),
            sum(len(v) for v in groups.values()),
        )
    _submit_warm(owners, groups)
    # Drive the periodic stats line from here too: under a fully-captured
    # decode steady state the eager op wrapper never runs, and telemetry
    # (liveness, oob, stale) must not depend on prefill traffic to be seen.
    _stats_log()


def _submit_warm(owners: list, groups: dict[int, list]) -> None:
    if not groups:
        return
    # Counted whenever warmable work exists (whether or not it gets dropped):
    # the per-window liveness alarm compares attempts vs completions, so a
    # wedged executor (all drops, zero warms) still registers as dead.
    _stats_add("warm_attempts", 1)
    if _WARM_EXEC[0] is None:
        _WARM_EXEC[0] = ThreadPoolExecutor(max_workers=1)
    # Backpressure: if the previous warm is still running, drop this one.
    # Decode steps outpace warms under load; an unbounded queue grows RSS
    # without limit and warms rows for token windows hundreds of steps stale
    # (wasted NVMe bandwidth). Dropping keeps warms fresh — the next step
    # re-derives windows from current state.
    prev = _WARM_PENDING[0]
    if prev is not None and not prev.done():
        _stats_add("warm_dropped", 1)
        return

    def _hash_and_warm() -> None:
        # Hashing here, off the step's critical path (host numpy only — no
        # CUDA calls on this thread): the synchronous part of the hook is
        # only the window copies above (microseconds).
        for owner in owners:
            ids = np.concatenate([
                host_ngram_ids(owner, np.stack(rows), q).reshape(-1)
                for q, rows in groups.items()
            ])
            owner.ngram_embedding.table.warm(ids)
            # Count AFTER the warm completes: this is the liveness signal,
            # and incrementing intent instead of completion is exactly how a
            # dead hook once reported healthy for a full benchmarked run.
            _stats_add("warmed_rows", int(ids.size))

    fut = _WARM_EXEC[0].submit(_hash_and_warm)
    _WARM_PENDING[0] = fut

    def _log_failure(f) -> None:
        exc = f.exception()
        if exc is not None:
            n = _stats_add("warm_errors", 1)
            # Rate-limited, never a permanent latch: one long/odd request
            # must not silently kill warming for the process lifetime.
            if n <= 5 or n % 1000 == 0:
                logger.warning(
                    "PLE mmap: decode warm failed (%d so far): %s", n, exc,
                )

    fut.add_done_callback(_log_failure)


def _install_decode_warm() -> None:
    """Wrap execute_model on the V2 model runner.

    gpu_worker.py selects between two runner classes at runtime; patching a
    class the deployment doesn't run is exactly the wrong-class bug that left
    the hook installed-but-dead (measured: hook-warmed rows 0 across a full
    serving run). Only the V2 runner (vllm.v1.worker.gpu.model_runner) is
    supported; on a vLLM build without that module the server serves without
    decode warming and says so in the log (the import really is guarded —
    this docstring is checked against the code below). The first-invocation
    line is the liveness proof to check on every boot.
    """
    if _DECODE_WARM["installed"]:
        return
    if not gpu_gather_enabled() or _env_int("VLLM_PLE_DECODE_WARM", 1) != 1:
        logger.info(
            "PLE mmap: decode warming disabled (VLLM_PLE_GPU_GATHER=%s, "
            "VLLM_PLE_DECODE_WARM=%s) — cold decode rows demand-fault",
            os.environ.get(ENV_GPU_GATHER, "0"),
            os.environ.get("VLLM_PLE_DECODE_WARM", "1"),
        )
        return

    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner as _V2
    except ImportError as exc:
        logger.warning(
            "PLE mmap: V2 model runner not importable (%s); serving WITHOUT "
            "decode warming — cold decode rows demand-fault", exc,
        )
        return

    # Verify the wrapper's assumptions against the real signature at install
    # time, not silently at step N: the dummy_run/is_profile probes below key
    # off parameter names and positions, and one upstream refactor otherwise
    # turns this into a silently-dead (or capture-warming) hook.
    import inspect

    orig = _V2.execute_model
    sig_params = inspect.signature(orig).parameters
    params = list(sig_params)
    for name in ("dummy_run", "is_profile"):
        if name not in params:
            logger.warning(
                "PLE mmap: %s.execute_model has no %r parameter "
                "(signature: %s); serving WITHOUT decode warming",
                _V2.__name__, name, params,
            )
            return
        if sig_params[name].kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            logger.warning(
                "PLE mmap: %s.execute_model parameter %r has kind %s the "
                "wrapper cannot probe; serving WITHOUT decode warming",
                _V2.__name__, name, sig_params[name].kind,
            )
            return
    # Positional index of each probe among the caller-supplied args after
    # (self, scheduler_output) — derived from the live signature instead of
    # hard-coded. A keyword-only parameter cannot arrive positionally, so its
    # probe is kwargs-only (index None).
    def _pos(name):
        if sig_params[name].kind is inspect.Parameter.KEYWORD_ONLY:
            return None
        pos = params.index(name) - 2
        if pos < 0:
            # Parameter moved ahead of scheduler_output — a probe at a
            # negative index would read an arbitrary caller argument.
            logger.warning(
                "PLE mmap: %r precedes scheduler_output in execute_model; "
                "probing it by keyword only", name,
            )
            return None
        return pos

    dummy_pos = _pos("dummy_run")
    profile_pos = _pos("is_profile")

    _dummy_default = _bool_default(sig_params["dummy_run"])
    _profile_default = _bool_default(sig_params["is_profile"])
    if _dummy_default is None or _profile_default is None:
        logger.warning(
            "PLE mmap: execute_model parameter default is not falsy "
            "(dummy_run=%r, is_profile=%r) — the omitted-arg probe would "
            "disable warming silently; serving WITHOUT decode warming",
            sig_params["dummy_run"].default, sig_params["is_profile"].default,
        )
        return
    _DECODE_WARM["first_pending"] = True

    def execute_model(self, scheduler_output, *args, **kwargs):
        def _probe(name, pos, default):
            if name in kwargs:
                return kwargs[name]
            if pos is not None and 0 <= pos < len(args):
                return args[pos]
            return default

        dummy = _probe("dummy_run", dummy_pos, _dummy_default)
        profile = _probe("is_profile", profile_pos, _profile_default)
        if not dummy and not profile:
            try:
                _decode_warm_step_v2(self, scheduler_output)
            except Exception as exc:
                # Rate-limited, never a one-shot process-wide latch: one odd
                # request must not silently kill warming forever.
                n = _stats_add("warm_errors", 1)
                if n <= 5 or n % 1000 == 0:
                    logger.warning(
                        "PLE mmap: decode warm hook error (%d so far): %s",
                        n, exc,
                    )
        return orig(self, scheduler_output, *args, **kwargs)

    _V2.execute_model = execute_model
    logger.info(
        "PLE mmap: decode warm hook installed on gpu.model_runner."
        "GPUModelRunner (V2); dummy_run/is_profile at arg positions %s/%s "
        "(None = keyword-only)",
        dummy_pos, profile_pos,
    )
    _DECODE_WARM["installed"] = True


def _find_shards(
    model_path: str, layer_idx: int
) -> tuple[
    dict[int, tuple[str, int, int]],
    int | None,
    str | None,
    tuple[str, int, int, str] | None,
]:
    """Locate ``layers.<idx>.ple.ple_embedding.ngram_embedding.shard_N.weight``.

    Returns (shards, cols, dtype_str, scale_entry) where scale_entry is
    (path, abs_offset, nbytes, dtype) of ``ngram_embedding.weight_scale`` or
    None, and cols is the shard row width (None when no shards were found).
    """
    shard_re = re.compile(
        rf"layers\.{layer_idx}\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$"
    )
    scale_re = re.compile(
        rf"layers\.{layer_idx}\.ple\.ple_embedding\.ngram_embedding\.weight_scale$"
    )
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        files = sorted(
            {
                os.path.join(model_path, fn)
                for name, fn in weight_map.items()
                if shard_re.search(name) or scale_re.search(name)
            }
        )
    else:
        files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

    shards: dict[int, tuple[str, int, int]] = {}
    shard_cols: int | None = None
    dtype_str: str | None = None
    scale_entry: tuple[str, int, int, str] | None = None
    for path in files:
        header, data_start = parse_safetensors_header(path)
        for name, meta in header.items():
            m = shard_re.search(name)
            if m:
                start, end = meta["data_offsets"]
                rows, cols = meta["shape"]
                if dtype_str is None:
                    dtype_str = meta["dtype"]
                elif meta["dtype"] != dtype_str:
                    raise ValueError("PLE shards have mixed dtypes")
                if end - start != rows * cols * _itemsize(dtype_str):
                    raise ValueError(f"PLE shard {name}: size/shape mismatch")
                shards[int(m.group(1))] = (path, data_start + start, rows)
                shard_cols = cols
            elif scale_re.search(name):
                start, end = meta["data_offsets"]
                scale_entry = (path, data_start + start, end - start, meta["dtype"])
    return shards, shard_cols, dtype_str, scale_entry


def _itemsize(dtype_str: str) -> int:
    return {
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "U8": 1,
        "I8": 1,
        "BF16": 2,
        "F16": 2,
        "F32": 4,
    }[dtype_str]


def _read_scale(entry: tuple) -> torch.Tensor:
    path, offset, nbytes, dtype_str = entry
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(nbytes)
    if dtype_str == "F32":
        return torch.tensor(struct.unpack("<f", raw[:4])[0], dtype=torch.float32)
    if dtype_str == "BF16":
        u16 = struct.unpack("<H", raw[:2])[0]
        return torch.tensor(u16 << 16, dtype=torch.int32).view(torch.float32).squeeze()
    if dtype_str == "F16":
        return torch.frombuffer(bytearray(raw[:2]), dtype=torch.float16).clone().squeeze()
    raise ValueError(f"unsupported weight_scale dtype {dtype_str}")


_REGISTRY: dict[str, nn.Module] = {}
_OP_NAME = "ple_mmap_lookup"

# Aggregate gather-overhead stats, logged every VLLM_PLE_MMAP_STATS_SEC seconds
# (0 = off). op_ms covers hashing + gather + H2D; gather_ms just the disk
# reads. calls/op_ms/gather_ms/rows/bytes reset each window; the warm/stale
# counters are cumulative for the process (and labeled so in the log line).
_STATS = {"calls": 0, "op_ms": 0.0, "gather_ms": 0.0, "rows": 0, "bytes": 0,
          "warmed_rows": 0, "stale_checks": 0, "stale_hits": 0,
          "warm_dropped": 0, "warm_errors": 0, "launch_errors": 0,
          "hook_calls": 0, "warm_attempts": 0}
# _STATS is written from the runner thread, the warm worker, and future done
# callbacks; bare `+=` loses counts across threads.
_STATS_LOCK = threading.Lock()


def _stats_add(key: str, val):
    """Locked increment; returns the new value so callers' rate-limit checks
    read the same consistent count they just wrote."""
    with _STATS_LOCK:
        _STATS[key] += val
        return _STATS[key]
# rid -> (position, token value) read at the previous hook invocation; used to
# measure whether GPU token appends had landed in the host view when read.
_STALE_CHECK: dict = {}
# Pinned single counter the gather kernel bumps for out-of-range/missing-shard
# ids (zero-filled, never dereferenced). Allocated by init_gpu at model load.
_OOB_COUNT: list = [None]
# 0.0 here is an equality-tested "no window origin yet" sentinel (never
# compared arithmetically against the boot-relative clock — unlike
# _ALARM_LAST below, where 0.0 would be a bug).
_STATS_LAST = [0.0]
# monotonic() of the last emitted health warning (wall-clock rate limit).
# -inf = never warned: monotonic() is time since BOOT, so a 0.0 sentinel
# would suppress the first fault on any host with under 60 s uptime.
_ALARM_LAST = [float("-inf")]
# Cumulative counters as of the last emitted stats line — the base for the
# per-window deltas. Order: warmed_rows, stale_checks, hook_calls,
# warm_errors, launch_errors, warm_attempts, warm_dropped.
_STATS_PREV_CUM = [0, 0, 0, 0, 0, 0, 0]
_STATS_SEC = _env_int("VLLM_PLE_MMAP_STATS_SEC", 30)


def _cum_of(d: dict) -> tuple:
    """The cumulative-counter tuple, in _STATS_PREV_CUM order."""
    return (d["warmed_rows"], d["stale_checks"], d["hook_calls"],
            d["warm_errors"], d["launch_errors"], d["warm_attempts"],
            d["warm_dropped"])


def _stats_log() -> None:
    import time as _time

    now = _time.monotonic()
    if _STATS_SEC <= 0:
        return
    # Window bookkeeping (boundary test, clock advance, snapshot+reset,
    # delta-base update) all under one lock: two racing callers must not
    # split one window's deltas across two lines (a split would zero
    # d_warmed on the second line and spuriously trip the alarms). Both
    # call sites run on the runner thread today; the lock makes that a
    # non-load-bearing fact.
    with _STATS_LOCK:
        if _STATS_LAST[0] == 0.0:
            # First call defines the window origin; nothing to report yet (a
            # fabricated 30 s elapsed would make every first-line rate
            # wrong). Seed the delta base too, so the first emitted window's
            # deltas measure the window, not everything since process start.
            _STATS_LAST[0] = now
            _STATS_PREV_CUM[:] = _cum_of(_STATS)
            return
        if now - _STATS_LAST[0] < _STATS_SEC:
            return
        s = dict(_STATS)
        cum = _cum_of(s)
        # No empty-window skip: both call sites bump a counter (calls or
        # hook_calls) before invoking, so every evaluated window has
        # activity by construction; an idle process never reaches here.
        elapsed = now - _STATS_LAST[0]
        _STATS_LAST[0] = now
        _STATS.update(calls=0, op_ms=0.0, gather_ms=0.0, rows=0, bytes=0)
        prev = tuple(_STATS_PREV_CUM)
        _STATS_PREV_CUM[:] = cum
    # Per-window deltas of the cumulative counters (the alarms below must
    # reason about THIS window: absolutes go stale after the first success).
    d_warmed = cum[0] - prev[0]
    d_hook = cum[2] - prev[2]
    d_errors = cum[3] - prev[3]
    d_launch = cum[4] - prev[4]
    d_attempts = cum[5] - prev[5]
    d_dropped = cum[6] - prev[6]
    calls = max(1, s["calls"])
    logger.info(
        "PLE mmap stats (last %.0fs): %d eager ops, op %.0f ms total "
        "(%.2f ms/op), gather %.0f ms total (%.2f ms/op), %d rows, "
        "%.1f MiB read | cumulative: hook-warmed rows %d, warm attempts %d, "
        "warms dropped %d, warm errors %d, launch errors %d, stale reads "
        "%d of %d checks (visibility-only: probe position excludes draft "
        "rollback)",
        elapsed, s["calls"], s["op_ms"], s["op_ms"] / calls,
        s["gather_ms"], s["gather_ms"] / calls,
        s["rows"], s["bytes"] / 2**20, s.get("warmed_rows", 0),
        s.get("warm_attempts", 0),
        s.get("warm_dropped", 0), s.get("warm_errors", 0),
        s.get("launch_errors", 0),
        s.get("stale_hits", 0), s.get("stale_checks", 0),
    )
    # --- Health warning: one dumb, honest per-window check. Anything
    # cleverer (severity taxonomies, consecutive-window streaks, decaying
    # counters) bred new edge cases in four consecutive review rounds — the
    # operator gets the raw deltas and the diagnosis stays theirs. An
    # unhealthy window = errors, failed launches, or warm attempts with
    # zero completions. Wall-clock rate limit (>= 60 s between warnings) so
    # no VLLM_PLE_MMAP_STATS_SEC setting can turn a persistent fault into a
    # flood — and a persistent fault keeps re-warning every minute, never
    # latching off.
    # Two DELIBERATE properties — do not "fix" them back into a state
    # machine:
    #  * A pure-backpressure window (all attempts dropped behind one
    #    in-flight warm, zero errors) IS flagged. That shape is also
    #    healthy-under-load; the breakdown line disambiguates it (0 errors,
    #    0 launches, N dropped) and the 60 s cap bounds the cost, while the
    #    same predicate catches a genuinely wedged executor.
    #  * The limiter is per-emitter, not per-fault-type: a second fault
    #    class arriving inside the 60 s shadow is not separately warned.
    #    The every-window INFO line (cumulative counters) and gather_gpu's
    #    own launch-failure log are the backstops.
    #  * A window ending between a warm's submit (attempts, runner thread)
    #    and its completion (warmed, worker thread) shows attempts > 0 with
    #    0 warmed, 0 dropped, 0 errors — also benign; the next window's
    #    warmed count carries the completion.
    # The limiter state below is read-modify-written outside _STATS_LOCK on
    # purpose: worst case under (currently nonexistent) concurrent callers
    # is one duplicate warning line.
    unhealthy = (d_errors + d_launch) > 0 or (d_attempts > 0 and d_warmed == 0)
    if unhealthy and now - _ALARM_LAST[0] >= 60.0:
        _ALARM_LAST[0] = now
        logger.warning(
            "PLE mmap health (this window): %d hook/warm errors, %d failed "
            "kernel launches (PLE embeddings ZEROED for those tokens), "
            "%d warm attempts, %d dropped, %d rows warmed — investigate if "
            "this repeats", d_errors, d_launch, d_attempts, d_dropped,
            d_warmed,
        )
    if _DECODE_WARM.get("installed"):
        if d_hook > 0:
            # execute_model demonstrably reaches the wrapper: the
            # wrong-runner failure mode is impossible from here on.
            _DECODE_WARM["proven_reachable"] = True
        elif s["calls"]:
            # Eager traffic but no hook invocations this window — the grace
            # clock for the wrong-runner warning. Accumulated in SECONDS
            # (not windows) so the grace does not scale with the tunable
            # window length, and clamped so idle gaps between invocations
            # do not masquerade as traffic time.
            _DECODE_WARM["busy_seconds"] = (
                _DECODE_WARM.get("busy_seconds", 0.0)
                + min(elapsed, float(_STATS_SEC) * 2)
            )
        if (
            not _DECODE_WARM.get("proven_reachable")
            and _DECODE_WARM.get("busy_seconds", 0.0) >= 300.0
            and not _DECODE_WARM.get("never_fired_logged")
        ):
            # >= 5 minutes of eager traffic and the wrapper never ran once.
            # (Residual known limit: an extraordinarily long compile/capture
            # phase can still trip this before first real traffic — hence
            # the hedged wording; the module has no independent signal for
            # "a real request was served".)
            _DECODE_WARM["never_fired_logged"] = True
            logger.warning(
                "PLE mmap: decode warm hook installed but never invoked "
                "after %.0f s of eager traffic — wrapped runner class not "
                "in use? (If this appears during an unusually long startup "
                "compile it may be premature.) Cold decode rows will "
                "demand-fault.",
                _DECODE_WARM.get("busy_seconds", 0.0),
            )
    # Pinned counter written by the kernel with system-scope atomics; host
    # read here is racy by at most one window, which is fine for a monotonic
    # cumulative alarm.
    if _OOB_COUNT[0] is not None and int(_OOB_COUNT[0][0]):
        logger.warning(
            "PLE mmap: %d out-of-range row ids zero-filled by the GPU "
            "gather (cumulative) — id computation and table shape disagree "
            "(checkpoint/VLLM_PLE_MMAP_DIR mismatch?)", int(_OOB_COUNT[0][0]),
        )


def _lookup_impl(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    import time as _time

    t0 = _time.perf_counter()
    layer = _REGISTRY[layer_name]
    result = layer._ple_mmap_orig_forward_impl(
        None, input_ids, query_start_loc, ngram_context
    )
    output[: result.shape[0]].copy_(result.to(output.dtype))
    _stats_add("calls", 1)
    _stats_add("op_ms", (_time.perf_counter() - t0) * 1e3)
    _stats_log()


def _lookup_fake(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


def _register_op() -> None:
    if hasattr(torch.ops.vllm, _OP_NAME):
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name=_OP_NAME,
        op_func=_lookup_impl,
        mutates_args=["output"],
        fake_impl=_lookup_fake,
    )


def apply(cls: type) -> None:
    """Patch ``Qwen3_8FlashNextNGramEmbedding`` (pass the class) when enabled."""
    if not enabled():
        return
    if getattr(cls, "_ple_mmap_patched", False):
        return
    mod = sys.modules[cls.__module__]
    orig_init = cls.__init__
    orig_load_weights = cls.load_weights

    def __init__(self, config, embedding_dim, ple_dense_layer_id, max_total_tokens,
                 max_num_reqs, prefix, quant_config=None, params_dtype=None):
        # Run the stock constructor (hash buffers, workspaces, ...) with the
        # embedding class swapped for our placeholder so nothing large is
        # allocated. quant_config=None keeps the stock code from selecting an
        # FP8 quant method that would create an FP8 weight parameter.
        real_embedding_cls = mod.VocabParallelEmbedding
        mod.VocabParallelEmbedding = lambda n, d, **_kw: _MmapNgramEmbedding(n, d)
        try:
            orig_init(self, config, embedding_dim, ple_dense_layer_id,
                      max_total_tokens, max_num_reqs, prefix,
                      quant_config=None, params_dtype=params_dtype)
        finally:
            mod.VocabParallelEmbedding = real_embedding_cls
        self._ple_mmap_prefix = prefix
        _REGISTRY[prefix] = self
        self._ple_mmap_model_path = None
        try:
            from vllm.config import get_current_vllm_config
            self._ple_mmap_model_path = get_current_vllm_config().model_config.model
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("PLE mmap: cannot read model path from vllm config: %s", exc)
        if params_dtype is not None:
            self.ngram_embedding._zeros_dtype = params_dtype
        logger.info(
            "PLE mmap: %s -> placeholder embedding (%d rows x %d), table will be mmapped",
            prefix, self.ngram_embedding.org_vocab_size, self.head_dim,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        rest: list[tuple[str, torch.Tensor]] = []
        for name, w in weights:
            if name.startswith("ngram_embedding.shard_") and name.endswith(".weight"):
                loaded.add(name)  # served from disk, never materialised
                continue
            if name == "ngram_embedding.weight_scale":
                self.register_buffer(
                    "_offload_weight_scale",
                    w.detach().to(device=torch.accelerator.current_accelerator()),
                    persistent=False,
                )
                loaded.add(name)
                continue
            rest.append((name, w))
        loaded.update(orig_load_weights(self, rest))
        _setup_table(self)
        return loaded

    def _setup_table(self) -> None:
        if self.ngram_embedding.table is not None:
            return
        # VLLM_PLE_MMAP_DIR: serve the table from a different directory than the
        # checkpoint (e.g. an FP8 copy of the table on local NVMe).
        model_path = os.environ.get("VLLM_PLE_MMAP_DIR") or self._ple_mmap_model_path
        if not model_path or not os.path.isdir(model_path):
            raise RuntimeError(
                f"PLE mmap: table path {model_path!r} is not a local directory; "
                "point --model at the downloaded snapshot or set VLLM_PLE_MMAP_DIR"
            )
        m = re.search(r"layers\.(\d+)\.", self._ple_mmap_prefix)
        if not m:
            raise RuntimeError(f"PLE mmap: cannot find layer index in {self._ple_mmap_prefix!r}")
        layer_idx = int(m.group(1))
        shards, cols, dtype_str, scale_entry = _find_shards(model_path, layer_idx)
        if not shards:
            raise RuntimeError(f"PLE mmap: no shard tensors for layer {layer_idx} under {model_path}")
        if cols != self.head_dim:
            raise RuntimeError(f"PLE mmap: shard width {cols} != head_dim {self.head_dim}")
        if dtype_str not in _TABLE_DTYPES:
            raise RuntimeError(f"PLE mmap: unsupported shard dtype {dtype_str}")
        if dtype_str in _FP8_DTYPES and not hasattr(self, "_offload_weight_scale"):
            if scale_entry is None:
                raise RuntimeError("PLE mmap: FP8 shards without ngram_embedding.weight_scale")
            self.register_buffer(
                "_offload_weight_scale",
                _read_scale(scale_entry).to(torch.accelerator.current_accelerator()),
                persistent=False,
            )
        parts = int(self.split_ngram_parts)
        vocab = int(self.ngram_embedding.org_vocab_size)
        shard_size = math.ceil(vocab / parts)
        # Completeness, not just per-shard shape: a truncated index or partial
        # VLLM_PLE_MMAP_DIR copy must fail the boot here — the alternative is
        # the kernel silently zero-filling every id above the highest found
        # shard and the model serving fluent garbage.
        missing = [i for i in range(parts) if i not in shards
                   and max(0, min(shard_size, vocab - i * shard_size)) > 0]
        if missing:
            raise RuntimeError(
                f"PLE mmap: {len(missing)} of {parts} shards missing under "
                f"{model_path} (first missing: {missing[:5]}) — incomplete "
                "checkpoint copy?"
            )
        for idx, (_p, _o, rows) in shards.items():
            expected = max(0, min(shard_size, vocab - idx * shard_size))
            if rows != expected:
                raise RuntimeError(
                    f"PLE mmap: shard {idx} has {rows} rows, expected {expected}"
                )
        table = MmapPleTable(
            shards, shard_size, cols * _itemsize(dtype_str), _TABLE_DTYPES[dtype_str],
            workers=_env_int("VLLM_PLE_MMAP_WORKERS", 32),
            chunk=_env_int("VLLM_PLE_MMAP_CHUNK", 2048),
        )
        if os.environ.get("VLLM_PLE_MMAP_PREWARM", "0").lower() in ("1", "true", "yes"):
            logger.info("PLE mmap: prewarming page cache (%.1f GiB)...", table.rows_total * table.row_bytes / 2**30)
            table.prewarm()
        # Host-side copies of the hash buffers for the decode-warm thread,
        # made once here on the load path — the warm thread itself must never
        # issue CUDA calls (D2H copies from a side thread serialize against
        # the compute stream and can corrupt graph capture). Assigned BEFORE
        # the table is published: the hook selects owners by `table is not
        # None` and then reads _ple_host_hash.
        self._ple_host_hash = (
            int(self.eos_token_id),
            self.layer_multipliers.detach().cpu().numpy().astype(np.int64),
            self.ngram_heads_vocab_sizes.detach().cpu().numpy().astype(np.int64),
            self.ngram_heads_offsets.detach().cpu().numpy().astype(np.int64),
            int(self.ngram_size),
            int(self.heads_per_ngram),
        )
        self.ngram_embedding.table = table
        if gpu_gather_enabled():
            # Eager GPU setup + one verified test gather, outside any capture.
            table.init_gpu(torch.accelerator.current_accelerator())
        else:
            logger.info(
                "PLE mmap: CPU threadpool gather mode (VLLM_PLE_GPU_GATHER "
                "unset) — vllm::ple_mmap_lookup must be in splitting_ops"
            )
        _install_decode_warm()
        logger.info(
            "PLE mmap: layer %d, %d shards, %d rows x %d B (%.1f GiB on disk), dtype %s, %d workers",
            layer_idx, len(shards), table.rows_total, table.row_bytes,
            table.rows_total * table.row_bytes / 2**30, dtype_str, table.workers,
        )

    def forward_impl(self, hidden_states, input_ids, query_start_loc, ngram_context,
                     output_buffer=None):
        del hidden_states, output_buffer
        num_tokens = input_ids.reshape(-1).shape[0]
        table = self.ngram_embedding.table
        dtype = table.torch_dtype if table is not None else self.ngram_embedding._zeros_dtype
        output = torch.empty(
            (num_tokens, self.embedding_dim), dtype=dtype, device=input_ids.device
        )
        getattr(torch.ops.vllm, _OP_NAME)(
            input_ids, query_start_loc, ngram_context, output, self._ple_mmap_prefix
        )
        return output

    _register_op()
    cls._ple_mmap_orig_forward_impl = cls.forward_impl
    cls.forward_impl = forward_impl
    cls.__init__ = __init__
    cls.load_weights = load_weights
    cls._setup_table = _setup_table
    cls._ple_mmap_patched = True
    logger.info("PLE mmap patch applied to %s.%s", cls.__module__, cls.__name__)
