# Qwen3.8-Flash-Next on a single DGX Spark (GB10)

Run **Qwen3.8-Flash-Next** — a ~176B-parameter model (125B main + 51B n-gram, 6B
active) — on **one NVIDIA DGX Spark / ASUS GX10** with **vLLM**, at full prefill
speed, with MTP speculative decoding, **working prefix caching**, **deterministic
greedy decoding**, and up to **500k tokens of context**.

The catch this repo solves: the NVFP4 checkpoint is **122 GiB**, which does not fit
next to a usable KV cache in the Spark's **128 GB unified pool**. 44 GiB of that is
the n-gram embedding ("PLE") table — a pure lookup that a token only touches 16 rows
of. This repo patches the official vLLM image to **serve that table from NVMe via
`mmap`** instead of keeping it resident. Weights drop to **~76 GiB**, the rest of the
pool goes to KV, and everything runs on stock GB10 kernels.

Along the way it also fixes two things that were broken for this model on GB10 —
**prefix caching** (a vLLM block-size bug silently restored an all-zero Mamba state
on every cache hit) and **non-deterministic top-k in the sparse attention** (a GB10
kernel that drops candidates) — and offers an optional **hybrid** checkpoint layout
(NVFP4 experts + fp8 side layers) that decodes ~20% faster at the same quality.

> **Independently reproduced** on a DGX Spark by
> [@jschmied](https://github.com/jschmied) — see
> [issue #1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1) and their
> [write-up](https://github.com/jschmied/qwen38-flash-next-gb10).

## Running with sparkrun

This fork adds [sparkrun](https://sparkrun.dev/) recipes so the whole thing is
one command on a fresh Spark — no manual `docker build`, weight download, or
`serve.sh`:

```bash
sparkrun registry add https://github.com/lancelind/qwen3.8-Flash-DGX
sparkrun run qwen38-flash-nvfp4-vllm --solo
```

sparkrun pulls the patched image from
`ghcr.io/lancelind/qwen38-flash-dgx` (built by CI from this repo's Dockerfile),
downloads `RadixArk/Qwen3.8-Flash-Next-NVFP4` into its model cache, and serves
the OpenAI-compatible API on port 8000 (override with `-o port=18300`).

Recipes:

- [`recipes/qwen38-flash-nvfp4-vllm.yaml`](recipes/qwen38-flash-nvfp4-vllm.yaml) —
  the default: 262k context, prefix caching, exact top-k, MTP=2.
- [`recipes/qwen38-flash-nvfp4-yarn500k-vllm.yaml`](recipes/qwen38-flash-nvfp4-yarn500k-vllm.yaml) —
  YaRN rope scaling for ~500k context.

Notes:

- **Single node only** (`max_nodes: 1`): the PLE table is mmapped from local
  NVMe; there is no multi-Spark tensor parallelism here.
- Any `defaults:` key can be overridden per-run, e.g.
  `-o gpu_memory_utilization=0.80 -o max_num_seqs=16`.
- **Hybrid mode** (`MODE=hybrid`) is not covered by a recipe yet: its fp8
  side-layer checkpoint is generated locally by `scripts/prepare-hybrid.sh`.
  Either keep using `serve.sh` for it, or upload the prepared checkpoint to
  Hugging Face and point a copy of the recipe at it with
  `env: {VLLM_FP8_HYBRID: "1", VLLM_USE_DEEP_GEMM: "0"}`.

## Update 2026-08-29 — what changed

If you cloned this before, here is the short version (details in the linked sections):

- **Prefix caching works now** — `--enable-prefix-caching` was crashing, then silently
  returning wrong answers on cache hits. Root cause was a vLLM block-size bug that made
  every prefix hit restore an *all-zero* Mamba state; two-line fix in the image. Getting
  there took [@Saren-Arterius](https://github.com/Saren-Arterius)'s pointer to
  vllm#50729 and their state-copy guard, and [@0xBakeer](https://github.com/0xBakeer)'s
  attempt to reproduce it, which sharpened the write-up.
  `PREFIX_CACHE=1` is the new default. Repeated prefixes (system prompts, multi-turn,
  tool loops) skip the prefill: ~14 s → ~1.4 s TTFT on a 20k-token prefix.
  → [Prefix caching now works](#prefix-caching-now-works-and-why-it-didnt)
- **Greedy decoding is deterministic now** — the GB10 sparse-attention top-k kernel was
  non-deterministic and dropped candidates, diagnosed and reported upstream by
  [@k3dani](https://github.com/k3dani) (issue #3, vllm#51782). Exact top-k
  is the new default (`EXACT_TOPK=1`); identical outputs at temperature 0, same
  tournament score, costs some long-prefill speed. → [Deterministic top-k](#deterministic-top-k-exact_topk1-default)
- **Two checkpoint modes** — `MODE=nvfp4` (as published) or `MODE=hybrid` (NVFP4 experts
  + fp8 side layers, one-time `scripts/prepare-hybrid.sh`): **+20% decode, +8% KV,
  same quality**. Our box runs the hybrid. The fp8 side-layer conversion and the
  original int4+fp8 dispatch it is ported from are
  [@Saren-Arterius](https://github.com/Saren-Arterius)'s. → [Two checkpoint modes](#two-checkpoint-modes-nvfp4-or-hybrid)
- **Also in the image**: vllm#50729 (Mamba state-copy race, by
  [@AndreasKaratzas](https://github.com/AndreasKaratzas)) + a bounds guard, the GB10 FLA
  fixes and the faster PLE gather from [@Saren-Arterius](https://github.com/Saren-Arterius)'s fork.
- **We benchmarked the int4 (Intel AutoRound) variant too** with the same patches:
  fastest raw decode, but not deterministic and slowest cached-TTFT, so we did not adopt
  it. Numbers in [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md#hybrid-mode-nvfp4-experts--blockwise-fp8-side-layers).
- **fp8 KV cache is available** (`KV_DTYPE=fp8_e4m3`), contributed by
  [@Nanetnounou](https://github.com/Nanetnounou): ×1.9 KV, 1M context on one box — at a
  speed and quality cost, so it is opt-in. → [fp8 KV cache](#optional-fp8-kv-cache-1m-context)
- `scripts/smoke-test.sh` now also checks the prefix-cache hit and determinism, and
  measures decode on a real answer instead of `ignore_eos` (which produces meaningless
  numbers with this model). `scripts/download-weights.sh` now forwards `HF_TOKEN`
  ([@wawimundo](https://github.com/wawimundo), PR #4).

Everything was measured on one ASUS GX10 with a 17-scenario agentic tournament (3 repeats
each), single-request speed benches on real prompts, and state checksums for the
prefix-caching work; nothing here is extrapolated.

| | llama.cpp IQ4_XS | **NVFP4 (this repo)** | **hybrid (this repo)** |
|---|---|---|---|
| Prefill | ~540 tok/s | **~1,500–2,000 tok/s** (exact top-k; ~2,400 with the stock kernel) | same |
| Decode, single stream | ~22 tok/s (no MTP) | **~26 tok/s** with MTP=2 | **~31 tok/s** |
| Prefix-cache hit, TTFT on a 20k-token prefix | n/a | **~1.4 s** (vs ~14 s cold) | same |
| Context | 262k | **262k native, 500k with YaRN** | same |
| KV cache @0.80 (500k YaRN, MTP) | — | ~580k tokens | ~630k tokens |
| Deterministic at temperature 0 | yes | **yes** (`EXACT_TOPK=1`) | **yes** |

*Measured on an ASUS GX10 (GB10, 128 GB), single request, real prompts, greedy. Quality
(a 17-scenario agentic tournament, 3 repeats) is identical across NVFP4 and hybrid:
45/51 both, same two scenarios failed by every quantization we tried. Details and the
full comparison tables are in [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md).*

---

## Requirements

- An **NVIDIA DGX Spark or compatible GB10 (sm_121)** box, 128 GB unified memory,
  aarch64, recent NVIDIA driver, Docker with the NVIDIA container runtime.
- **~130 GB free disk** for the checkpoint (+13 GB for the hybrid variant), on
  reasonably fast storage (the table is read from it at runtime — NVMe strongly
  recommended; the Spark's onboard NVMe is ideal).
- The base image is multi-arch, so `docker build` also works on x86 Blackwell
  (sm_120, e.g. RTX PRO 6000) for testing, though this is tuned for the Spark.

## Quickstart

```bash
git clone https://github.com/blazux/qwen3.8-Flash-DGX.git
cd qwen3.8-Flash-DGX

docker build -t qwen38-flash-dgx .        # ~1 min: official image + the patches
scripts/download-weights.sh               # ~122 GiB, resumable (one-time)
scripts/serve.sh                          # NVFP4 mode, boots on :18300 (~8-13 min to load)
docker logs -f qwen38-flash               # wait for "Application startup complete"
scripts/smoke-test.sh                     # health, coherence, prefix-cache hit, determinism, tok/s
```

Then hit the OpenAI-compatible API:

```bash
curl http://localhost:18300/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next",
  "messages": [{"role":"user","content":"Write a haiku about a desktop supercomputer."}],
  "max_tokens": 512
}'
```

500k context (YaRN, validated with a needle-in-a-haystack at 414k tokens):

```bash
YARN=1 CTX=500000 GPU_MEM=0.80 scripts/serve.sh
```

## Two checkpoint modes: NVFP4 or hybrid

`scripts/serve.sh` serves one of two layouts of the same RadixArk NVFP4 checkpoint;
pick with `MODE=`.

| | `MODE=nvfp4` (default) | `MODE=hybrid` |
|---|---|---|
| Routed experts (the bulk, ~63 GiB) | NVFP4 | NVFP4 (unchanged) |
| GDN in/out projections, QSA q/k/v/o, shared experts (~15 GiB) | bf16, as published | **blockwise fp8-e4m3** (128×128 blocks, DeepSeek layout) |
| Extra preparation | none | `scripts/prepare-hybrid.sh` once (~10 min, +13 GB disk) |
| Decode (MTP=2, greedy, real answers) | ~26 tok/s | **~31 tok/s (+20%)** |
| Prefill | same | same (±5%) |
| KV cache | ~580k tokens | **~630k tokens (+8%)**, weights ~7 GiB smaller |
| Tournament quality (17 agentic scenarios × 3) | 45/51 | 45/51 |
| Deterministic at T=0 | yes | yes |
| Behavioural difference we noticed | — | slightly "more careful" in tool loops: it sometimes checks state first (one extra tool call), which is the only place it scored differently before we raised the turn budget |

Why it works: those side layers are dense and read in full on every decoded token, so
they dominate decode bandwidth; the experts are sparse (10 of 512 active) and already
4-bit. Halving the dense part is where the tokens/s come from. The MoE path — where
the quality lives — is untouched. Conversion uses
[@Saren-Arterius](https://github.com/Saren-Arterius)'s `fp8_convert.py` (worst
per-tensor max relative error 3.5%), and a small dispatch shim
(`src/vllm_fp8_hybrid_modelopt.py`) that routes those layers to vLLM's blockwise-fp8
GEMM while the ModelOpt NVFP4 config keeps handling the experts.

```bash
scripts/prepare-hybrid.sh                 # builds <snapshot>-fp8hybrid/ next to the HF snapshot
MODE=hybrid YARN=1 CTX=500000 GPU_MEM=0.80 scripts/serve.sh
```

Our own box runs the hybrid. If you want the checkpoint exactly as published, stay on
`MODE=nvfp4` — you lose ~5 tok/s and nothing else.

## Prefix caching now works (and why it didn't)

`--enable-prefix-caching` used to crash this model on GB10 (`CUDA illegal memory
access`) and, with a bounds guard in place, to **silently return different answers on
cache hits**. We traced it (details in [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md)):
vLLM's engine core overwrites `cache_config.block_size` with the *smallest* KV-group
block size — 8 tokens here with MTP=2 (4 without), the QSA raw-key ring — while the Mamba
state block is 1600 tokens. Two places used the former as the latter, so on a prefix hit
the worker computed the state slot as `(3200-1)//8 = 399` instead of `1`, read past the
block table row, and restored an **all-zero Mamba state**. The image carries a two-line fix;
with it, cold and cache-hit outputs are bit-identical (state checksums and first-token
logprobs match exactly) and the tournament score is unchanged.

What you get: multi-turn chats, agent/tool loops and shared system prompts skip the
prefill of everything already seen. On a 20k-token prefix, TTFT goes from ~14 s to
~1.4 s; with 8 concurrent conversations, from ~80 s to ~4–6 s. `PREFIX_CACHE=1` is the
default. Mamba states are cached at 1600-token boundaries, so the tail of a prefix is
recomputed — expect the benefit to start around a couple of thousand tokens.

## Deterministic top-k (`EXACT_TOPK=1`, default)

The sparse attention (QSA) picks the top-k key blocks per query with a `persistent_topk`
kernel. On GB10 that kernel is **non-deterministic** — identical greedy requests produce
different outputs 2 times out of 4 — and can drop legitimate candidates
([vllm#51782](https://github.com/vllm-project/vllm/issues/51782)); reported against
this repo by [@k3dani](https://github.com/k3dani) in
[issue #3](https://github.com/blazux/qwen3.8-Flash-DGX/issues/3). The GB10 is more
exposed than other GPUs: the cooperative kernel used elsewhere for decode is disabled on
sm_12x, so this kernel runs for both prefill and decode.

`EXACT_TOPK=1` replaces it with an exact `torch.topk` over the visible columns:
**4/4 prompts stable, first-token logprobs identical to the 4th decimal**, tournament
score unchanged. Cost: nothing on decode, ~10% on 8k prefill, ~20–40% on 32k+ prefill.
`EXACT_TOPK=0` gives the stock kernel back if you prefer speed over reproducibility.
(We also tried just masking the never-written columns before the stock kernel: still
unstable, so it is the kernel itself.)

## Optional: fp8 KV cache (1M context)

[@Nanetnounou](https://github.com/Nanetnounou) contributed a patch
([issue #6](https://github.com/blazux/qwen3.8-Flash-DGX/issues/6),
[vllm#54426](https://github.com/vllm-project/vllm/issues/54426)) that lets the QSA
layers read an `fp8_e4m3` KV cache — the model used to refuse anything but bf16. It is
in the image, inert unless you pass `KV_DTYPE=fp8_e4m3`. Measured on the GX10 (hybrid,
MTP=2, exact top-k, prefix caching, `GPU_MEM=0.80`):

| | bf16 KV (default) | `KV_DTYPE=fp8_e4m3` |
|---|---|---|
| KV pool | 634k tokens | **1,219k tokens (×1.9)** |
| Max single request | 500k (YaRN) | **1,000k (YaRN)** — boots and serves |
| Decode | 30.8 tok/s | 27.9 tok/s (−9%) |
| Prefill 32k | 1,794 tok/s | 1,254 tok/s (−30%) |
| Needle 92k | 69 s | 89 s |
| Tournament | 45/51 | 45/51, but `b3_itinerary` (long multi-constraint reasoning) drops from 6/6 to 2/6 passes and its successful run took 4× longer |

The slowdown comes from the in-kernel dequantization and the halved `block_n` needed to
fit the GB10's shared memory; the quality dip is the fp8 rounding of K/V in the sparse
attention. Our recommendation: keep bf16 unless you actually need > 500k tokens in one
request or 2× the concurrency at 500k, and re-check your own workload's quality if you
switch. Attention blocks become 3,184 tokens in this mode (page alignment), so prefix
caching captures Mamba states less often.

```bash
KV_DTYPE=fp8_e4m3 YARN=1 CTX=1000000 GPU_MEM=0.80 MODE=hybrid scripts/serve.sh
```

## Tuning (env vars for `scripts/serve.sh`)

| Var | Default | Notes |
|---|---|---|
| `MODE` | `nvfp4` | `hybrid` = fp8 side layers (see above; needs `scripts/prepare-hybrid.sh`). |
| `PREFIX_CACHE` | `1` | `--enable-prefix-caching`. Correct with this image (block_size fix). |
| `EXACT_TOPK` | `1` | Exact, deterministic QSA top-k. **Costs prefill on long prompts: −8% at 8k, −20–40% at 32k+** (decode unchanged; with prefix caching only the new tokens pay it). `0` = stock kernel: full prefill speed, but non-deterministic and may drop attention candidates (issue #3). |
| `PORT` | `18300` | API port |
| `CTX` | `262144` | Max context. Native is 262144; with `YARN=1` up to `500000` is validated. |
| `YARN` | `0` | `1` = YaRN rope scaling (factor 4, Qwen's recipe) for `CTX` > 262144. |
| `SEQS` | `8` | Max concurrent sequences. **Do not benchmark with 1–2**: excess requests queue silently and aggregate tok/s flatlines (see below). |
| `GPU_MEM` | `0.85` | Fraction of the 128 GB pool for weights+KV. `0.875` got OOM-killed on a 300k-token prefill with MTP; for long-running service we run `0.80`, because after a day at `0.85` the box drifted into swap. Right after stopping another big container the first boot can fail with "13.5 GiB KV cache is needed, larger than available" — memory not yet released; the `unless-stopped` retry succeeds. |
| `MTP` | `2` | Speculative tokens from the model's MTP head (`0` = off). |
| `KV_DTYPE` | `auto` | `auto` = bf16 (recommended). `fp8_e4m3` = ~1.9× KV pool, 1M context on one box, at −10% decode / −30% prefill and a measurable quality cost — see [fp8 KV cache](#optional-fp8-kv-cache-1m-context) before using it. |
| `PREWARM` | `0` | `1` streams the 48 GiB table once at boot to warm the page cache — steadier first-request latency, ~10 s extra startup. |
| `WORKERS` | `32` | Threads used for the mmap gather (only used above `VLLM_PLE_MMAP_FAST_ROWS`=512 unique rows; decode-sized gathers run inline). |
| `EXTRA` | | Extra vLLM flags, passed verbatim. |

## Throughput and concurrency

Single-stream numbers understate this model on a GB10. @jschmied traced one box
under load (RadixArk NVFP4, 8k ctx, **no** speculative decoding, using vLLM's native
PLE CPU offload rather than this repo's mmap — the table-serving cost behaves the
same way) and found aggregate throughput scales far past single-stream:

| concurrent streams | aggregate tok/s | per stream | major faults / token | TTFT |
|---:|---:|---:|---:|---:|
| 1 | 17.1 | 17.1 | 16.0 | 0.22 s |
| 8 | 87.5 | 10.9 | 7.0 | 0.53 s |
| 16 | 131.6 | 8.2 | 9.6 | 0.83 s |
| 32 | 212.0 | 6.6 | 4.3 | 1.19 s |
| 48 | **266.8** | 5.6 | 3.6 | 1.60 s |

Two things worth knowing (their words, lightly condensed):

- **The paged table is an argument *for* concurrency, not against it.** Page-fault cost
  per token *falls* 4.4× from c=1 to c=48: batched tokens share n-gram rows and the
  page cache keeps the hot set, so the marginal token is far cheaper than the first.
  The table gather itself never exceeded ~25% of one CPU core.
- **A low `--max-num-seqs` is indistinguishable from saturation if you only look at
  tok/s.** With `--max-num-seqs 2` their sweep flatlined at ~33 tok/s while
  `vllm:request_queue_time_seconds_sum` climbed to 142 s. Check `max-num-seqs` before
  quoting an aggregate number — this repo's default is now `8` for that reason.

Method and harness: [load-and-waits.md](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/load-and-waits.md).

## How it fits — the one idea

A token's n-gram lookup reads **16 rows × 160 bytes ≈ 2.5 KB**. Over a 20k-token
prefill that's ~1.3 GB of small reads — under a second on NVMe, and the hot n-grams
stay in the page cache. So the 44 GiB table never needs to be in the unified pool:
we `mmap` the checkpoint's `model-plefp8-*.safetensors` shards and gather rows on
demand. Nothing else about the model changes — the hashing, dequant, and the sparse
attention all run stock.

Full details, including the GB10-specific bugs this works around and the long-context
findings, are in [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md).

### GB10 kernel fixes and the faster gather (contributed)

From [@Saren-Arterius](https://github.com/Saren-Arterius)'s fork, merged here with thanks:

- **FLA shared-memory gate** — sm_121 reports 99 KiB of shared memory per block but the
  flash-linear-attention gate asked for 100 KiB, so all 36 GDN layers silently ran on
  small tiles. One `sed` in the Dockerfile lowers the gate to 99 KiB.
- **`chunk_delta_h` `num_warps` pin** — works around a `tl.dot` race on Blackwell
  ([fla#953](https://github.com/fla-org/flash-linear-attention/issues/953)). Correctness, not speed.
- **PLE gather hot path** — CPU dedup of row ids, a persistent pinned staging buffer with an
  async H2D copy, GPU-side expansion through the inverse index, and an inline fast path
  for decode-sized batches (`VLLM_PLE_MMAP_FAST_ROWS`, default 512). Also: bf16/f16 tables,
  `VLLM_PLE_MMAP_DIR` to serve the table from another directory, and a periodic
  `PLE mmap stats` log line (`VLLM_PLE_MMAP_STATS_SEC`, default 30).
- **Mamba state-copy guard** — with [vllm#50729](https://github.com/vllm-project/vllm/pull/50729)
  (the overlapping-copy race fix by @AndreasKaratzas), a bounds check that turns an
  out-of-range block id into a skipped copy plus a log counter instead of a dead CUDA
  context. With the block_size fix above the counter stays at 0; if you ever see
  `mamba state-copy guard: N out-of-range`, something upstream regressed — please report it.
- **`fp8_convert.py`** — the side-layer conversion behind `MODE=hybrid`.

Their fork goes further with an **int4 (Intel AutoRound) + fp8 hybrid** checkpoint:
[qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound).
We benchmarked it with the same patches: ~34 tok/s decode, 44/51 on the tournament,
but it is not deterministic even with the exact top-k and Marlin's atomic adds off, and
it has the slowest prefill of the three — we kept the NVFP4-based layouts.

### Alternative: vLLM's native PLE CPU offload

vLLM ships its own path (`VLLM_PLE_CPU_OFFLOAD=1`) that keeps the table in pinned host
RAM in a separate worker process. On a Spark that RAM is the same pool as the GPU, so
it saves less than the mmap — but @jschmied got it running and documented two things
you will need if you go that way (neither applies to the mmap patch, which is a single
process):

1. `_get_ple_embedding_quant_method()` in `ple_layer.py` only accepts `Fp8Config`;
   with the NVFP4 checkpoint the quant config is `modelopt_fp4`, so the FP8 PLE shards
   are rejected and loading dies on `ngram_embedding.weight_scale`. Accepting
   `modelopt`/`modelopt_fp4` there fixes it.
2. The worker hands CUDA tensors to the GPU process over IPC via `pidfd_getfd`, which
   `kernel.yama.ptrace_scope=1` (the Ubuntu/DGX OS default) forbids between sibling
   processes. In Docker: `--cap-add=SYS_PTRACE`. Under systemd:
   `AmbientCapabilities=CAP_SYS_PTRACE`. It fails ~10 minutes in, after all shards
   have loaded, with an unhelpful `Engine core initialization failed`.

Details: [results-radixark-vllm.md](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/results-radixark-vllm.md).

## What's in here

```
Dockerfile                        official vLLM Flash-Next image + the patches below
src/vllm_ple_mmap.py              1. mmap PLE table (opaque splitting op)            VLLM_PLE_MMAP=1
src/mamba_utils_guarded.py        3. vllm#50729 + bounds guard (drop-in mamba_utils.py)
src/patch_mamba_block_size.py     4. prefix-caching block_size fix
src/patch_qsa_exact_topk.py       5. exact, deterministic QSA top-k                  VLLM_QSA_EXACT_TOPK=1
src/vllm_fp8_hybrid_modelopt.py   6. NVFP4 experts + fp8 side layers dispatch        VLLM_FP8_HYBRID=1
src/patch_qsa_fp8_kv.py           7. fp8_e4m3 KV cache on the QSA path (by @Nanetnounou) --kv-cache-dtype fp8_e4m3
src/test_ple_mmap_cpu.py          CPU unit test for the gather (no GPU needed)
src/test_qsa_exact_topk_cpu.py    CPU unit test for the exact top-k (no GPU needed)
tools/fp8_convert.py              side-layer bf16 -> blockwise fp8 (by @Saren-Arterius)
scripts/download-weights.sh
scripts/prepare-hybrid.sh         one-time: build the -fp8hybrid snapshot
scripts/serve.sh                  MODE=nvfp4|hybrid, PREFIX_CACHE, EXACT_TOPK, YARN, ...
scripts/smoke-test.sh             health, coherence, prefix-cache hit, determinism, tok/s
docs/HOW-IT-WORKS.md
```

Run the unit tests (no GPU):

```bash
docker run --rm -v "$PWD/src:/t" -w /t --entrypoint python3 qwen38-flash-dgx test_ple_mmap_cpu.py
docker run --rm -v "$PWD/src:/t" -w /t --entrypoint python3 qwen38-flash-dgx test_qsa_exact_topk_cpu.py
```

## Limitations & notes

- **One big model at a time.** At `GPU_MEM=0.85` this uses most of the 128 GB pool;
  don't co-locate another large model (an 8B embedding model next to it already
  starves the KV cache — we moved ours to another machine).
- **Full `torch.compile` is off** (an Inductor int64-indexing assert on sm_121); the
  serve script uses PIECEWISE CUDA graphs with the PLE lookup as a splitting op.
- **1M context** needs the fp8 KV cache (`KV_DTYPE=fp8_e4m3`, see above), which costs
  speed and some quality; in bf16 a single 1M request needs ~26 GiB of KV and 500k with
  YaRN is the validated ceiling (800k booted but got OOM-killed on a long prefill).
- **Exact top-k costs prefill** on long prompts (see above). The implementation is a
  plain `torch.topk` over the full visible width per chunk; a fused kernel would recover
  most of it — PRs welcome.
- **Weights are not included** and the checkpoint carries Qwen's license (with a
  MAU/revenue clause) — review it before production use.

## Credits

- Model: **Qwen team, Alibaba** — Qwen3.8-Flash-Next.
- NVFP4 checkpoint: **[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)**.
- Serving engine and base image: **vLLM** (`vllm/vllm-openai:qwen38-flash-next`,
  the `release/qwen38next` recipe / PR #53896); the Mamba state-copy race fix is
  [vllm#50729](https://github.com/vllm-project/vllm/pull/50729) by **@AndreasKaratzas**.
- GB10 FLA fixes, the faster PLE gather, the state-copy guard and the fp8 side-layer
  conversion: **[@Saren-Arterius](https://github.com/Saren-Arterius)**
  ([qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)).
- The fp8_e4m3 KV cache patch for the QSA path: **[@Nanetnounou](https://github.com/Nanetnounou)**
  ([issue #6](https://github.com/blazux/qwen3.8-Flash-DGX/issues/6), [vllm#54426](https://github.com/vllm-project/vllm/issues/54426)).
- The non-deterministic `persistent_topk` diagnosis and upstream report:
  **[@k3dani](https://github.com/k3dani)** ([issue #3](https://github.com/blazux/qwen3.8-Flash-DGX/issues/3),
  [vllm#51782](https://github.com/vllm-project/vllm/issues/51782)).
- Independent reproduction on a DGX Spark, the native-offload fixes and the
  concurrency measurements: **[@jschmied](https://github.com/jschmied)**
  ([issue #1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1),
  [qwen38-flash-next-gb10](https://github.com/jschmied/qwen38-flash-next-gb10)).
- The mmap-PLE patch, the hybrid dispatch for ModelOpt-NVFP4, the prefix-caching root
  cause and fix, the exact top-k path and the GB10 serving recipe in this repo: see
  [LICENSE](LICENSE) (Apache-2.0).
