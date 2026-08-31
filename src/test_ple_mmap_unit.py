"""Unit tests for vllm_ple_mmap's table/hash machinery (no GPU needed).

Covers the pieces the aggressive review flagged as untested:
  * vocab computation and CPU gather byte-identity against direct reads.
  * host_ngram_ids vs a torch reimplementation using torch.remainder on
    int64 across random contexts that provably include negative mixed hash
    values (the floored-vs-truncated modulo split).

Run inside the serving container:  python -m pytest src/test_ple_mmap_unit.py -q
(or: python src/test_ple_mmap_unit.py)
"""

import os
import tempfile

import numpy as np

# Config must be set before the module import caches it.
os.environ["VLLM_PLE_MMAP"] = "1"
os.environ["VLLM_PLE_GPU_GATHER"] = "1"
os.environ["VLLM_PLE_DECODE_WARM"] = "1"

import torch  # noqa: E402

import vllm_ple_mmap as M  # noqa: E402

ROW_BYTES = 160
SHARD_SIZE = 3000


def _make_table(tmpdir: str) -> M.MmapPleTable:
    """3 shards with distinct non-page-aligned data offsets; last one short."""
    rng = np.random.default_rng(7)
    shards = {}
    rows_per = [SHARD_SIZE, SHARD_SIZE, 137]
    for i, rows in enumerate(rows_per):
        path = os.path.join(tmpdir, f"shard{i}.bin")
        offset = 8 + 100 * i + 3  # deliberately unaligned data starts
        data = rng.integers(0, 256, size=(rows, ROW_BYTES), dtype=np.uint8)
        with open(path, "wb") as f:
            f.write(b"\x00" * offset)
            f.write(data.tobytes())
        shards[i] = (path, offset, rows)
    return M.MmapPleTable(
        shards, SHARD_SIZE, ROW_BYTES, torch.uint8, workers=4, chunk=64
    )


def test_vocab():
    with tempfile.TemporaryDirectory() as d:
        t = _make_table(d)
        assert t.vocab == 2 * SHARD_SIZE + 137


def test_warm_all_shards_and_pools():
    with tempfile.TemporaryDirectory() as d:
        t = _make_table(d)
        rng = np.random.default_rng(11)
        ids = rng.choice(t.vocab, size=256, replace=False).astype(np.int64)
        t.warm(ids)                          # warm pool
        t.warm(ids, use_gather_pool=True)    # gather pool (prefill path)
        # After warming, the touched bytes must read back identical to the
        # direct file contents (verifies the offset arithmetic touched the
        # right rows, not just that nothing raised).
        got = t.gather(ids)
        for k, gid in enumerate(ids[:16].tolist()):
            sh, loc = gid // SHARD_SIZE, gid % SHARD_SIZE
            assert bytes(got[k]) == bytes(t.mm[sh][loc])


def test_warm_missing_shard_raises():
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(13)
        shards = {}
        for i in (0, 2):  # shard 1 deliberately absent
            path = os.path.join(d, f"s{i}.bin")
            data = rng.integers(0, 256, size=(SHARD_SIZE, ROW_BYTES),
                                dtype=np.uint8)
            with open(path, "wb") as f:
                f.write(b"\x00" * 11)
                f.write(data.tobytes())
            shards[i] = (path, 11, SHARD_SIZE)
        t = M.MmapPleTable(shards, SHARD_SIZE, ROW_BYTES, torch.uint8,
                           workers=2, chunk=64)
        try:
            t.warm(np.arange(SHARD_SIZE, SHARD_SIZE + 8, dtype=np.int64))
        except IndexError:
            return
        raise AssertionError("warm into a missing shard did not raise")


def test_cpu_gather_bounds_match_kernel_vocab():
    # Ids in [vocab, shard_size*len(mm)) must be rejected by the same
    # boundary the kernel enforces, with a message naming the vocab.
    with tempfile.TemporaryDirectory() as d:
        t = _make_table(d)
        assert t.vocab < t.shard_size * len(t.mm)
        for bad in (t.vocab, t.shard_size * len(t.mm) - 1, -1):
            try:
                t.gather(np.array([bad], dtype=np.int64))
            except IndexError as exc:
                assert "vocab" in str(exc)
                continue
            raise AssertionError(f"id {bad} did not raise")


class _UvaStub:
    def __init__(self, arr):
        self.np = arr


def test_decode_warm_step_v2():
    """Drive the real hook body with a stub runner: window clamp at the
    buffer edge, stale-entry eviction, warmed_rows counted after completion,
    and drop-if-busy when a warm is already in flight."""
    from types import SimpleNamespace
    import time

    with tempfile.TemporaryDirectory() as d:
        t = _make_table(d)
        rng = np.random.default_rng(29)
        emb = _make_emb(rng)
        # Keep hashed ids inside the toy table (offs[-1] + sizes[-1] < vocab),
        # mirroring the production invariant that checkpoint sizes/offsets
        # tile [0, vocab).
        eos, mult, sizes, offs, n, hpn = emb._ple_host_hash
        sizes = np.full_like(sizes, t.vocab // sizes.size - 1)
        offs = np.concatenate(([0], np.cumsum(sizes[:-1])))
        emb._ple_host_hash = (eos, mult, sizes, offs, n, hpn)
        owner = SimpleNamespace(
            ngram_size=n, eos_token_id=eos,
            ngram_embedding=SimpleNamespace(table=t),
            _ple_host_hash=emb._ple_host_hash,
        )
        M._REGISTRY.clear()
        M._REGISTRY["layers.1.ple"] = owner
        width = 32
        toks = np.zeros((4, width), dtype=np.int64)
        toks[0, :] = rng.integers(0, 100, size=width)
        buf = SimpleNamespace(_uva_buf=_UvaStub(toks))
        done = np.zeros(8, dtype=np.int32)
        done[0] = width - 1  # optimistic count at the buffer edge
        rs = SimpleNamespace(
            req_id_to_index={"a": 0}, all_token_ids=buf,
            num_computed_tokens_np=done, num_speculative_steps=2,
        )
        runner = SimpleNamespace(req_states=rs)
        sched = SimpleNamespace(num_scheduled_tokens={"a": 2, "ghost": 1})
        # Ensure no warm from another test is still in flight, then snapshot.
        prior = M._WARM_PENDING[0]
        if prior is not None:
            prior.result(timeout=5)
        before = M._STATS["warmed_rows"]
        M._STALE_CHECK.clear()
        M._STALE_CHECK["gone"] = (0, 1)  # must be evicted (not scheduled)
        try:
            M._decode_warm_step_v2(runner, sched)  # end=done+q > width: clamps
            for _ in range(100):
                fut = M._WARM_PENDING[0]
                if fut is None or fut.done():
                    break
                time.sleep(0.05)
            assert fut is not None and fut.exception() is None, (
                f"warm failed: {fut and fut.exception()}"
            )
            assert M._STATS["warmed_rows"] > before, "no rows counted as warmed"
            assert "gone" not in M._STALE_CHECK, "finished request not evicted"
            assert "a" in M._STALE_CHECK
            # Drop-if-busy: with an unfinished future parked in _WARM_PENDING,
            # a second submission must be dropped and counted.
            from concurrent.futures import Future

            blocker = Future()  # never completed -> "warm still running"
            M._WARM_PENDING[0] = blocker
            dropped_before = M._STATS["warm_dropped"]
            M._decode_warm_step_v2(runner, sched)
            assert M._STATS["warm_dropped"] == dropped_before + 1, (
                "busy warm was not dropped"
            )
            assert M._WARM_PENDING[0] is blocker, "dropped submit replaced future"
        finally:
            M._REGISTRY.clear()
            M._STALE_CHECK.clear()
            M._WARM_PENDING[0] = None


def test_cpu_gather_byte_identity():
    with tempfile.TemporaryDirectory() as d:
        t = _make_table(d)
        rng = np.random.default_rng(3)
        # Fast (inline) path and pooled path, with duplicates.
        for n in (17, 2000):
            ids = rng.integers(0, t.vocab, size=n, dtype=np.int64)
            got = t.gather(ids)
            for k, gid in enumerate(ids.tolist()):
                sh, loc = gid // SHARD_SIZE, gid % SHARD_SIZE
                assert bytes(got[k]) == bytes(t.mm[sh][loc]), f"row {gid} differs"


def test_cpu_gather_range_check():
    with tempfile.TemporaryDirectory() as d:
        t = _make_table(d)
        try:
            t.gather(np.array([t.vocab + SHARD_SIZE], dtype=np.int64))
        except IndexError:
            return
        raise AssertionError("out-of-range id did not raise on the CPU path")


class _EmbStub:
    pass


def _make_emb(rng) -> _EmbStub:
    e = _EmbStub()
    n, hpn = 3, 16
    mult = rng.integers(1, np.iinfo(np.int64).max // 4, size=n, dtype=np.int64)
    # Large odd multipliers like the checkpoint's — guarantees int64 wraparound
    # (negative mixed values) on realistic token ids.
    mult = mult | 1
    n_heads = (n - 1) * hpn
    sizes = rng.integers(1000, 20_000_000, size=n_heads, dtype=np.int64)
    offs = np.concatenate(([0], np.cumsum(sizes[:-1])))
    e._ple_host_hash = (2, mult, sizes, offs, n, hpn)
    return e


def _torch_reference(emb, ctx: np.ndarray, q_len: int) -> np.ndarray:
    """Same computation in torch int64 with torch.remainder (floored, matching
    the model's tensor modulo). Divergence here = sign-handling bug."""
    eos, mult, sizes, offs, n, hpn = emb._ple_host_hash
    ctx_t = torch.from_numpy(np.ascontiguousarray(ctx, dtype=np.int64))
    n_reqs, width = ctx_t.shape
    positions = torch.arange(width, dtype=torch.int64)
    eos_pos = torch.where(ctx_t == eos, positions, torch.tensor(-1))
    prev_incl = torch.cummax(eos_pos, dim=1).values
    prev = torch.cat(
        [torch.full((n_reqs, 1), -1, dtype=torch.int64), prev_incl[:, :-1]], dim=1
    )
    pos_in_seg = positions[None, :] - prev - 1
    mult_t = torch.from_numpy(mult)
    sizes_t = torch.from_numpy(sizes)
    offs_t = torch.from_numpy(offs)
    shifted = [ctx_t]
    for s in range(1, n):
        src = positions - s
        gathered = ctx_t[:, src.clamp(min=0)]
        valid = (src[None, :] >= 0) & (pos_in_seg >= s)
        shifted.append(torch.where(valid, gathered, torch.tensor(eos)))
    cols = torch.arange(width - q_len, width)
    blocks = []
    for ngram in range(2, n + 1):
        start = (ngram - 2) * hpn
        end = start + hpn
        mixed = shifted[0] * mult_t[0]
        for i in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[i] * mult_t[i])
        ids = torch.remainder(mixed[:, cols, None], sizes_t[start:end])
        blocks.append(ids + offs_t[start:end])
    return torch.cat(blocks, dim=-1).numpy()


def test_host_ngram_ids_matches_torch_remainder():
    rng = np.random.default_rng(23)
    emb = _make_emb(rng)
    saw_negative = False
    for trial in range(200):
        n_reqs = int(rng.integers(1, 5))
        q = int(rng.integers(1, 9))
        width = 2 + q  # n_ctx (= ngram_size-1 = 2) + q
        ctx = rng.integers(0, 160_000, size=(n_reqs, width), dtype=np.int64)
        if trial % 3 == 0:  # sprinkle EOS resets
            ctx[rng.random(ctx.shape) < 0.15] = 2
        # Prove the sign split is exercised at least once.
        _, mult, _, _, n, _ = emb._ple_host_hash
        mixed = ctx * mult[0]
        for i in range(1, n):
            mixed = np.bitwise_xor(mixed, ctx * mult[i])
        saw_negative = saw_negative or bool((mixed < 0).any())
        got = M.host_ngram_ids(emb, ctx, q)
        ref = _torch_reference(emb, ctx, q)
        assert got.shape == ref.shape
        assert (got == ref).all(), "host hash diverges from torch.remainder"
        assert (got >= 0).all()
    assert saw_negative, "test inputs never produced a negative mixed hash"


class _LogCapture(__import__("logging").Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _alarm_env():
    """Save module alarm state; yield (capture, advance, restore) helpers."""
    import logging
    import time

    saved = (dict(M._STATS), list(M._STATS_PREV_CUM), M._STATS_LAST[0],
             dict(M._DECODE_WARM), M._STATS_SEC, M.logger.level,
             M._ALARM_LAST[0], M.logger.propagate)
    cap = _LogCapture()
    M.logger.addHandler(cap)
    # Pin the level: under a host logging config that silences vllm.*, the
    # negative assertions would otherwise pass vacuously. Propagation off so
    # alarm lines don't spray the test output via the root logger.
    M.logger.setLevel(logging.INFO)
    M.logger.propagate = False
    M._STATS_SEC = 1

    def advance(window_s: float = 2.0, origin: bool = False, **bumps):
        """Bump counters, then run _stats_log — either as the true origin
        call (_STATS_LAST untouched at 0.0, exercising the seeding branch)
        or with the window boundary forced `window_s` seconds back."""
        for k, v in bumps.items():
            M._STATS[k] += v
        if not origin:
            M._STATS_LAST[0] = time.monotonic() - window_s
        M._stats_log()

    def restore():
        M.logger.removeHandler(cap)
        M.logger.setLevel(saved[5])
        M.logger.propagate = saved[7]
        M._STATS.clear(); M._STATS.update(saved[0])
        M._STATS_PREV_CUM[:] = saved[1]
        M._STATS_LAST[0] = saved[2]
        M._DECODE_WARM.clear(); M._DECODE_WARM.update(saved[3])
        M._STATS_SEC = saved[4]
        M._ALARM_LAST[0] = saved[6]

    return cap, advance, restore


def _fresh_alarm_state():
    for k in M._STATS:
        M._STATS[k] = 0 if isinstance(M._STATS[k], int) else 0.0
    M._STATS_PREV_CUM[:] = [0] * len(M._STATS_PREV_CUM)
    M._STATS_LAST[0] = 0.0
    # The real module default ("never warned"), NOT 0.0: monotonic() is
    # boot-relative, so 0.0 would suppress warnings on hosts with < 60 s
    # uptime and make these tests uptime-flaky.
    M._ALARM_LAST[0] = float("-inf")
    M._DECODE_WARM.clear()
    M._DECODE_WARM.update({"installed": True})


def test_alarms_healthy_silent():
    cap, advance, restore = _alarm_env()
    try:
        _fresh_alarm_state()
        advance(origin=True)  # true origin call: seeding branch
        for _ in range(5):
            advance(calls=3, hook_calls=10, warm_attempts=10, warmed_rows=480)
        assert not [m for m in cap.messages if "PLE mmap health" in m or "never invoked" in m], cap.messages
        assert M._DECODE_WARM.get("proven_reachable")
    finally:
        restore()


def test_alarm_origin_seeds_delta_base():
    # Counters accumulated BEFORE the origin call (boot activity) must not
    # leak into the first window's deltas and fire alarms.
    cap, advance, restore = _alarm_env()
    try:
        _fresh_alarm_state()
        M._STATS["warm_errors"] = 500       # pre-origin history
        M._STATS["hook_calls"] = 500
        advance(origin=True)                # seeds _STATS_PREV_CUM
        assert M._STATS_PREV_CUM[3] == 500, "origin did not seed delta base"
        advance(calls=2, hook_calls=5, warm_attempts=5, warmed_rows=240)
        assert not [m for m in cap.messages if "PLE mmap health" in m or "never invoked" in m], cap.messages
    finally:
        restore()


def test_alarm_unhealthy_windows_fire():
    # Each unhealthy shape produces the single health warning: hook errors
    # (thrown before any attempt), failed launches, attempts with zero
    # completions.
    cap, advance, restore = _alarm_env()
    try:
        for bumps in (
            dict(calls=1, hook_calls=30, warm_errors=30),
            dict(calls=1, launch_errors=2),
            dict(calls=1, hook_calls=20, warm_attempts=5, warm_dropped=5),
        ):
            _fresh_alarm_state()
            advance(origin=True)
            advance(**bumps)
            health = [m for m in cap.messages if "PLE mmap health" in m]
            assert health, (bumps, cap.messages)
            if "launch_errors" in bumps:
                assert "ZEROED" in health[0]
            cap.messages.clear()
    finally:
        restore()


def test_alarm_module_default_is_never_warned():
    # The module-level _ALARM_LAST default must mean "never warned" so the
    # very first fault warns even at < 60 s host uptime (monotonic() is
    # boot-relative; a 0.0 sentinel would equal boot time). The fire tests
    # exercise the behavior via _fresh_alarm_state; this guards the module
    # default itself against regressing to 0.0 (runtime value may already
    # be a real timestamp by the time this test runs, so check the source).
    src = open(M.__file__).read()
    assert '_ALARM_LAST = [float("-inf")]' in src, (
        "_ALARM_LAST module default regressed from -inf"
    )


def test_alarm_wall_clock_rate_limit():
    # A persistent fault warns once, then stays quiet for the 60 s wall
    # limit — regardless of how short the stats window is.
    cap, advance, restore = _alarm_env()
    try:
        _fresh_alarm_state()
        advance(origin=True)
        for _ in range(5):
            advance(calls=1, hook_calls=30, warm_errors=30)
        health = [m for m in cap.messages if "PLE mmap health" in m]
        assert len(health) == 1, cap.messages
        # Fault persisting past the wall limit re-warns (no latch).
        M._ALARM_LAST[0] -= 61.0
        advance(calls=1, hook_calls=30, warm_errors=30)
        health = [m for m in cap.messages if "PLE mmap health" in m]
        assert len(health) == 2, cap.messages
    finally:
        restore()


def test_alarm_prefill_only_silent():
    # Prefill-only traffic: hook invoked, no warmable work — no alarms, and
    # the wrong-runner alarm is disarmed by proof of reachability.
    cap, advance, restore = _alarm_env()
    try:
        _fresh_alarm_state()
        advance(origin=True)
        for _ in range(25):
            advance(calls=2, hook_calls=4, window_s=30.0)
        assert not [m for m in cap.messages if "PLE mmap health" in m or "never invoked" in m], cap.messages
    finally:
        restore()


def test_alarm_wrong_runner_after_grace():
    # Eager traffic, hook never invoked: silent until 300 s of busy time
    # accumulate, then a one-shot warning. Busy time is clamped per window
    # (2 x STATS_SEC) so idle gaps don't count — hence STATS_SEC=30 here:
    # each 31 s window credits ~31 s (< 60 clamp).
    cap, advance, restore = _alarm_env()
    try:
        _fresh_alarm_state()
        M._STATS_SEC = 30
        advance(origin=True)
        for _ in range(9):  # ~9 x 31 s ≈ 280 s busy
            advance(calls=2, window_s=31.0)
        assert not any("never invoked" in m for m in cap.messages), cap.messages
        advance(calls=2, window_s=31.0)  # crosses 300 s
        assert any("never invoked" in m for m in cap.messages), cap.messages
        cap.messages.clear()
        advance(calls=2, window_s=31.0)  # latched: no repeat
        assert not any("never invoked" in m for m in cap.messages), cap.messages
    finally:
        restore()


def test_bool_default():
    import inspect

    def f(a, flag=False, truthy=True, sentinel=object(), none=None,
          npf=np.bool_(False), npt=np.bool_(True), zero=0, one=1): ...
    p = inspect.signature(f).parameters
    assert M._bool_default(p["flag"]) is False
    assert M._bool_default(p["truthy"]) is None      # truthy bool -> refuse:
    # a True default would make the omitted-arg probe skip warming forever
    assert M._bool_default(p["sentinel"]) is None    # truthy sentinel -> refuse
    assert M._bool_default(p["a"]) is False          # empty default
    assert M._bool_default(p["none"]) is False       # falsy accepted
    assert M._bool_default(p["npf"]) is False        # falsy numpy bool
    assert M._bool_default(p["npt"]) is None         # truthy numpy bool
    assert M._bool_default(p["zero"]) is False       # falsy int
    assert M._bool_default(p["one"]) is None         # truthy int


def test_prewarm_runs():
    with tempfile.TemporaryDirectory() as d:
        t = _make_table(d)
        t.prewarm()  # smoke: no exception, exact loop bounds


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} unit tests passed")
