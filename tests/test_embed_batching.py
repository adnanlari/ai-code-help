import time

from backend.indexing.embed import _batches, _est_tokens, _RollingLimiter


def test_batches_respect_token_budget():
    texts = ["x" * 4000] * 10  # ~1000 est-tokens each
    batches = _batches(texts, max_batch_tokens=2500)  # ~2 per batch
    assert all(sum(_est_tokens(t) for t in b) <= 2500 for b in batches)
    assert sum(len(b) for b in batches) == 10  # nothing dropped


def test_batches_respect_item_cap():
    texts = ["x"] * 300  # tiny; only the 128-item cap can split these
    batches = _batches(texts, max_batch_tokens=10_000_000)
    assert [len(b) for b in batches] == [128, 128, 44]


def test_single_oversized_text_gets_its_own_batch():
    texts = ["x" * 40_000, "y" * 8]
    batches = _batches(texts, max_batch_tokens=1000)
    assert len(batches) == 2
    assert batches[0] == [texts[0]]


def test_limiter_disabled_is_noop():
    lim = _RollingLimiter(rpm=0, tpm=0)
    start = time.monotonic()
    for _ in range(50):
        lim.acquire(9999)
    assert time.monotonic() - start < 0.1


def test_limiter_blocks_on_rpm():
    lim = _RollingLimiter(rpm=2, tpm=0)
    # window is 60s and we don't want the test to actually sleep that long, so
    # just assert the 3rd acquire would need to wait (by checking event bookkeeping)
    lim.acquire(1)
    lim.acquire(1)
    assert len(lim._events) == 2
    now = time.monotonic()
    lim._trim(now)
    reqs = len(lim._events)
    assert reqs + 1 > lim.rpm  # 3rd request is over the limit -> would block
