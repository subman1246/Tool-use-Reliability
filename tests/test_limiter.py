"""Rate limiter and daily-cap tests. No network.

These exist because the failure mode they guard against is silent: if the
limiter miscounts or a daily 429 is mistaken for a transient one, the run keeps
going and fills the log with backend_error records, producing a truncated sweep
that looks complete to every downstream analysis step.
"""

import sys
import time

sys.path.insert(0, "src")

from tur.harness.runner import (RateLimiter, DailyCapReached,
                                _is_daily_cap_error, _count_tokens)


def test_rpd_cap_raises_at_configured_count():
    lim = RateLimiter(tpm=None, rpd=3)
    for _ in range(3):
        lim.acquire(10)
    try:
        lim.acquire(10)
        raise AssertionError("expected DailyCapReached")
    except DailyCapReached:
        pass
    assert lim.n_requests == 3


def test_requests_counted_without_a_tpm_ceiling():
    """Regression: an early return skipped the counter when tpm was None,
    so the daily cap never tripped for an unpaced model."""
    lim = RateLimiter(tpm=None, rpd=100)
    for _ in range(5):
        lim.acquire(10)
    assert lim.n_requests == 5


def test_provider_remaining_trips_the_reserve():
    lim = RateLimiter(tpm=None, rpd=100000, reserve_requests=5)
    lim.acquire(10)
    lim.sync_from_headers({"x-ratelimit-remaining-requests": "4"})
    try:
        lim.acquire(10)
        raise AssertionError("expected DailyCapReached")
    except DailyCapReached:
        pass


def test_header_sync_is_tolerant_of_junk():
    lim = RateLimiter(tpm=1000, rpd=1000)
    lim.sync_from_headers({})
    lim.sync_from_headers({"x-ratelimit-remaining-requests": "not-a-number"})
    lim.sync_from_headers(None)
    assert lim.remaining_requests is None


def test_tpm_window_accumulates_under_budget_without_sleeping():
    lim = RateLimiter(tpm=1000, rpd=None, headroom=0.8)   # budget 800/min
    t0 = time.time()
    lim.acquire(300)
    lim.acquire(300)
    assert time.time() - t0 < 0.5, "should not sleep while under budget"
    assert sum(t for _, t in lim._window) == 600
    assert lim.sleep_seconds == 0.0


def test_window_prunes_entries_older_than_a_minute():
    lim = RateLimiter(tpm=1000, rpd=None, headroom=0.8)
    lim._window.append((time.time() - 61.0, 700))
    assert lim._prune(time.time()) == 0


def test_daily_vs_transient_classification():
    daily = ["RateLimitError: 429 rate limit exceeded, requests per day",
             "Error 429: quota exceeded for this project",
             "RESOURCE_EXHAUSTED",
             "You exceeded your current quota, please check your plan"]
    transient = ["RateLimitError: 429 rate limit reached, please retry in 2s",
                 "APIConnectionError: connection reset",
                 "500 internal server error",
                 "Timeout waiting for response"]
    for m in daily:
        assert _is_daily_cap_error(Exception(m)), m
    for m in transient:
        assert not _is_daily_cap_error(Exception(m)), m


def test_token_count_includes_output_allowance():
    n = _count_tokens([{"role": "user", "content": "hello world " * 100}])
    assert 150 < n < 400, n


def test_no_limits_configured_is_a_noop():
    lim = RateLimiter()
    for _ in range(50):
        lim.acquire(10_000)
    assert lim.n_requests == 50 and lim.sleep_seconds == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  {fn.__name__}: OK")
    print(f"\nALL {len(fns)} LIMITER TESTS PASSED")
