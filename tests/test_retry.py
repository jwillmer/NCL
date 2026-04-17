"""Tests for ``mtss._io.retry_with_backoff``.

Covers the shared exponential-backoff helper extracted from
``archive_storage.list_folder`` and ``import_cmd._upload_with_retry`` so both
retry loops use a single, well-tested implementation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mtss._io import retry_with_backoff


class TestRetryWithBackoff:
    def test_backoff_progression(self):
        """Fails twice then succeeds; sleep called with [1.0, 2.0]."""
        sleep_mock = MagicMock()
        attempts = {"n": 0}

        def fn() -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError(f"boom-{attempts['n']}")
            return "ok"

        result = retry_with_backoff(
            fn,
            max_attempts=3,
            backoff_base=1.0,
            sleep=sleep_mock,
        )

        assert result == "ok"
        assert attempts["n"] == 3
        delays = [c.args[0] for c in sleep_mock.call_args_list]
        assert delays == [1.0, 2.0]  # 1.0 * 2**0, 1.0 * 2**1

    def test_reraises_after_exhaustion(self):
        """After max_attempts=3 of consistent failure, the last exc propagates."""
        sleep_mock = MagicMock()
        exc = ValueError("persistent failure")

        def fn() -> None:
            raise exc

        with pytest.raises(ValueError, match="persistent failure") as exc_info:
            retry_with_backoff(
                fn,
                max_attempts=3,
                backoff_base=0.5,
                sleep=sleep_mock,
            )

        # The originally-raised exception propagates (identity check).
        assert exc_info.value is exc
        # 3 attempts => 2 sleeps between them.
        assert sleep_mock.call_count == 2
        delays = [c.args[0] for c in sleep_mock.call_args_list]
        assert delays == [0.5, 1.0]  # 0.5 * 2**0, 0.5 * 2**1

    def test_only_retries_on_allowed_exceptions(self):
        """Non-retriable exceptions propagate immediately without any sleep."""
        sleep_mock = MagicMock()
        attempts = {"n": 0}

        def fn() -> None:
            attempts["n"] += 1
            raise KeyError("nope")

        with pytest.raises(KeyError):
            retry_with_backoff(
                fn,
                max_attempts=3,
                backoff_base=1.0,
                retriable=(ValueError,),
                sleep=sleep_mock,
            )

        assert attempts["n"] == 1  # Called once, no retries.
        assert sleep_mock.call_count == 0

    def test_returns_value_on_success(self):
        """Happy path: no retries, value returned, no sleeps."""
        sleep_mock = MagicMock()

        def fn() -> int:
            return 42

        result = retry_with_backoff(fn, sleep=sleep_mock)

        assert result == 42
        assert sleep_mock.call_count == 0

    def test_on_retry_hook_receives_attempt_exc_delay(self):
        """``on_retry`` callback fires with 1-indexed attempt, exc, and delay."""
        sleep_mock = MagicMock()
        on_retry_calls: list[tuple[int, BaseException, float]] = []
        exc_1 = RuntimeError("first")
        exc_2 = RuntimeError("second")
        side_effects = iter([exc_1, exc_2, "done"])

        def fn() -> str:
            value = next(side_effects)
            if isinstance(value, BaseException):
                raise value
            return value

        def on_retry(attempt: int, exc: BaseException, delay: float) -> None:
            on_retry_calls.append((attempt, exc, delay))

        result = retry_with_backoff(
            fn,
            max_attempts=3,
            backoff_base=2.0,
            on_retry=on_retry,
            sleep=sleep_mock,
        )

        assert result == "done"
        assert len(on_retry_calls) == 2
        assert on_retry_calls[0] == (1, exc_1, 2.0)
        assert on_retry_calls[1] == (2, exc_2, 4.0)
