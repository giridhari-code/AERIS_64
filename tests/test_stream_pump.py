"""Streaming cancellation safety: the model lock must never be stranded.

Regression guard for a real bug: when an SSE client disconnected mid-stream,
Starlette abandoned the (sync) response generator suspended inside
``_token_stream``'s ``with _lock`` block, so the lock was never released —
every later request then blocked forever at zero CPU until a server restart.
``_TokenPump`` drives the token generator from a plain worker thread that
always reaches the end of the ``with`` block (or closes the generator), so
the lock is released even if the consumer vanishes without a cleanup.
"""

import asyncio
import time

from sentinel.serving import server as srv


def _fake_stream(hold_s: float = 0.05, tokens: int = 6):
    """Mimic _token_stream: hold the real model lock while yielding tokens."""

    def factory():
        with srv._lock:
            for i in range(tokens):
                time.sleep(hold_s)  # simulate one token step
                yield i, None

    return factory


def _lock_is_free(timeout: float = 5.0) -> bool:
    got = srv._lock.acquire(timeout=timeout)
    if got:
        srv._lock.release()
    return got


def test_pump_releases_lock_on_cancel():
    async def main():
        pump = srv._TokenPump(asyncio.get_running_loop(), _fake_stream())
        pump.start()
        kind, payload = await pump.next_event()
        assert (kind, payload) == ("token", 0)
        pump.signal_cancel()
        while True:
            kind, payload = await pump.next_event()
            if kind == "done":
                assert payload is True  # reported as aborted
                break
            assert kind == "token"
        # The worker must have unwound `with _lock`.
        assert await asyncio.to_thread(_lock_is_free)
        pump._thread.join(timeout=5)
        assert not pump._thread.is_alive()

    asyncio.run(main())


def test_pump_releases_lock_when_consumer_is_abandoned():
    """No cancel signal (the exact stranding scenario): the pump still drains
    the generator to completion, so the lock is released without a restart."""

    async def main():
        pump = srv._TokenPump(asyncio.get_running_loop(), _fake_stream(tokens=3))
        pump.start()
        await pump.next_event()  # consume one event, then walk away
        await asyncio.to_thread(pump._thread.join, 10)
        assert not pump._thread.is_alive()
        assert await asyncio.to_thread(_lock_is_free)

    asyncio.run(main())


def test_pump_releases_lock_on_worker_error():
    """An exception inside the stream still frees the lock and reports it."""

    def factory():
        with srv._lock:
            yield 1, None
            raise RuntimeError("boom")

    async def main():
        pump = srv._TokenPump(asyncio.get_running_loop(), factory)
        pump.start()
        kinds = []
        while True:
            kind, payload = await pump.next_event()
            kinds.append(kind)
            if kind == "error":
                assert isinstance(payload, RuntimeError)
            if kind == "done":
                break
        assert kinds == ["token", "error", "done"]
        assert await asyncio.to_thread(_lock_is_free)

    asyncio.run(main())


def test_pump_does_not_block_event_loop():
    """The worker thread (not the loop) does the heavy lifting."""

    async def main():
        pump = srv._TokenPump(asyncio.get_running_loop(), _fake_stream(hold_s=0.2, tokens=4))
        pump.start()

        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        task = asyncio.create_task(ticker())
        while True:
            kind, _ = await pump.next_event()
            if kind == "done":
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # A blocked loop would show very few ticks over ~0.8s of pumping.
        assert ticks > 20

    asyncio.run(main())
