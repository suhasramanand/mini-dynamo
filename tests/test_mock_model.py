"""Tests for the mock model backend and KV-transfer cost model."""

from common.config import Settings
from common.kv_transfer import transfer_ms
from common.mock_model import MockModel, count_tokens


def _settings(**kw):
    base = dict(
        prefill_base_ms=1.0, prefill_ms_per_token=0.1, decode_ms_per_token=1.0,
        latency_jitter=0.0, kv_mb_per_token=0.5,
    )
    base.update(kw)
    return Settings(**base)


def test_count_tokens():
    assert count_tokens("hello world") == 2
    assert count_tokens("") == 1          # never zero
    assert count_tokens("   ") == 1


async def test_prefill_produces_sized_kv():
    m = MockModel(_settings())
    res = await m.prefill("sess", "one two three four")
    assert res.num_prompt_tokens == 4
    assert res.kv_size_mb == 4 * 0.5
    assert res.kv_cache_id.startswith("kv-")
    assert res.prefill_ms > 0


def test_next_token_is_deterministic_per_session_index():
    m = MockModel(_settings())
    a = m.next_token("sess", 3)
    b = m.next_token("sess", 3)
    c = m.next_token("other", 3)
    assert a == b            # same session+index -> same token
    assert isinstance(a, str) and a.endswith(" ")
    # Different session very likely differs (not guaranteed, but check type).
    assert isinstance(c, str)


def test_transfer_cost_model():
    # colocated -> free
    assert transfer_ms(100, 25, 2, colocated=True) == 0.0
    # 25 Gbps = 3.125 MB/ms ; 100MB -> 32ms + 2ms fixed = 34ms
    ms = transfer_ms(100, 25, 2, colocated=False)
    assert abs(ms - 34.0) < 0.01
    # zero-size -> free
    assert transfer_ms(0, 25, 2) == 0.0
