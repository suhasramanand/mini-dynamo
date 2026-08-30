"""Tests for the simulated GPU memory + eviction policies."""

import pytest

from common.memory_sim import MemorySimulator, OutOfMemoryError


def test_basic_allocation_and_free():
    m = MemorySimulator(total_mb=100, eviction_policy="lru")
    assert m.allocate("a", "s1", 10, 40) == []
    assert m.used_mb == 40
    assert 0.39 < m.utilization < 0.41
    assert m.free("a") is True
    assert m.used_mb == 0
    assert m.free("missing") is False


def test_lru_eviction_picks_least_recently_used():
    m = MemorySimulator(total_mb=100, eviction_policy="lru")
    m.allocate("a", "s", 10, 40)
    m.allocate("b", "s", 10, 40)
    # Touch 'a' so 'b' becomes the least-recently-used victim.
    m.touch("a")
    evicted = m.allocate("c", "s", 10, 40)  # needs 40, only 20 free -> evict one
    assert evicted == ["b"]
    assert m.contains("a") and m.contains("c") and not m.contains("b")
    assert m.evictions == 1


def test_fifo_eviction_picks_oldest():
    m = MemorySimulator(total_mb=100, eviction_policy="fifo")
    m.allocate("a", "s", 10, 40)
    m.allocate("b", "s", 10, 40)
    m.touch("a")  # touch must NOT save 'a' under FIFO
    evicted = m.allocate("c", "s", 10, 40)
    assert evicted == ["a"]
    assert not m.contains("a")


def test_no_eviction_policy_ooms():
    m = MemorySimulator(total_mb=100, eviction_policy="none")
    m.allocate("a", "s", 10, 80)
    with pytest.raises(OutOfMemoryError):
        m.allocate("b", "s", 10, 40)   # no room, policy forbids eviction
    assert m.oom_events == 1


def test_block_larger_than_pool_ooms():
    m = MemorySimulator(total_mb=50, eviction_policy="lru")
    with pytest.raises(OutOfMemoryError):
        m.allocate("big", "s", 10, 100)


def test_reallocation_replaces_footprint():
    m = MemorySimulator(total_mb=100, eviction_policy="lru")
    m.allocate("a", "s", 10, 40)
    m.allocate("a", "s", 20, 60)   # same id, new size
    assert m.used_mb == 60
    assert m.num_blocks == 1
