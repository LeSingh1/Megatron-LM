# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe import fused_a2a

MAX_NIC_TOKENS = 21840
FIRST_ILLEGAL_NIC_TOKENS = MAX_NIC_TOKENS + 1
LAST_LEGAL_ALIGNMENT_WINDOW = range(
    MAX_NIC_TOKENS - fused_a2a._HYBRID_EP_TOKEN_ALIGNMENT + 1, MAX_NIC_TOKENS + 1
)
FIRST_ILLEGAL_ALIGNMENT_WINDOW = range(
    FIRST_ILLEGAL_NIC_TOKENS, FIRST_ILLEGAL_NIC_TOKENS + fused_a2a._HYBRID_EP_TOKEN_ALIGNMENT
)


class ExistingBuffer:
    def __init__(self, num_of_nodes: int) -> None:
        self.num_of_nodes = num_of_nodes


class FakeGroup:
    def size(self) -> int:
        return 72


class FakeDispatchContext:
    pass


class FakeHybridEPBuffer:
    instances = []

    def __init__(
        self, *, group, hidden_dim, max_num_of_tokens_per_rank, num_local_experts, use_fp8, **kwargs
    ) -> None:
        self.num_of_nodes = 1
        self.max_num_of_tokens_per_rank = max_num_of_tokens_per_rank
        self.dispatched_num_tokens = None
        FakeHybridEPBuffer.instances.append(self)

    def dispatch_with_permute(self, *, hidden, probs, scaling_factor, **kwargs):
        self.dispatched_num_tokens = hidden.size(0)
        return hidden, probs, scaling_factor, torch.empty(0), ("handle",)


@pytest.fixture(autouse=True)
def reset_hybrid_ep_buffer(monkeypatch) -> None:
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", None)
    FakeHybridEPBuffer.instances.clear()


def test_hybrid_ep_ib_tx_depth_boundary_windows_match_qp_limit() -> None:
    last_legal_tx_depth = fused_a2a._hybrid_ep_ib_tx_depth(MAX_NIC_TOKENS)
    first_illegal_buffer_tokens = fused_a2a._hybrid_ep_buffer_tokens(FIRST_ILLEGAL_NIC_TOKENS)
    first_illegal_tx_depth = fused_a2a._hybrid_ep_ib_tx_depth(FIRST_ILLEGAL_NIC_TOKENS)

    assert fused_a2a._hybrid_ep_max_supported_tokens() == MAX_NIC_TOKENS
    assert MAX_NIC_TOKENS == 21840
    assert last_legal_tx_depth == 65521
    assert last_legal_tx_depth <= fused_a2a._HYBRID_EP_IB_QP_MAX_DEPTH
    assert first_illegal_buffer_tokens == 21856
    assert first_illegal_tx_depth == 65569
    assert first_illegal_tx_depth > fused_a2a._HYBRID_EP_IB_QP_MAX_DEPTH


def test_hybrid_ep_initial_buffer_tokens_preserves_legal_boundary() -> None:
    assert fused_a2a._hybrid_ep_initial_buffer_tokens(MAX_NIC_TOKENS) == MAX_NIC_TOKENS


def test_hybrid_ep_initial_buffer_tokens_caps_first_illegal_boundary() -> None:
    assert fused_a2a._hybrid_ep_initial_buffer_tokens(FIRST_ILLEGAL_NIC_TOKENS) == MAX_NIC_TOKENS


def test_hybrid_ep_dispatch_allows_oversized_first_call_after_single_nvl_resolution(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", "8")
    monkeypatch.setattr(fused_a2a, "HybridEPBuffer", FakeHybridEPBuffer, raising=False)
    x = torch.empty(FIRST_ILLEGAL_NIC_TOKENS, 4)
    routing_map = torch.zeros(FIRST_ILLEGAL_NIC_TOKENS, 2, dtype=torch.bool)

    dispatched_hidden, _, _, _, _ = fused_a2a.HybridEPDispatch.forward(
        FakeDispatchContext(), x, routing_map, None, FakeGroup(), 2
    )

    assert dispatched_hidden is x
    assert FakeHybridEPBuffer.instances[0].num_of_nodes == 1
    assert FakeHybridEPBuffer.instances[0].max_num_of_tokens_per_rank == MAX_NIC_TOKENS
    assert FakeHybridEPBuffer.instances[0].dispatched_num_tokens == FIRST_ILLEGAL_NIC_TOKENS


def test_hybrid_ep_ib_tx_depth_waits_for_initialized_buffer() -> None:
    fused_a2a._validate_hybrid_ep_ib_tx_depth(FIRST_ILLEGAL_NIC_TOKENS)


def test_hybrid_ep_ib_tx_depth_allows_first_nic_illegal_window_for_single_nvlink_domain(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", ExistingBuffer(num_of_nodes=1))

    for num_tokens in FIRST_ILLEGAL_ALIGNMENT_WINDOW:
        assert fused_a2a._hybrid_ep_ib_tx_depth(num_tokens) > fused_a2a._HYBRID_EP_IB_QP_MAX_DEPTH
        fused_a2a._validate_hybrid_ep_ib_tx_depth(num_tokens)


def test_hybrid_ep_ib_tx_depth_allows_last_legal_window_for_nic_path(monkeypatch) -> None:
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", ExistingBuffer(num_of_nodes=2))

    for num_tokens in LAST_LEGAL_ALIGNMENT_WINDOW:
        assert fused_a2a._hybrid_ep_ib_tx_depth(num_tokens) <= fused_a2a._HYBRID_EP_IB_QP_MAX_DEPTH
        fused_a2a._validate_hybrid_ep_ib_tx_depth(num_tokens)


def test_hybrid_ep_ib_tx_depth_rejects_first_illegal_window_for_nic_path(monkeypatch) -> None:
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", ExistingBuffer(num_of_nodes=2))

    for num_tokens in FIRST_ILLEGAL_ALIGNMENT_WINDOW:
        assert fused_a2a._hybrid_ep_buffer_tokens(num_tokens) == 21856
        assert fused_a2a._hybrid_ep_ib_tx_depth(num_tokens) == 65569
        with pytest.raises(ValueError, match="65569"):
            fused_a2a._validate_hybrid_ep_ib_tx_depth(num_tokens)
