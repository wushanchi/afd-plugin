# SPDX-License-Identifier: Apache-2.0
"""CPU-safe contract tests for the V4.1 Window U1 seam."""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

import torch

from afd_plugin.connectors.npu.window import WindowAFDExtraInfo
from afd_plugin.distributed.topology import build_window_expert_layout
from afd_plugin.model_executor.models.npu.deepseek_v2_attention_gate import (
    compute_attention_gate_moe_ffn,
)
from afd_plugin.model_executor.models.npu.deepseek_v41_window_contract import (
    V41MHCContinuation,
    V41MHCContinuationKey,
    V41MHCContinuationRegistry,
    validate_window_activation_shape,
)


def _key(*, layer_idx: int = 0) -> V41MHCContinuationKey:
    return V41MHCContinuationKey(
        generation_id="request-1",
        ubatch_idx=0,
        layer_idx=layer_idx,
        stage_idx=0,
        token_order=(4, 7),
    )


def _continuation(layer_idx: int = 0) -> V41MHCContinuation:
    return V41MHCContinuation(
        key=_key(layer_idx=layer_idx),
        pre_mix_in="pre_mix",
        attn_pre="attn_pre",
        residual="residual",
        post_mix="post_mix",
        res_mix="res_mix",
        valid_tokens=2,
    )


def test_registry_returns_continuation_once_and_rejects_reuse() -> None:
    registry = V41MHCContinuationRegistry()
    item = _continuation()

    registry.put(item)
    assert registry.pop(item.key) is item
    with pytest.raises(RuntimeError, match="missing V4.1 mHC continuation"):
        registry.pop(item.key)


def test_registry_discards_only_one_generation() -> None:
    registry = V41MHCContinuationRegistry()
    stale = _continuation(layer_idx=0)
    current = V41MHCContinuation(
        key=V41MHCContinuationKey("request-2", 0, 0, 0, (4, 7)),
        pre_mix_in="pre_mix",
        attn_pre="attn_pre",
        residual="residual",
        post_mix="post_mix",
        res_mix="res_mix",
        valid_tokens=2,
    )
    registry.put(stale)
    registry.put(current)

    registry.discard_generation("request-1")
    assert len(registry) == 1
    assert registry.pop(current.key) is current


@pytest.mark.parametrize("shape", [(2, 2, 8), (2,)])
def test_window_activation_requires_two_dimensional_token_hidden_shape(shape) -> None:
    with pytest.raises(ValueError, match=r"\[tokens, hidden_size\]"):
        validate_window_activation_shape(shape, token_count=2, hidden_size=8)


def test_window_expert_layout_reserves_shared_slot_before_routed_shards() -> None:
    shared = build_window_expert_layout(
        routed_expert_num=8,
        ffn_size=3,
        ffn_rank=0,
    )
    routed0 = build_window_expert_layout(
        routed_expert_num=8,
        ffn_size=3,
        ffn_rank=1,
    )
    routed1 = build_window_expert_layout(
        routed_expert_num=8,
        ffn_size=3,
        ffn_rank=2,
    )

    assert (shared.kind, shared.local_expert_start, shared.local_expert_count) == (
        "shared",
        8,
        1,
    )
    assert (routed0.kind, routed0.local_expert_start, routed0.local_expert_count) == (
        "routed",
        0,
        4,
    )
    assert (routed1.kind, routed1.local_expert_start, routed1.local_expert_count) == (
        "routed",
        4,
        4,
    )


def test_window_extra_config_keeps_u1_single_slot_and_quant_mode() -> None:
    assert WindowAFDExtraInfo.from_mapping(
        {"micro_batch_num": 1, "quant_mode": 4}
    ) == WindowAFDExtraInfo(micro_batch_num=1, quant_mode=4)
    with pytest.raises(ValueError, match="micro_batch_num"):
        WindowAFDExtraInfo.from_mapping({"micro_batch_num": 2})
    with pytest.raises(ValueError, match="quant_mode"):
        WindowAFDExtraInfo.from_mapping({"quant_mode": 3})


def test_shared_window_rank_bypasses_routed_expert_abi() -> None:
    class Shared(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states + 1

    layer = SimpleNamespace(
        mlp=SimpleNamespace(
            experts=None,
            shared_experts=Shared(),
        ),
    )
    hidden_states = torch.zeros((2, 4), dtype=torch.float32)

    payload = compute_attention_gate_moe_ffn(
        layer,
        hidden_states=hidden_states,
        group_list=torch.empty((0,), dtype=torch.int32),
        dynamic_scales=None,
        expand_x_shared=None,
        dynamic_scales_shared=None,
        topk_scales=None,
        group_list_type=1,
    )

    assert torch.equal(payload.routed_output, torch.ones_like(hidden_states))
    assert payload.shared_output is None
