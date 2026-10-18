# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

pytest.importorskip("vllm_ascend")

from afd_plugin.model_executor.models.deepseek_v41 import (  # noqa: E402
    AFDDeepseekV41ForCausalLM,
    _checkpoint_weight_roles,
    _iter_role_weights,
)


@pytest.mark.parametrize(
    ("name", "roles"),
    [
        ("model.embed_tokens.weight", {"attention"}),
        ("model.norm.weight", {"attention"}),
        ("model.layers.0.self_attn.wq_a.weight", {"attention"}),
        ("model.layers.0.input_layernorm.weight", {"attention"}),
        ("model.layers.0.hc_ffn_fn", {"attention"}),
        ("model.layers.0.mlp.gate.weight", {"ffn"}),
        ("model.layers.0.ffn.gate.weight", {"ffn"}),
        ("model.layers.0.mlp.experts.7.w1.weight", {"ffn"}),
        ("model.layers.0.mlp.shared_experts.gate_proj.weight", {"ffn"}),
        ("mtp.0.layers.0.mlp.experts.0.w1.weight", set()),
    ],
)
def test_checkpoint_weight_roles(name, roles):
    assert _checkpoint_weight_roles(name) == roles


def test_role_weight_iterator_is_one_shot_and_filters_layer_count():
    class OneShot:
        consumed = False

        def __iter__(self):
            if self.consumed:
                raise AssertionError("checkpoint iterator consumed twice")
            self.consumed = True
            yield "model.layers.0.mlp.gate.weight", object()
            yield "model.layers.40.mlp.gate.weight", object()
            yield "model.layers.0.self_attn.wq_a.weight", object()

    weights = OneShot()
    selected = list(
        _iter_role_weights(
            weights,
            role="ffn",
            num_hidden_layers=40,
        )
    )
    assert [name for name, _ in selected] == ["model.layers.0.mlp.gate.weight"]
    assert weights.consumed


def test_window_weight_iterator_separates_shared_and_routed_experts():
    weights = [
        ("model.layers.0.mlp.gate.weight", object()),
        ("model.layers.0.mlp.experts.7.w1.weight", object()),
        ("model.layers.0.mlp.shared_experts.gate_proj.weight", object()),
    ]

    attention = list(
        _iter_role_weights(
            weights,
            role="attention",
            num_hidden_layers=40,
            compute_gate_on_attention=True,
        )
    )
    shared = list(
        _iter_role_weights(
            weights,
            role="ffn",
            num_hidden_layers=40,
            window_ffn_kind="shared",
        )
    )
    routed = list(
        _iter_role_weights(
            weights,
            role="ffn",
            num_hidden_layers=40,
            window_ffn_kind="routed",
        )
    )

    assert [name for name, _ in attention] == ["model.layers.0.mlp.gate.weight"]
    assert [name for name, _ in shared] == [
        "model.layers.0.mlp.shared_experts.gate_proj.weight"
    ]
    assert [name for name, _ in routed] == ["model.layers.0.mlp.experts.7.w1.weight"]


def test_v41_causal_lm_exposes_layer_major_u2_hook():
    """The NPU ubatch wrapper is installed around the CausalLM object."""

    assert callable(AFDDeepseekV41ForCausalLM.forward_ubatches_layer_major)
