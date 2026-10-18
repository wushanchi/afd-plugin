# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.model_executor.models.npu.deepseek_v41_dspark import (  # noqa: E402
    DeepseekV41DSparkTargetState,
)


def test_dspark_target_state_uses_config_order_and_one_based_aux_ids():
    collector = DeepseekV41DSparkTargetState.from_config(
        SimpleNamespace(dspark_target_layer_ids=[4, 1]),
    )
    assert collector.target_layer_ids == (4, 1)
    assert collector.aux_layer_ids == (5, 2)
    assert collector.enabled

    collector.collect(1, torch.tensor([[[10.0, 20.0], [14.0, 26.0]]]))
    collector.collect(4, torch.tensor([[[1.0, 3.0], [5.0, 7.0]]]))
    flattened = collector.flatten()
    assert torch.equal(flattened, torch.tensor([[3.0, 5.0, 12.0, 23.0]]))


def test_dspark_target_state_rejects_partial_or_duplicate_capture():
    collector = DeepseekV41DSparkTargetState([0, 2])
    collector.collect(0, torch.ones((2, 2, 3)))
    with pytest.raises(RuntimeError, match="captured twice"):
        collector.collect(0, torch.ones((2, 2, 3)))
    with pytest.raises(RuntimeError, match="incomplete"):
        collector.flatten()


def test_dspark_target_state_store_checks_reusable_buffer_shape():
    collector = DeepseekV41DSparkTargetState([0])
    collector.collect(0, torch.arange(12, dtype=torch.float32).reshape(2, 2, 3))
    output = torch.zeros((4, 3))
    collector.store(output)
    assert torch.equal(output[:2], torch.tensor([[1.5, 2.5, 3.5], [7.5, 8.5, 9.5]]))
    with pytest.raises(ValueError, match="smaller"):
        collector.store(torch.zeros((1, 3)))
