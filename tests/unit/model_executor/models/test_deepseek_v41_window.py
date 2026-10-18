from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm_ascend")

# noqa: E402 - imports are intentionally delayed until the optional runtime
# dependency above has been confirmed.
from afd_plugin.distributed.topology import AFDWindowExpertLayout  # noqa: E402
from afd_plugin.model_executor.models.npu.deepseek_v41_window import (  # noqa: E402
    _cumulative_group_list,
    _install_local_expert_map,
    selected_expert_num,
    v41_window_expert_holder,
)


def test_v41_window_compact_group_list_is_normalized_to_cumulative():
    compact = torch.tensor([[0, 2], [5, 1], [12, 3]], dtype=torch.int32)

    cumulative, group_list_type = _cumulative_group_list(compact, 1)

    assert group_list_type == 0
    assert torch.equal(cumulative, torch.tensor([2, 3, 6], dtype=torch.int32))


def test_v41_window_cumulative_group_list_is_preserved():
    cumulative = torch.tensor([2, 3, 6], dtype=torch.int32)

    result, group_list_type = _cumulative_group_list(cumulative, 0)

    assert group_list_type == 0
    assert result is cumulative


def test_v41_window_payload_width_includes_one_shared_slot():
    config = SimpleNamespace(num_experts_per_tok=6, n_shared_experts=1)

    assert selected_expert_num(config) == 7


def test_v41_window_installs_global_to_local_map_on_current_runner_api():
    holder = SimpleNamespace(
        expert_map_manager=SimpleNamespace(),
        moe_config=SimpleNamespace(),
    )
    runner = SimpleNamespace(routed_experts=holder)
    layout = AFDWindowExpertLayout(
        kind="routed",
        local_expert_start=128,
        local_expert_count=128,
    )

    _install_local_expert_map(
        runner,
        routed_expert_num=384,
        layout=layout,
    )

    expert_map = holder._expert_map
    assert expert_map.shape == (384,)
    assert torch.equal(expert_map[:128], torch.full((128,), -1, dtype=torch.int32))
    assert torch.equal(expert_map[128:256], torch.arange(128, dtype=torch.int32))
    assert torch.equal(expert_map[256:], torch.full((128,), -1, dtype=torch.int32))
    assert holder.local_num_experts == 128
    assert holder.global_num_experts == 384
    assert holder.moe_config.num_experts == 384
    assert holder.moe_config.num_logical_experts == 384
    assert holder.moe_config.num_local_experts == 128


def test_v41_window_expert_holder_handles_current_runner_shape():
    holder = SimpleNamespace()
    runner = SimpleNamespace(experts=SimpleNamespace(routed_experts=holder))

    assert v41_window_expert_holder(runner) is holder
