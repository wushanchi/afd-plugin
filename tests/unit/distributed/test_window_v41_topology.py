from __future__ import annotations

import pytest

from afd_plugin.distributed import build_window_expert_layout


@pytest.mark.parametrize(
    ("ffn_size", "expected_routed_count"),
    [
        (5, 96),
        (9, 48),
        (17, 24),
    ],
)
def test_deepseek_v41_window_layout_partitions_384_routed_experts(
    ffn_size,
    expected_routed_count,
):
    """The V4.1 384-expert checkpoint has a contiguous Window mapping."""

    layouts = [
        build_window_expert_layout(
            routed_expert_num=384,
            ffn_size=ffn_size,
            ffn_rank=ffn_rank,
        )
        for ffn_rank in range(ffn_size)
    ]

    assert layouts[0].kind == "shared"
    assert layouts[0].local_expert_start == 384
    assert layouts[0].local_expert_count == 1

    routed_layouts = layouts[1:]
    assert all(layout.kind == "routed" for layout in routed_layouts)
    assert all(
        layout.local_expert_count == expected_routed_count
        for layout in routed_layouts
    )
    assert [
        (layout.local_expert_start, layout.local_expert_count)
        for layout in routed_layouts
    ] == [
        (rank * expected_routed_count, expected_routed_count)
        for rank in range(ffn_size - 1)
    ]


def test_deepseek_v41_window_layout_rejects_more_ranks_than_routed_experts():
    with pytest.raises(ValueError, match="every routed FFN rank"):
        build_window_expert_layout(
            routed_expert_num=384,
            ffn_size=386,
            ffn_rank=0,
        )
