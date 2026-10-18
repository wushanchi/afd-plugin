# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Small DSpark target-state contract shared by the V4.1 AFD wrapper.

The native V4.1 DSpark proposer consumes one hidden-state block per configured
target layer.  The AFD model runs those layers on the Attention role, so this
module keeps the capture/ordering rule independent from the model wrapper and
from connector transport code.  It deliberately does not implement the draft
model; ``vllm-ascend`` remains the owner of that implementation.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


class DeepseekV41DSparkTargetState:
    """Collect and flatten native V4.1 DSpark target-layer states.

    ``dspark_target_layer_ids`` are zero-based decoder-layer IDs in the
    checkpoint.  The EAGLE3 interface exposes the same layers as one-based
    ``aux_hidden_state_layers``; :attr:`aux_layer_ids` provides that mapping
    for the runner.  The captured representation matches native V4.1:
    ``hidden_states.mean(dim=1)`` followed by concatenation in configured
    target-layer order.
    """

    def __init__(self, target_layer_ids: Iterable[int] = ()) -> None:
        ids = tuple(int(layer_id) for layer_id in target_layer_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("DeepSeek-V4.1 DSpark target layer IDs must be unique")
        if any(layer_id < 0 for layer_id in ids):
            raise ValueError(
                "DeepSeek-V4.1 DSpark target layer IDs must be non-negative"
            )
        self.target_layer_ids = ids
        self.aux_layer_ids = tuple(layer_id + 1 for layer_id in ids)
        self._states: dict[int, torch.Tensor] = {}

    @classmethod
    def from_config(cls, config) -> DeepseekV41DSparkTargetState:
        """Build a collector from native V4.1 text/draft config fields."""

        return cls(getattr(config, "dspark_target_layer_ids", ()) or ())

    @property
    def enabled(self) -> bool:
        return bool(self.target_layer_ids)

    def clear(self) -> None:
        self._states.clear()

    def collect(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        """Capture one configured layer's HC state exactly once per step."""

        layer_idx = int(layer_idx)
        if layer_idx not in self.target_layer_ids:
            return
        if layer_idx in self._states:
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark target layer was captured twice: "
                f"layer={layer_idx}"
            )
        if hidden_states.ndim < 2:
            raise ValueError(
                "DeepSeek-V4.1 DSpark target state must include the HC dimension"
            )
        self._states[layer_idx] = hidden_states.mean(dim=1)

    def flatten(self) -> torch.Tensor:
        """Return target states in checkpoint order, or fail on a partial step."""

        missing = [
            layer_id
            for layer_id in self.target_layer_ids
            if layer_id not in self._states
        ]
        if missing:
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark target states are incomplete: "
                f"missing={missing}"
            )
        if not self.target_layer_ids:
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark requires at least one target layer"
            )
        return torch.cat(
            [self._states[layer_id] for layer_id in self.target_layer_ids],
            dim=-1,
        )

    def store(self, output_buffer: torch.Tensor) -> torch.Tensor:
        """Copy the flattened target state into a runner-owned reusable buffer."""

        flattened = self.flatten()
        if output_buffer.ndim != flattened.ndim or any(
            output_buffer.shape[dim] < flattened.shape[dim]
            for dim in range(flattened.ndim)
        ):
            raise ValueError(
                "DeepSeek-V4.1 DSpark output buffer is smaller than target states: "
                f"buffer={tuple(output_buffer.shape)} states={tuple(flattened.shape)}"
            )
        output_buffer[
            tuple(slice(0, size) for size in flattened.shape)
        ].copy_(flattened)
        return flattened


__all__ = ["DeepseekV41DSparkTargetState"]
