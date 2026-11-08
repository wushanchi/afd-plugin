# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Small, backend-neutral contracts shared by the V4.1 Window U1 path.

The module intentionally contains no vLLM or Ascend imports.  It can therefore
be used by CPU-side metadata tests and by the NPU wrapper without duplicating
the continuation bookkeeping in the connector and model layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class V41MHCContinuationKey:
    """Identity for one request/ubatch/layer continuation."""

    generation_id: str
    ubatch_idx: int
    layer_idx: int
    stage_idx: int
    token_order: tuple[int, ...]


@dataclass(slots=True)
class V41MHCContinuation:
    """Attention-local mHC state retained across the Window FFN call."""

    key: V41MHCContinuationKey
    pre_mix_in: Any
    attn_pre: Any
    residual: Any
    post_mix: Any
    res_mix: Any
    valid_tokens: int
    ffn_pre: Any | None = None


class V41MHCContinuationRegistry:
    """Explicit continuation ownership for one runner/request scope."""

    def __init__(self) -> None:
        self._items: dict[V41MHCContinuationKey, V41MHCContinuation] = {}

    def put(self, continuation: V41MHCContinuation) -> None:
        if continuation.key in self._items:
            raise RuntimeError(f"duplicate V4.1 mHC continuation: {continuation.key}")
        self._items[continuation.key] = continuation

    def pop(self, key: V41MHCContinuationKey) -> V41MHCContinuation:
        try:
            return self._items.pop(key)
        except KeyError as exc:
            raise RuntimeError(f"missing V4.1 mHC continuation: {key}") from exc

    def discard(self, key: V41MHCContinuationKey) -> None:
        self._items.pop(key, None)

    def discard_generation(self, generation_id: str) -> None:
        stale = [key for key in self._items if key.generation_id == generation_id]
        for key in stale:
            del self._items[key]

    def __len__(self) -> int:
        return len(self._items)


def validate_window_activation_shape(
    shape: tuple[int, ...],
    *,
    token_count: int,
    hidden_size: int,
) -> None:
    """Validate the U1 transport boundary ``[tokens, hidden_size]``."""

    expected = (token_count, hidden_size)
    if tuple(shape) != expected:
        raise ValueError(
            "V4.1 Window activation must have shape [tokens, hidden_size]; "
            f"got {tuple(shape)}, expected {expected}"
        )


__all__ = [
    "V41MHCContinuation",
    "V41MHCContinuationKey",
    "V41MHCContinuationRegistry",
    "validate_window_activation_shape",
]
