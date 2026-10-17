# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Utilities for AFD model configuration."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import TYPE_CHECKING

from afd_plugin import _DEEPSEEK_MODEL_REGISTRATIONS

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig


# ### PATCH START: centralize DeepSeek-V4/V4.1 architecture contracts.
# Keep the raw and AFD-prefixed names together.  vLLM rewrites the selected
# architecture to ``AFD<architecture>`` on the AFD side, while the original
# name remains in the native model configuration.  DeepSeek-V4 and V4.1 use
# the token IDs for their hash/Engram routing contract, so connectors must
# make the decision from either spelling.
DEEPSEEK_INPUT_IDS_ARCHITECTURES = frozenset(
    {
        "DeepseekV4ForCausalLM",
        "AFDDeepseekV4ForCausalLM",
        "DeepseekV41ForCausalLM",
        "AFDDeepseekV41ForCausalLM",
        "DeepseekV41VisionForCausalLM",
        "AFDDeepseekV41VisionForCausalLM",
    },
)


DEEPSEEK_V4_FAMILY_ARCHITECTURES = frozenset(
    {
        "DeepseekV4ForCausalLM",
        "AFDDeepseekV4ForCausalLM",
        "DeepseekV41ForCausalLM",
        "AFDDeepseekV41ForCausalLM",
        "DeepseekV41VisionForCausalLM",
        "AFDDeepseekV41VisionForCausalLM",
    },
)


def architecture_requires_input_ids(architectures: Iterable[str] | None) -> bool:
    """Return whether an AFD model needs the raw token-ID side channel.

    DeepSeek-V4's hash router and DeepSeek-V4.1's Engram/routing paths consume
    the original token IDs.  The helper accepts ``None`` because incomplete
    or synthetic model configs are valid during CPU-only config tests.
    """

    return any(
        architecture in DEEPSEEK_INPUT_IDS_ARCHITECTURES
        for architecture in (architectures or ())
    )


def is_deepseek_v4_family_architecture(architectures: Iterable[str] | None) -> bool:
    """Return whether an architecture belongs to the DeepSeek-V4 family."""

    return any(
        architecture in DEEPSEEK_V4_FAMILY_ARCHITECTURES
        for architecture in (architectures or ())
    )


def is_deepseek_v41_architecture(architectures: Iterable[str] | None) -> bool:
    """Return whether an architecture uses the V4.1 native model contract."""

    return any(
        architecture
        in {
            "DeepseekV41ForCausalLM",
            "AFDDeepseekV41ForCausalLM",
            "DeepseekV41VisionForCausalLM",
            "AFDDeepseekV41VisionForCausalLM",
        }
        for architecture in (architectures or ())
    )


# ### PATCH END: centralize DeepSeek-V4/V4.1 architecture contracts.


def get_afd_model_config(model_config: ModelConfig) -> ModelConfig:
    """Return a model config that resolves to an AFD model implementation."""

    for model_arch in model_config.hf_config.architectures:
        if model_arch in _DEEPSEEK_MODEL_REGISTRATIONS:
            # deepcopy preserves aliasing within the copied object graph, so
            # the pure-text identity hf_text_config is hf_config is retained
            # automatically. vLLM Ascend uses that identity to distinguish
            # text models from multimodal models.
            afd_model_config = deepcopy(model_config)
            afd_model_config.hf_config.architectures = [f"AFD{model_arch}"]
            return afd_model_config
    return model_config


def install_afd_speculative_model_config(vllm_config: VllmConfig) -> None:
    """Resolve the draft model to its AFD role wrapper in-place."""
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is None:
        return
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    if draft_model_config is None:
        return
    speculative_config.draft_model_config = get_afd_model_config(
        draft_model_config,
    )
