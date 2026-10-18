from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from afd_plugin.compat.npu.feature_validation import (
    _fail_if_unsupported_deepseek_v4_features,
)
from afd_plugin.config import AFDConfig


def test_deepseek_v41_window_u2_uses_global_ffn_contract():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            enforce_eager=True,
            hf_config=SimpleNamespace(
                architectures=["DeepseekV41ForCausalLM"],
            ),
        ),
        parallel_config=SimpleNamespace(
            use_ubatching=True,
            num_ubatches=2,
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_sequence_parallel_moe=False,
            enable_eplb=False,
        ),
        speculative_config=None,
    )
    afd_config = AFDConfig(
        connector="WindowAFDConnector",
        role="attention",
        compute_gate_on_attention=True,
        num_attention_ranks=1,
        num_ffn_ranks=2,
    )

    _fail_if_unsupported_deepseek_v4_features(config, afd_config)
