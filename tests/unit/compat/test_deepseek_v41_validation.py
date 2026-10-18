from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from afd_plugin.compat.npu.feature_validation import (
    _fail_if_unsupported_deepseek_v4_features,
)
from afd_plugin.config import AFDConfig


def test_deepseek_v41_window_stays_blocked_until_window_metadata_is_supported():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            enforce_eager=True,
            hf_config=SimpleNamespace(
                architectures=["DeepseekV41ForCausalLM"],
            ),
        ),
        parallel_config=SimpleNamespace(
            use_ubatching=False,
        ),
        speculative_config=None,
    )
    afd_config = AFDConfig(
        connector="WindowAFDConnector",
        role="attention",
        num_attention_ranks=1,
        num_ffn_ranks=2,
    )

    with pytest.raises(RuntimeError, match="only P2pHcclAFDConnector"):
        _fail_if_unsupported_deepseek_v4_features(config, afd_config)
