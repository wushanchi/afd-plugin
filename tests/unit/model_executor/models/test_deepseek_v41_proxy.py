# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm_ascend")

from afd_plugin.model_executor.models import deepseek_v2 as proxy_runtime  # noqa: E402
from afd_plugin.model_executor.models import deepseek_v41 as adapter  # noqa: E402


class _RecordingConnector:
    yield_after_attn_send = False

    def __init__(self):
        self.sent = []
        self.received = []

    def send_attn_output(self, hidden_states, context, **kwargs):
        self.sent.append((hidden_states.clone(), context, kwargs))

    def recv_ffn_output(self, *, ref_tensor, ubatch_idx, **kwargs):
        self.received.append((ref_tensor, ubatch_idx, kwargs))
        return ref_tensor


def _install_forward_context(monkeypatch, connector, *, input_ids=None, ubatch_idx=1):
    metadata = SimpleNamespace(connector=connector, stage_idx=0)
    forward_context = SimpleNamespace(ubatch_idx=ubatch_idx, input_ids=input_ids)
    monkeypatch.setattr(
        proxy_runtime,
        "get_afd_metadata_from_forward_context",
        lambda: metadata,
    )
    monkeypatch.setattr(
        proxy_runtime,
        "get_forward_context",
        lambda: forward_context,
    )
    monkeypatch.setattr(
        proxy_runtime,
        "maybe_apply_dbo_yield",
        lambda tensor, *, role: tensor,
    )
    return forward_context


def test_v41_layer_zero_forwards_raw_input_ids_once(monkeypatch):
    connector = _RecordingConnector()
    input_ids = torch.tensor([11, 12, 13], dtype=torch.int32)
    _install_forward_context(monkeypatch, connector, input_ids=input_ids)

    proxy = adapter.AFDDeepseekV41RemoteMoEProxy(layer_idx=0)
    hidden_states = torch.ones((3, 8), dtype=torch.bfloat16)
    output = proxy(hidden_states)

    assert output.shape == hidden_states.shape
    assert len(connector.sent) == 1
    sent_hidden, context, send_kwargs = connector.sent[0]
    assert torch.equal(sent_hidden, hidden_states)
    assert send_kwargs["input_ids"] is input_ids
    assert context.metadata.layer_idx == 0
    assert context.metadata.phase == "decoder"
    assert context.metadata.stage_idx == 1
    assert connector.received[0][1] == 1
    # The standalone proxy path does not add a layer keyword to receive;
    # layer identity is already part of the transfer metadata.
    assert connector.received[0][2] == {}


def test_v41_later_layers_do_not_repeat_input_ids_side_channel(monkeypatch):
    connector = _RecordingConnector()
    _install_forward_context(
        monkeypatch,
        connector,
        input_ids=torch.tensor([21, 22], dtype=torch.int32),
        ubatch_idx=0,
    )

    proxy = adapter.AFDDeepseekV41RemoteMoEProxy(layer_idx=7)
    proxy(torch.ones((2, 8), dtype=torch.bfloat16))

    assert len(connector.sent) == 1
    assert "input_ids" not in connector.sent[0][2]
    assert connector.sent[0][1].metadata.layer_idx == 7
    assert connector.received[0][2] == {}


def test_v41_layer_zero_requires_raw_input_ids(monkeypatch):
    connector = _RecordingConnector()
    _install_forward_context(monkeypatch, connector, input_ids=None)

    proxy = adapter.AFDDeepseekV41RemoteMoEProxy(layer_idx=0)
    with pytest.raises(RuntimeError, match="layer 0 requires input_ids"):
        proxy(torch.ones((1, 8), dtype=torch.bfloat16))

    assert connector.sent == []

