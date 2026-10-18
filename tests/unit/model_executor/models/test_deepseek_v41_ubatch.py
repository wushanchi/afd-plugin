# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the DeepSeek-V4.1 AFD layer-major path.

The production adapter imports the optional vLLM-Ascend native model.  These
tests replace that optional dependency with small Python doubles, then drive
the adapter methods with real CPU tensors.  This keeps the tests useful on a
developer machine without CANN while still exercising the U1/U2 state
transitions rather than checking source text.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402


def _load_adapter_with_native_stubs():
    """Import the adapter once with a minimal CPU-only native model surface."""

    module_name = "afd_plugin.model_executor.models.deepseek_v41"
    if module_name in sys.modules:
        return sys.modules[module_name]

    import afd_plugin  # noqa: F401  # Import the real package/config first.

    saved: dict[str, ModuleType | None] = {}
    stub_names = (
        "vllm",
        "vllm.config",
        "vllm.forward_context",
        "vllm_ascend",
        "vllm_ascend.models",
        "vllm_ascend.models.deepseek_v41",
        "vllm_ascend.models.deepseek_v41.model",
        "vllm_ascend.models.deepseek_v41.vl_model",
        "vllm_ascend.utils",
        "afd_plugin.model_executor",
        "afd_plugin.model_executor.models",
        "afd_plugin.model_executor.models.deepseek_v2",
        "afd_plugin.model_executor.models.model_utils",
        "afd_plugin.model_executor.models.npu",
        "afd_plugin.model_executor.models.npu.deepseek_v41_dspark",
    )
    for name in stub_names:
        saved[name] = sys.modules.get(name)

    vllm = ModuleType("vllm")
    vllm.__path__ = []
    config = ModuleType("vllm.config")
    config.VllmConfig = object
    forward_context = ModuleType("vllm.forward_context")
    current_context = SimpleNamespace()

    def get_forward_context():
        return current_context

    @contextlib.contextmanager
    def override_forward_context(context):
        previous = current_context.__dict__.copy()
        current_context.__dict__.clear()
        current_context.__dict__.update(vars(context))
        try:
            yield
        finally:
            current_context.__dict__.clear()
            current_context.__dict__.update(previous)

    forward_context.get_forward_context = get_forward_context
    forward_context.override_forward_context = override_forward_context

    native = ModuleType("vllm_ascend.models.deepseek_v41.model")

    class NativeDecoderLayer(nn.Module):
        @staticmethod
        def hc_collapse(hidden_states, pre_mix):
            return hidden_states + pre_mix.to(hidden_states.dtype).unsqueeze(-1)

    class NativeModel(nn.Module):
        pass

    class NativeCausalLM(nn.Module):
        moe_comm_methods = {}

    class _IntermediateTensors(dict):
        pass

    class _MoE:
        @staticmethod
        def extract_moe_parameters(*_args, **_kwargs):
            return None

    native.DeepseekV41DecoderLayer = NativeDecoderLayer
    native.DeepseekV41Model = NativeModel
    native.AscendDeepseekV41LLMForCausalLM = NativeCausalLM
    native.DeepseekV41MixtureOfExperts = _MoE
    native.DeepseekV41MoE = nn.Identity
    native.DeepseekV41Attention = nn.Identity
    native.RMSNorm = nn.Identity
    native.PPMissingLayer = nn.Identity
    native.VocabParallelEmbedding = nn.Embedding
    native.ParallelLMHead = nn.Linear
    native.LogitsProcessor = nn.Identity
    native.IntermediateTensors = _IntermediateTensors
    native.DeepseekV41SharedAttentionState = SimpleNamespace
    native.current_platform = SimpleNamespace(device_type="cpu")
    native.get_pp_group = lambda: SimpleNamespace(
        is_first_rank=True,
        is_last_rank=True,
    )
    native.support_torch_compile = lambda **_kwargs: lambda cls: cls
    native.normalize_deepseek_v41_config = lambda config: config
    native.make_layers = lambda *_args, **_kwargs: (0, 0, [])
    native.maybe_prefix = lambda prefix, suffix: f"{prefix}.{suffix}"
    native.build_layer_plan = lambda _config: SimpleNamespace(candidate_topk_blocks=1)
    native.sp_all_gather = lambda x: x
    native.sp_reduce_scatter = lambda x: x

    native_vl = ModuleType("vllm_ascend.models.deepseek_v41.vl_model")
    native_vl.AscendDeepseekV41ForCausalLM = NativeCausalLM
    native_utils = ModuleType("vllm_ascend.utils")
    native_utils.is_dspark_config = lambda _config: False

    @dataclass(frozen=True)
    class Transfer:
        connector: Any
        context: Any
        stage_idx: int
        ref_tensor: torch.Tensor

    class RemoteFFNProxy(nn.Module):
        def __init__(self, *, layer_idx: int, phase: str = "decoder"):
            super().__init__()
            self.layer_idx = layer_idx
            self.phase = phase

        def dispatch_remote_ffn(self, hidden_states, **_kwargs):
            return Transfer(None, None, 0, hidden_states)

        def receive_remote_ffn(self, transfer, **_kwargs):
            return transfer.ref_tensor

        def _send_and_receive(self, hidden_states, **kwargs):
            return self.receive_remote_ffn(
                self.dispatch_remote_ffn(hidden_states, **kwargs),
            )

    deepseek_v2 = ModuleType("afd_plugin.model_executor.models.deepseek_v2")
    deepseek_v2.RemoteFFNProxy = RemoteFFNProxy
    deepseek_v2.AFDRemoteFFNTransfer = Transfer
    model_utils = ModuleType("afd_plugin.model_executor.models.model_utils")
    model_utils.is_dspark_config = lambda _config: False

    dspark = ModuleType("afd_plugin.model_executor.models.npu.deepseek_v41_dspark")

    class DeepseekV41DSparkTargetState:
        @classmethod
        def from_config(cls, _config):
            return SimpleNamespace(enabled=False, aux_layer_ids=())

    dspark.DeepseekV41DSparkTargetState = DeepseekV41DSparkTargetState
    afd_npu = ModuleType("afd_plugin.model_executor.models.npu")
    afd_npu.__path__ = []

    vllm_ascend = ModuleType("vllm_ascend")
    vllm_ascend.__path__ = []
    ascend_models = ModuleType("vllm_ascend.models")
    ascend_models.__path__ = []
    model_package = ModuleType("vllm_ascend.models.deepseek_v41")
    model_package.__path__ = []
    afd_executor = ModuleType("afd_plugin.model_executor")
    afd_executor.__path__ = []
    afd_models = ModuleType("afd_plugin.model_executor.models")
    afd_models.__path__ = []

    replacements = {
        "vllm": vllm,
        "vllm.config": config,
        "vllm.forward_context": forward_context,
        "vllm_ascend": vllm_ascend,
        "vllm_ascend.models": ascend_models,
        "vllm_ascend.models.deepseek_v41": model_package,
        "vllm_ascend.models.deepseek_v41.model": native,
        "vllm_ascend.models.deepseek_v41.vl_model": native_vl,
        "vllm_ascend.utils": native_utils,
        "afd_plugin.model_executor": afd_executor,
        "afd_plugin.model_executor.models": afd_models,
        "afd_plugin.model_executor.models.deepseek_v2": deepseek_v2,
        "afd_plugin.model_executor.models.model_utils": model_utils,
        "afd_plugin.model_executor.models.npu": afd_npu,
        "afd_plugin.model_executor.models.npu.deepseek_v41_dspark": dspark,
    }
    sys.modules.update(replacements)
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            Path(__file__).parents[4]
            / "afd_plugin"
            / "model_executor"
            / "models"
            / "deepseek_v41.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("unable to locate DeepSeek-V4.1 adapter")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


adapter = _load_adapter_with_native_stubs()


class _Connector:
    def __init__(self):
        self.idle_calls = 0
        self.waits: list[int] = []
        self.reset_calls = 0

    def require_attention_pipeline_idle(self):
        self.idle_calls += 1

    def wait_for_attention_stage_receive(self, *, stage_idx, tensor):
        del tensor
        self.waits.append(stage_idx)

    def reset_attention_pipeline_state(self):
        self.reset_calls += 1


def _model_for_layer_major(monkeypatch, layers, connector):
    model = object.__new__(adapter.AFDDeepseekV41Model)
    nn.Module.__init__(model)
    model.afd_role = "attention"
    model.hc_mult = 2
    model.start_layer = 0
    model.end_layer = len(layers)
    model.layers = nn.ModuleList(layers)
    model._shared_attention_states = {}
    model.shared_attention_state = None
    model._dspark_target_state = None
    model._dspark_hidden_buffer = None
    model._mtp_hidden_buffer = None
    model.embed_input_ids = lambda input_ids: input_ids.to(torch.float32)
    model.norm = nn.Identity()
    monkeypatch.setattr(
        adapter.native,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    return model


def _ubatch_item(stage_idx, connector):
    forward_context = SimpleNamespace(
        ubatch_idx=stage_idx,
        input_ids=None,
        additional_kwargs={"afd_metadata": SimpleNamespace(connector=connector)},
    )
    return SimpleNamespace(
        context=SimpleNamespace(forward_context=forward_context),
        input_ids=torch.tensor([[stage_idx + 1.0, stage_idx + 2.0]]),
        positions=torch.tensor([0]),
        inputs_embeds=None,
        intermediate_tensors=None,
    )


def test_v41_u1_passes_last_layer_pre_mix_to_collapse(monkeypatch):
    """The ordinary U1 path must collapse with the final layer's state."""

    model = object.__new__(adapter.AFDDeepseekV41Model)
    nn.Module.__init__(model)
    model.afd_role = "attention"
    model.hc_mult = 2
    model.start_layer = 0
    model.end_layer = 1
    model.embed_input_ids = lambda input_ids: input_ids.to(torch.float32)
    model.norm = nn.Identity()
    model._shared_attention_states = {}
    model.shared_attention_state = None
    model._dspark_target_state = None
    model._dspark_hidden_buffer = None
    model._mtp_hidden_buffer = None
    final_pre_mix = torch.tensor([[3.0, 4.0]])

    class Layer:
        def __call__(self, _positions, hidden, pre_mix, _scale, *, input_ids):
            del pre_mix, input_ids
            return hidden, final_pre_mix

    model.layers = [Layer()]
    seen: list[torch.Tensor] = []
    monkeypatch.setattr(
        adapter.native.DeepseekV41DecoderLayer,
        "hc_collapse",
        staticmethod(lambda hidden, pre_mix: seen.append(pre_mix) or hidden),
    )
    monkeypatch.setattr(
        adapter.native,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    model.forward(
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([0]),
        None,
    )

    assert len(seen) == 1
    assert torch.equal(seen[0], final_pre_mix)


def test_v41_u2_drains_last_layer_with_its_pre_mix(monkeypatch):
    """U2's final receive must update the state used by the final collapse."""

    connector = _Connector()
    final_pre_mixes = [torch.tensor([[7.0, 8.0]]), torch.tensor([[9.0, 10.0]])]
    seen: list[torch.Tensor] = []

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward_attention_to_remote_ffn(
            self, _positions, hidden, _pre_mix, _scale, *, input_ids
        ):
            del input_ids
            return hidden, (hidden, None, None, torch.tensor([[1.0, 1.0]]))

        def complete_remote_ffn(self, hidden, continuation):
            del continuation
            result = final_pre_mixes[self.calls]
            self.calls += 1
            return hidden, result

    layer = Layer()
    model = _model_for_layer_major(monkeypatch, [layer], connector)
    monkeypatch.setattr(
        adapter.native.DeepseekV41DecoderLayer,
        "hc_collapse",
        staticmethod(lambda hidden, pre_mix: seen.append(pre_mix) or hidden),
    )

    model.forward_ubatches_layer_major(
        [_ubatch_item(0, connector), _ubatch_item(1, connector)],
    )

    assert len(seen) == len(final_pre_mixes)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(seen, final_pre_mixes, strict=True)
    )
    assert connector.waits == [0, 1]


def test_v41_u2_reuses_connector_only_after_a_complete_drain(monkeypatch):
    connector = _Connector()

    class Layer(nn.Module):
        def forward_attention_to_remote_ffn(
            self, _positions, hidden, _pre_mix, _scale, *, input_ids
        ):
            del input_ids
            return hidden, (hidden, None, None, torch.ones((1, 2)))

        def complete_remote_ffn(self, hidden, continuation):
            del continuation
            return hidden, torch.ones((1, 2))

    layer = Layer()
    model = _model_for_layer_major(monkeypatch, [layer], connector)
    batches = [_ubatch_item(0, connector), _ubatch_item(1, connector)]
    model.forward_ubatches_layer_major(batches)
    model.forward_ubatches_layer_major(batches)

    assert connector.idle_calls == 2
    assert connector.reset_calls == 0
    assert connector.waits == [0, 1, 0, 1]


def test_v41_u2_resets_pipeline_after_layer_failure(monkeypatch):
    connector = _Connector()

    class FailingLayer(nn.Module):
        def forward_attention_to_remote_ffn(
            self, _positions, _hidden, _pre_mix, _scale, *, input_ids
        ):
            del input_ids
            raise ValueError("synthetic layer failure")

    model = _model_for_layer_major(monkeypatch, [FailingLayer()], connector)
    with pytest.raises(ValueError, match="synthetic layer failure"):
        model.forward_ubatches_layer_major(
            [_ubatch_item(0, connector), _ubatch_item(1, connector)],
        )

    assert connector.idle_calls == 1
    assert connector.reset_calls == 1


def test_v41_u2_rejects_mixed_stage_connectors(monkeypatch):
    first = _Connector()
    second = _Connector()
    model = _model_for_layer_major(monkeypatch, [], first)
    with pytest.raises(RuntimeError, match="must share one connector"):
        model.forward_ubatches_layer_major(
            [_ubatch_item(0, first), _ubatch_item(1, second)],
        )


def test_v41_u2_switches_each_stage_to_its_own_dsa_state():
    """Layer-major stage changes must not reuse one mutable DSA buffer."""

    model = object.__new__(adapter.AFDDeepseekV41Model)
    nn.Module.__init__(model)
    state0 = SimpleNamespace(stage=0)
    state1 = SimpleNamespace(stage=1)
    layer = object.__new__(adapter.AFDDeepseekV41DecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = nn.Module()
    model.layers = nn.ModuleList([layer])
    model._shared_attention_states = {0: state0, 1: state1}
    model.shared_attention_state = state0

    model._activate_attention_shared_state(0)
    assert layer.self_attn.shared_state is state0
    model._activate_attention_shared_state(1)
    assert layer.self_attn.shared_state is state1
    assert state0 is not state1


def test_v41_u2_activates_dsa_state_before_each_stage_layer(monkeypatch):
    """Every layer-major stage call must use that stage's DSA buffers."""

    connector = _Connector()
    state0 = SimpleNamespace(stage=0)
    state1 = SimpleNamespace(stage=1)
    observed: list[tuple[int, object]] = []

    class Layer(adapter.AFDDeepseekV41DecoderLayer):
        def __init__(self):
            nn.Module.__init__(self)
            self.layer_idx = 0
            self.self_attn = nn.Module()

        def forward_attention_to_remote_ffn(
            self, _positions, hidden, _pre_mix, _scale, *, input_ids
        ):
            observed.append((int(input_ids[0, 0]), self.self_attn.shared_state))
            return hidden, (hidden, None, None, torch.ones((1, 2)))

        def complete_remote_ffn(self, hidden, continuation):
            del continuation
            return hidden, torch.ones((1, 2))

    layer = Layer()
    model = _model_for_layer_major(monkeypatch, [layer], connector)
    model._shared_attention_states = {0: state0, 1: state1}
    model.shared_attention_state = state0

    model.forward_ubatches_layer_major(
        [_ubatch_item(0, connector), _ubatch_item(1, connector)],
    )

    assert observed == [(1, state0), (2, state1)]
    assert layer.self_attn.shared_state is state0
