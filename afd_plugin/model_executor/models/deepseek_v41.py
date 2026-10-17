# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V4.1 text AFD adapter.

The first V4.1 adapter stage deliberately targets the text-only P2P/U1 path.
It reuses the existing AFD transfer contract and native V4.1 attention/MoE
operators while keeping model ownership role selective.  V4.1 multimodal,
Engram, Window and speculative paths are added only after their metadata and
operator contracts are validated against the matching vLLM-Ascend release.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm_ascend.models.deepseek_v41 import model as native
from vllm_ascend.models.deepseek_v41 import vl_model as native_vl

from afd_plugin.config import parse_afd_config
from afd_plugin.model_executor.models.deepseek_v2 import (
    AFDRemoteFFNTransfer,
    RemoteFFNProxy,
)


_ATTENTION_ROLE = "attention"
_FFN_ROLE = "ffn"


def _checkpoint_weight_roles(name: str) -> frozenset[str]:
    """Return the V4.1 AFD owner for one native checkpoint key."""

    normalized = name.removeprefix("model.")
    if normalized.startswith("mtp."):
        return frozenset()
    parts = normalized.split(".")
    if len(parts) >= 3 and parts[0] == "layers" and parts[1].isdigit():
        return frozenset({_FFN_ROLE}) if parts[2] == "mlp" else frozenset({_ATTENTION_ROLE})
    return frozenset({_ATTENTION_ROLE})


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
    num_hidden_layers: int,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Filter a checkpoint iterator once, preserving native key names."""

    for name, loaded_weight in weights:
        normalized = name.removeprefix("model.")
        parts = normalized.split(".")
        if (
            len(parts) >= 2
            and parts[0] == "layers"
            and parts[1].isdigit()
            and int(parts[1]) >= num_hidden_layers
        ):
            continue
        if role in _checkpoint_weight_roles(name):
            yield name, loaded_weight


class AFDDeepseekV41RemoteMoEProxy(RemoteFFNProxy):
    """Parameter-free V4.1 MoE stage executed by the remote FFN role."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._send_and_receive(hidden_states)

    def dispatch_remote_ffn(
        self,
        hidden_states: torch.Tensor,
        **send_kwargs: torch.Tensor,
    ) -> AFDRemoteFFNTransfer:
        # V4.1 hash routing is owned by the native FFN MoE in the first stage.
        # The raw IDs still cross the A2F boundary once, exactly as V4 does.
        if self.phase == "decoder" and self.layer_idx == 0:
            input_ids = send_kwargs.get("input_ids")
            if input_ids is None:
                input_ids = getattr(get_forward_context(), "input_ids", None)
            if input_ids is None:
                raise RuntimeError("DeepSeek-V4.1 layer 0 requires input_ids")
            send_kwargs["input_ids"] = input_ids
        return super().dispatch_remote_ffn(hidden_states, **send_kwargs)


class AFDDeepseekV41DecoderLayer(native.DeepseekV41DecoderLayer):
    """Role-selective V4.1 decoder layer for text P2P/U1 execution."""

    # Patch reason: native V4.1 constructs attention and MoE on every worker.
    # Patch functionality: construct only the active AFD role and transfer the
    # post-attention MoE input to the FFN worker.
    # Signature: matches vLLM-Ascend V4.1 DecoderLayer; no added parameters.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v41/model.py
    # Commit: f769db916b53446de5779bced0b7d3d38e8a7540
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: Any | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        is_draft_layer: bool = False,
    ) -> None:
        # ### PATCH START: role-selective V4.1 construction.
        nn.Module.__init__(self)
        afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = afd_config.role
        if config is None:
            config = native.normalize_deepseek_v41_config(
                vllm_config.model_config.hf_config,
            )
        self.vllm_config = vllm_config
        self.config = config
        self.layer_idx = int(prefix.split(".")[-1])
        self.hidden_size = config.hidden_size
        self.norm_eps = config.rms_norm_eps
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.use_sequence_parallel = vllm_config.parallel_config.use_sequence_parallel_moe
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.compute_gate_on_attention = bool(afd_config.compute_gate_on_attention)
        self.is_moe_layer = True
        if self.compute_gate_on_attention:
            raise RuntimeError(
                "DeepSeek-V4.1 P2P/U1 requires FFN-side native routing; "
                "Attention-side gate is reserved for the later Window adapter"
            )

        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        if self.afd_role == _ATTENTION_ROLE:
            max_position_embeddings = config.rope_parameters[
                "original_max_position_embeddings"
            ]
            self.self_attn = native.DeepseekV41Attention(
                vllm_config=vllm_config,
                config=config,
                max_position_embeddings=max_position_embeddings,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
                reduce_results=not self.use_sequence_parallel,
                need_gather_q_kv=False,
            )
            self.mlp = AFDDeepseekV41RemoteMoEProxy(
                layer_idx=self.layer_idx,
                phase="mtp" if is_draft_layer else "decoder",
            )
            self.input_layernorm = native.RMSNorm(
                config.hidden_size,
                eps=self.norm_eps,
            )
            self.post_attention_layernorm = native.RMSNorm(
                config.hidden_size,
                eps=self.norm_eps,
            )
        else:
            self.self_attn = native.PPMissingLayer()
            self.mlp = native.DeepseekV41MoE(
                config=config,
                parallel_config=vllm_config.parallel_config,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
                is_draft_layer=is_draft_layer,
            )

        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        # ### PATCH END: role-selective V4.1 construction.

    def hc_pre(self, x, hc_fn, hc_scale, hc_base, pre_mix=None):
        return torch.ops._C_ascend.npu_hc_pre_v3(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            pre_mix,
            hc_mult=self.hc_mult,
            hc_sinkhorn_iters=self.hc_sinkhorn_iters,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
        )

    def hc_post(self, x, residual, post, comb):
        return torch.ops._C_ascend.npu_hc_post(
            x.unsqueeze(0),
            residual.unsqueeze(0),
            post.unsqueeze(0),
            comb.unsqueeze(0),
        ).squeeze(0)

    def rms_norm_cast(self, hidden_states: torch.Tensor):
        normalized = self.post_attention_layernorm(hidden_states)
        return normalized, normalized.float()

    # Patch reason: native forward invokes the local V4.1 MoE unconditionally.
    # Patch functionality: Attention sends normalized MoE input to the FFN
    # worker and completes mHC after receiving the remote result.
    # Signature: matches vLLM-Ascend V4.1 DecoderLayer.forward exactly.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v41/model.py
    # Commit: f769db916b53446de5779bced0b7d3d38e8a7540
    def forward(
        self,
        positions,
        hidden_states,
        pre_mix,
        llama_4_scaling=None,
        input_ids=None,
    ):
        if self.afd_role != _ATTENTION_ROLE:
            raise RuntimeError("DeepSeek-V4.1 FFN model execution is connector-driven")
        residual = hidden_states
        x, attn_post, attn_comb, attn_pre = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            pre_mix,
        )
        x = self.input_layernorm(x)
        if self.use_sequence_parallel:
            x = native.sp_all_gather(x)[: positions.shape[0]]
        x = self.self_attn(positions, x, llama_4_scaling)
        if self.use_sequence_parallel:
            x = native.sp_reduce_scatter(x)
        hidden_states = self.hc_post(x, residual, attn_post, attn_comb)

        residual = hidden_states
        x, ffn_post, ffn_comb, ffn_pre = self.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            attn_pre,
        )
        x, _ = self.rms_norm_cast(x)
        transfer = self.mlp.dispatch_remote_ffn(
            x,
            input_ids=input_ids if self.layer_idx == 0 else None,
        )
        x = self.mlp.receive_remote_ffn(transfer, layer_idx=self.layer_idx)
        return self.hc_post(x, residual, ffn_post, ffn_comb), ffn_pre

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.afd_role != _FFN_ROLE:
            raise RuntimeError("DeepSeek-V4.1 Attention role has no local MoE")
        return self.mlp(
            hidden_states,
            input_ids=input_ids,
            **{
                key: value
                for key, value in kwargs.items()
                if key in {"hidden_states_fp32", "already_sequence_parallel"}
            },
        )


@native.support_torch_compile(
    dynamic_arg_dims={"input_ids": 0, "positions": 0, "intermediate_tensors": 0},
)
class AFDDeepseekV41Model(native.DeepseekV41Model):
    """Role-aware V4.1 backbone for the text P2P/U1 baseline."""

    # Patch reason: native model allocates full attention and MoE layers on all
    # workers and initializes Engram tables before AFD role selection.
    # Patch functionality: keep the native V4.1 cache/state setup needed by
    # attention, while constructing role-owned decoder layers only.
    # Signature: matches vLLM-Ascend V4.1 model constructor.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v41/model.py
    # Commit: f769db916b53446de5779bced0b7d3d38e8a7540
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = self.afd_config.role
        if getattr(vllm_config, "engram_config", None) is not None:
            raise RuntimeError(
                "DeepSeek-V4.1 AFD text baseline does not yet transfer Engram state"
            )
        config = native.normalize_deepseek_v41_config(
            vllm_config.model_config.hf_config,
        )
        self.config = config
        self.device = native.current_platform.device_type
        self.vocab_size = config.vocab_size
        self.use_sequence_parallel = vllm_config.parallel_config.use_sequence_parallel_moe
        self.hc_mult = config.hc_mult
        if hasattr(config, "index_topk"):
            self.topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.topk_indices_buffer = None

        if self.afd_role == _ATTENTION_ROLE and native.get_pp_group().is_first_rank:
            self.embed_tokens = native.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = native.PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = native.make_layers(
            config.num_hidden_layers,
            lambda layer_prefix: AFDDeepseekV41DecoderLayer(
                vllm_config,
                layer_prefix,
                topk_indices_buffer=self.topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )
        if self.afd_role == _ATTENTION_ROLE and native.get_pp_group().is_last_rank:
            self.norm = native.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = native.PPMissingLayer()

        self._mtp_hidden_buffer = None
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        self.has_engram = False
        self.engram_hash = None
        self.requires_uncompiled_fallback = False
        self.make_empty_intermediate_tensors = self._make_empty_intermediate_tensors

        if self.afd_role == _ATTENTION_ROLE and self.topk_indices_buffer is not None:
            topology = native.build_layer_plan(config)
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            candidate = torch.full(
                (max_tokens, 1, topology.candidate_topk_blocks),
                -1,
                dtype=torch.int32,
                device=self.topk_indices_buffer.device,
            )
            self.shared_attention_state = native.DeepseekV41SharedAttentionState(
                self.topk_indices_buffer,
                candidate,
                torch.zeros((max_tokens, 1), dtype=torch.int32, device=self.topk_indices_buffer.device),
                torch.zeros((max_tokens, 1), dtype=torch.int32, device=self.topk_indices_buffer.device),
            )
            for layer in self.layers:
                if isinstance(layer, AFDDeepseekV41DecoderLayer):
                    layer.self_attn.shared_state = self.shared_attention_state
        else:
            self.shared_attention_state = None

    def _make_empty_intermediate_tensors(self, batch_size, dtype, device):
        hidden_states = torch.empty(
            (batch_size, self.hc_mult, self.config.hidden_size),
            dtype=dtype,
            device=device,
        )
        return native.IntermediateTensors(
            {"hidden_states": hidden_states},
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.afd_role != _ATTENTION_ROLE:
            raise RuntimeError("DeepSeek-V4.1 FFN role does not own token embeddings")
        return self.embed_tokens(input_ids)

    # Patch reason: native V4.1 forward always owns and executes every local MoE.
    # Patch functionality: run only Attention layers and expose the same
    # intermediate tensor contract used by the existing AFD runner.
    # Signature: matches the native V4.1 model forward entry point for the
    # arguments consumed by the vLLM runner.
    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds=None,
        **kwargs: Any,
    ):
        del kwargs
        if self.afd_role != _ATTENTION_ROLE:
            raise RuntimeError("DeepSeek-V4.1 FFN model execution is connector-driven")
        if native.get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        else:
            if intermediate_tensors is None:
                raise RuntimeError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]
        pre_mix = hidden_states.new_zeros(hidden_states.shape[0], self.hc_mult, dtype=torch.float32)
        pre_mix[:, 0] = 1.0
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, pre_mix = layer(
                positions,
                hidden_states,
                pre_mix,
                None,
                input_ids=input_ids,
            )
        if not native.get_pp_group().is_last_rank:
            return native.IntermediateTensors({"hidden_states": hidden_states})
        hidden_states = native.DeepseekV41DecoderLayer.hc_collapse(hidden_states, pre_mix)
        return self.norm(hidden_states)

    def compute_ffn_output(self, hidden_states: torch.Tensor, layer_idx: int, **kwargs: Any):
        return self.layers[layer_idx].compute_ffn_output(hidden_states, **kwargs)

    def get_mtp_target_hidden_states(self):
        return None

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.aux_hidden_state_layers = layers


class AFDDeepseekV41ForCausalLM(native.AscendDeepseekV41LLMForCausalLM):
    """V4.1 language model wrapper with strict AFD role ownership."""

    model_cls = AFDDeepseekV41Model
    requires_raw_input_tokens = True

    # Patch reason: native V4.1 LM construction allocates the output head and
    # MoE metadata for both AFD roles.
    # Patch functionality: retain the head only on Attention and use native
    # V4.1 weight loading after one-shot role filtering.
    # Signature: matches the native V4.1 LM constructor.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v41/model.py
    # Commit: f769db916b53446de5779bced0b7d3d38e8a7540
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = self.afd_config.role
        self.vllm_config = vllm_config
        self.config = native.normalize_deepseek_v41_config(vllm_config.model_config.hf_config)
        self.quant_config = vllm_config.quant_config
        self.model = self.model_cls(vllm_config=vllm_config, prefix=native.maybe_prefix(prefix, "model"))
        if self.afd_role == _ATTENTION_ROLE and native.get_pp_group().is_last_rank:
            self.lm_head = native.ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=self.quant_config,
                prefix=native.maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = native.PPMissingLayer()
        self.logits_processor = native.LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        self.requires_uncompiled_fallback = False
        self.num_moe_layers = self.config.num_hidden_layers
        self.set_moe_parameters()
        self.moe_comm_methods = getattr(native.AscendDeepseekV41LLMForCausalLM, "moe_comm_methods", {})

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs: Any,
    ):
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def prepare_engram_inputs(self, *args, **kwargs):
        raise RuntimeError("DeepSeek-V4.1 AFD Engram transport is not implemented")

    def prepare_engram_graph_inputs(self, *args, **kwargs):
        return {"engram_lookups": {}, "engram_mask": torch.empty(0)}

    def prime_engram_v2_graph_inputs(self, *args, **kwargs):
        return self.prepare_engram_graph_inputs(*args, **kwargs)

    def retire_engram_lookups(self, *, reset_events=False):
        del reset_events

    def get_model_state_cls(self):
        return None

    @property
    def supports_engram_graph_producer(self) -> bool:
        return False

    @property
    def token_lookback_depth(self) -> int:
        return 0

    def set_moe_parameters(self):
        self.expert_weights = []
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        if self.afd_role == _FFN_ROLE:
            for layer in self.model.layers:
                if isinstance(getattr(layer, "mlp", None), native.DeepseekV41MoE):
                    example_moe = layer.mlp
                    self.moe_mlp_layers.append(layer.mlp)
                    self.moe_layers.append(layer.mlp.experts)
        native.DeepseekV41MixtureOfExperts.extract_moe_parameters(self, example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor):
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_ffn_output(self, hidden_states: torch.Tensor, layer_idx: int, **kwargs: Any):
        return self.model.compute_ffn_output(hidden_states, layer_idx, **kwargs)

    def get_mtp_target_hidden_states(self):
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        role_weights = _iter_role_weights(
            weights,
            role=self.afd_role,
            num_hidden_layers=int(self.config.num_hidden_layers),
        )
        return super().load_weights(role_weights)


class AFDDeepseekV41VisionForCausalLM(
    native_vl.AscendDeepseekV41ForCausalLM,
):
    """V4.1 vision wrapper using the role-aware language model."""

    language_model_cls = AFDDeepseekV41ForCausalLM


__all__ = [
    "AFDDeepseekV41ForCausalLM",
    "AFDDeepseekV41VisionForCausalLM",
]
