# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V4.1 AFD role wrapper for the Window U1 eager seam.

The native Ascend V4.1 decoder remains the source of truth for DSA, Engram and
mHC operators.  This adapter changes only the FFN boundary: Attention computes
the native gate, Window exchanges a two-dimensional activation, and the
Attention side applies the native FFN ``hc_post`` after F2A returns.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator
from typing import Any
from weakref import proxy

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.models.utils import PPMissingLayer

from afd_plugin.config import parse_afd_config
from afd_plugin.connectors.metadata import (
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.distributed import (
    build_window_expert_layout,
    resolve_role_rank,
)
from afd_plugin.model_executor.models import get_afd_metadata_from_forward_context
from afd_plugin.model_executor.models.npu.deepseek_v41_window_contract import (
    V41MHCContinuation,
    V41MHCContinuationKey,
    V41MHCContinuationRegistry,
    validate_window_activation_shape,
)

try:
    from vllm_ascend.models.deepseek_v41 import model as native
except ImportError as exc:  # pragma: no cover - NPU-only import
    raise ImportError(
        "DSV4.1 AFD support requires the pinned vLLM-Ascend V4.1 model"
    ) from exc

_ATTENTION_ROLE = "attention"
_FFN_ROLE = "ffn"


def _window_expert_layout(vllm_config: VllmConfig, role_rank: int):
    """Resolve the deterministic shared-first layout used by Window U1."""
    afd_config = parse_afd_config(vllm_config, validate=False)
    if vllm_config.parallel_config.enable_eplb:
        raise ValueError("DSV4.1 Window U1 does not support EPLB")
    if bool(
        (getattr(vllm_config, "additional_config", {}) or {}).get(
            "mix_placement",
            False,
        ),
    ):
        raise ValueError(
            "DSV4.1 Window U1 requires mix_placement=false for a dedicated "
            "shared-expert rank",
        )
    hf_config = vllm_config.model_config.hf_config
    if int(hf_config.n_shared_experts) != 1:
        raise ValueError(
            "DSV4.1 Window U1 requires exactly one shared expert rank",
        )
    return build_window_expert_layout(
        routed_expert_num=int(hf_config.n_routed_experts),
        ffn_size=afd_config.num_ffn_ranks,
        ffn_rank=role_rank,
    )


class AFDDeepseekV41SharedMoE(nn.Module):
    """Shared-only FFN module placed on Window FFN rank zero."""

    def __init__(
        self,
        *,
        config: Any,
        parallel_config: Any,
        quant_config: Any,
        prefix: str,
    ) -> None:
        super().__init__()
        self.routed_scaling_factor = float(
            getattr(config, "routed_scaling_factor", 1.0),
        )
        self.swiglu_limit = getattr(config, "swiglu_limit", None)
        self.experts = None
        self.shared_experts = native.DeepseekV41MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            hidden_act=config.hidden_act,
            swiglu_limit=self.swiglu_limit,
            quant_config=quant_config,
            is_sequence_parallel=parallel_config.use_sequence_parallel_moe,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.shared_experts(hidden_states)


class AFDDeepseekV41RoutedMoE(native.DeepseekV41MoE):
    """Routed-only V4.1 MoE with a Window-local expert map."""

    def __init__(
        self,
        *,
        config: Any,
        parallel_config: Any,
        quant_config: Any,
        prefix: str,
        expert_layout: Any,
        is_draft_layer: bool = False,
    ) -> None:
        if expert_layout.is_shared:
            raise ValueError("routed V4.1 MoE requires a routed expert layout")
        routed_config = copy.copy(config)
        # Native V4.1 treats None as the no-shared-expert configuration and
        # consequently constructs the same routed FusedMoE ABI used by U1.
        object.__setattr__(routed_config, "n_shared_experts", None)
        super().__init__(
            config=routed_config,
            parallel_config=parallel_config,
            quant_config=quant_config,
            prefix=prefix,
            is_draft_layer=is_draft_layer,
        )
        self.n_shared_experts = 0
        self.shared_experts = None
        self.expert_layout = expert_layout
        # Window owns cross-rank dispatch, so allocate only this rank's
        # routed expert slots and keep the checkpoint's global IDs in the
        # explicit map below.
        self.experts = FusedMoEFactory(
            shared_experts=None,
            gate=None,
            num_experts=expert_layout.local_expert_count,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            tp_size=1,
            dp_size=1,
            use_grouped_topk=True,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "softmax"),
            routed_scaling_factor=self.routed_scaling_factor,
            swiglu_limit=self.swiglu_limit,
            e_score_correction_bias=None,
            enable_eplb=False,
            num_redundant_experts=0,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=0,
        )
        expert_map = torch.full(
            (self.n_routed_experts,),
            -1,
            dtype=torch.int32,
        )
        start = expert_layout.local_expert_start
        count = expert_layout.local_expert_count
        expert_map[start : start + count] = torch.arange(
            count,
            dtype=torch.int32,
        )
        global_expert_num = int(self.n_routed_experts)
        self.experts.moe_config.num_experts = global_expert_num
        self.experts.expert_map_manager.global_num_experts = global_expert_num
        self.experts.routed_experts.moe_config.num_experts = global_expert_num
        self.experts.routed_experts.global_num_experts = global_expert_num
        self.experts.expert_map_manager._expert_map = expert_map
        self.experts.expert_map_manager._local_num_experts = count
        self.experts.routed_experts._expert_map = expert_map
        self.experts.routed_experts.local_num_experts = count
        self.experts.routed_experts.update_expert_map_info()


def _weight_layer_stage(name: str) -> str | None:
    parts = name.split(".")
    for index, part in enumerate(parts[:-2]):
        if part == "layers":
            try:
                int(parts[index + 1])
            except (IndexError, ValueError):
                continue
            return parts[index + 2]
    return None


def _role_owns_weight(name: str, role: str) -> bool:
    stage = _weight_layer_stage(name)
    if stage is None:
        return True
    if stage in {"self_attn", "attn"}:
        return role == _ATTENTION_ROLE
    if stage in {"mlp", "ffn"}:
        if role == _FFN_ROLE:
            return True
        # U1 computes top-k on Attention, so its retained native MoE shell
        # needs only the replicated gate (the expert matrices stay on FFN).
        return ".gate." in name
    # Norms and mHC coefficients are retained on both roles until the U1
    # numerical seam is proven.  This avoids silently dropping a native load
    # path while keeping Window payloads free of these tensors.
    return True


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        if _role_owns_weight(name, role):
            yield name, tensor


class AFDDeepseekV41RemoteMoE(nn.Module):
    """Attention-side V4.1 gate plus a synchronous U1 Window transfer."""

    def __init__(self, *, layer: AFDDeepseekV41DecoderLayer) -> None:
        super().__init__()
        # A strong Module reference would create layer -> proxy -> layer and
        # recurse through state_dict()/to(device). Keep the owner non-owning.
        object.__setattr__(self, "layer", proxy(layer))

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        hidden_states_fp32: torch.Tensor | None,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        del already_sequence_parallel
        afd_metadata = get_afd_metadata_from_forward_context()
        if afd_metadata is None:
            raise RuntimeError("V4.1 Window FFN requires AFD forward metadata")
        connector = afd_metadata.connector
        router_input = (
            hidden_states.float() if hidden_states_fp32 is None else hidden_states_fp32
        )
        router_logits = torch.nn.functional.linear(
            router_input,
            self.layer._native_mlp.gate.weight_fp32,
        )
        topk_weights, topk_ids = connector.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=int(self.layer.config.num_experts_per_tok),
            use_grouped_topk=True,
            renormalize=bool(self.layer.config.norm_topk_prob),
            scoring_func=getattr(self.layer.config, "scoring_func", "softmax"),
            num_expert_group=int(getattr(self.layer.config, "n_group", 1)),
            topk_group=int(getattr(self.layer.config, "topk_group", 1)),
            routed_scaling_factor=(
                float(getattr(self.layer.config, "routed_scaling_factor", 1.0))
                if self.layer.mix_placement
                else 1.0
            ),
            e_score_correction_bias=self.layer._native_mlp.gate.e_score_correction_bias,
            tid2eid=getattr(self.layer._native_mlp.gate, "tid2eid", None),
            mix_placement=self.layer.mix_placement,
            num_logical_experts=int(self.layer.config.n_routed_experts),
            num_shared_experts=int(self.layer.config.n_shared_experts),
            input_ids=input_ids,
            num_experts=(
                int(self.layer.config.n_routed_experts)
                + int(self.layer.config.n_shared_experts)
                if self.layer.mix_placement
                else int(self.layer.config.n_routed_experts)
            ),
        )
        validate_window_activation_shape(
            tuple(hidden_states.shape),
            token_count=int(hidden_states.shape[0]),
            hidden_size=int(hidden_states.shape[-1]),
        )
        stage_idx = int(getattr(get_forward_context(), "ubatch_idx", 0))
        generation_id = afd_metadata.transaction_id or "eager"
        key = V41MHCContinuationKey(
            generation_id=generation_id,
            ubatch_idx=stage_idx,
            layer_idx=self.layer.layer_idx,
            stage_idx=stage_idx,
            token_order=tuple(range(int(hidden_states.shape[0]))),
        )
        self.layer.continuations.put(
            V41MHCContinuation(
                key=key,
                pre_mix_in=self.layer._pre_mix_in,
                attn_pre=self.layer._attn_pre,
                residual=self.layer._ffn_residual,
                post_mix=self.layer._ffn_post,
                res_mix=self.layer._ffn_comb,
                valid_tokens=int(hidden_states.shape[0]),
            )
        )
        self.layer._continuation_key = key
        context = AFDTransferContext(
            metadata=AFDTransferMetadata.create_attention_metadata(
                layer_idx=self.layer.layer_idx,
                stage_idx=stage_idx,
                seq_len=int(hidden_states.shape[0]),
            )
        )
        try:
            connector.send_attn_output(
                hidden_states,
                context,
                expert_ids=topk_ids,
                expert_scales=topk_weights,
                input_ids=input_ids,
            )
            return connector.recv_ffn_output(
                ref_tensor=hidden_states,
                ubatch_idx=stage_idx,
                layer_idx=self.layer.layer_idx,
            )
        except BaseException:
            self.layer.continuations.discard(key)
            raise


class AFDDeepseekV41DecoderLayer(native.DeepseekV41DecoderLayer):
    """Native V4.1 layer with a role-local U1 FFN seam."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        vllm_config = args[0] if args else kwargs["vllm_config"]
        prefix = args[1] if len(args) > 1 else kwargs.get("prefix", "")
        config = args[2] if len(args) > 2 else kwargs.get("config")
        if config is None:
            config = native.normalize_deepseek_v41_config(
                vllm_config.model_config.hf_config,
            )
        self.vllm_config = vllm_config
        self.config = config
        self.mix_placement = bool(
            (getattr(vllm_config, "additional_config", {}) or {}).get(
                "mix_placement",
                False,
            ),
        )
        # ### PATCH START: select one AFD role after native V4.1 construction.
        self.afd_role = parse_afd_config(vllm_config, validate=False).role
        self.continuations = V41MHCContinuationRegistry()
        if self.afd_role == _ATTENTION_ROLE:
            self._native_mlp = self.mlp
            self.mlp = AFDDeepseekV41RemoteMoE(layer=self)
        elif self.afd_role == _FFN_ROLE:
            self.self_attn = PPMissingLayer()
            afd_config = parse_afd_config(vllm_config, validate=False)
            self._native_mlp = self.mlp
            if afd_config.connector == "WindowAFDConnector":
                role_rank = resolve_role_rank(vllm_config, afd_config)
                expert_layout = _window_expert_layout(vllm_config, role_rank)
                if expert_layout.is_shared:
                    self.mlp = AFDDeepseekV41SharedMoE(
                        config=config,
                        parallel_config=vllm_config.parallel_config,
                        quant_config=vllm_config.quant_config,
                        prefix=f"{prefix}.mlp",
                    )
                else:
                    self.mlp = AFDDeepseekV41RoutedMoE(
                        config=config,
                        parallel_config=vllm_config.parallel_config,
                        quant_config=vllm_config.quant_config,
                        prefix=f"{prefix}.mlp",
                        expert_layout=expert_layout,
                    )
                # The native full MoE was only needed to let the pinned
                # decoder constructor allocate its common fields. Drop that
                # registered module so Window FFN ranks keep one role-local
                # weight tree.
                del self._native_mlp
        else:  # pragma: no cover - parse_afd_config validates roles
            raise ValueError(f"unsupported AFD role {self.afd_role!r}")
        # ### PATCH END

    # ### PATCH START: retain V4.1 delayed mHC state across the U1 FFN call.
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        pre_mix: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.afd_role == _FFN_ROLE:
            raise RuntimeError("V4.1 AFD FFN role is driven by compute_ffn_output")
        self._pre_mix_in = pre_mix
        residual = hidden_states
        x, attn_post, attn_comb, attn_pre = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            pre_mix,
        )
        x = self.input_layernorm(x)
        x = self.self_attn(positions, x, llama_4_scaling)
        hidden_states = self.hc_post(x, residual, attn_post, attn_comb)
        self._attn_pre = attn_pre
        self._ffn_residual = hidden_states
        x, ffn_post, ffn_comb, ffn_pre = self.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            attn_pre,
        )
        x, x_fp32 = self.rms_norm_cast(x)
        self._ffn_post = ffn_post
        self._ffn_comb = ffn_comb
        x = self.mlp(
            x,
            input_ids=input_ids,
            hidden_states_fp32=x_fp32,
            already_sequence_parallel=False,
        )
        continuation = self.continuations.pop(self._continuation_key)
        return self.hc_post(
            x,
            continuation.residual,
            continuation.post_mix,
            continuation.res_mix,
        ), ffn_pre

    # ### PATCH END

    def compute_ffn_output(
        self, hidden_states: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        if self.afd_role != _FFN_ROLE:
            raise RuntimeError("V4.1 compute_ffn_output is FFN-role only")
        afd_metadata = get_afd_metadata_from_forward_context()
        if afd_metadata is not None and getattr(
            afd_metadata.connector, "is_window_connector", False
        ):
            state = afd_metadata.connector.get_active_state(
                layer_idx=self.layer_idx,
                ubatch_idx=int(getattr(get_forward_context(), "ubatch_idx", 0)),
            )
            if state.group_list is None:
                raise RuntimeError("V4.1 Window FFN requires group_list from batching")
            # The generic Ascend helper owns the native expert ABI and scale
            # handling.  Window U1 supplies its compact batching state here;
            # a dedicated FP4 packer remains a later quantization milestone.
            from afd_plugin.model_executor.models.npu import (
                deepseek_v2_attention_gate,
            )

            return deepseek_v2_attention_gate.compute_attention_gate_moe_ffn(
                self,
                hidden_states=hidden_states,
                group_list=state.group_list,
                dynamic_scales=state.dynamic_scale,
                expand_x_shared=None,
                dynamic_scales_shared=None,
                topk_scales=None,
                group_list_type=int(state.group_list_type),
            )
        return self._native_mlp(hidden_states, **kwargs)


class AFDDeepseekV41Model(native.DeepseekV41Model):
    """Native V4.1 model selecting the role-aware decoder layer."""

    # ### PATCH START: use the AFD layer factory while preserving native model
    # construction, Engram preparation, and delayed HC collapse.
    decoder_layer_cls = AFDDeepseekV41DecoderLayer
    # ### PATCH END

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.layers[layer_idx].compute_ffn_output(hidden_states, **kwargs)


class AFDDeepseekV41ForCausalLM(native.AscendDeepseekV41LLMForCausalLM):
    """V4.1 AFD registration and role-filtered weight loading."""

    model_cls = AFDDeepseekV41Model
    afd_requires_input_ids = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        self.afd_role = parse_afd_config(vllm_config, validate=False).role
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.model.compute_ffn_output(hidden_states, layer_idx, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(_iter_role_weights(weights, role=self.afd_role))


__all__ = [
    "AFDDeepseekV41RoutedMoE",
    "AFDDeepseekV41SharedMoE",
    "AFDDeepseekV41DecoderLayer",
    "AFDDeepseekV41ForCausalLM",
    "AFDDeepseekV41Model",
]
