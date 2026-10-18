# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek V4.1 text AFD adapter.

The adapter keeps role ownership selective while reusing the native V4.1
attention, HC, MoE, DSpark and quantization contracts.  P2P/CAMP eager U1/U2
and Window packed-global-FFN U1/U2 are implemented behind the existing
connector contracts; Graph, Engram transport and generic MTP remain explicitly
guarded until their metadata and operator contracts are validated on the target
stack.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, override_forward_context
from vllm_ascend.models.deepseek_v41 import model as native
from vllm_ascend.models.deepseek_v41 import vl_model as native_vl

from afd_plugin.config import parse_afd_config
from afd_plugin.distributed.topology import (
    build_window_expert_layout,
    resolve_role_rank,
)
from afd_plugin.model_executor.models.deepseek_v2 import (
    AFDRemoteFFNTransfer,
    RemoteFFNProxy,
)
from afd_plugin.model_executor.models.model_utils import is_dspark_config
from afd_plugin.model_executor.models.npu.deepseek_v41_dspark import (
    DeepseekV41DSparkTargetState,
)

if TYPE_CHECKING:
    from afd_plugin.connectors.metadata import (
        AFDExpertRoutingSpec,
        AFDF2ATransferPayload,
    )


_ATTENTION_ROLE = "attention"
_FFN_ROLE = "ffn"


def _window_expert_layout(vllm_config: VllmConfig, role_rank: int):
    """Resolve V4.1's contiguous routed/shared Window rank ownership."""
    afd_config = parse_afd_config(vllm_config, validate=False)
    config = vllm_config.model_config.hf_config
    return build_window_expert_layout(
        routed_expert_num=int(config.n_routed_experts),
        ffn_size=int(afd_config.num_ffn_ranks),
        ffn_rank=role_rank,
    )


def _checkpoint_weight_roles(
    name: str,
    *,
    compute_gate_on_attention: bool = False,
) -> frozenset[str]:
    """Return the V4.1 AFD owner for one native checkpoint key."""

    normalized = name.removeprefix("model.")
    if normalized.startswith("mtp."):
        return frozenset()
    parts = normalized.split(".")
    if len(parts) >= 3 and parts[0] == "layers" and parts[1].isdigit():
        # GitCode checkpoints use ``ffn`` while vLLM-Ascend's native module
        # namespace uses ``mlp``; native load_weights remaps both spellings.
        if parts[2] not in {"mlp", "ffn"}:
            return frozenset({_ATTENTION_ROLE})
        if compute_gate_on_attention and len(parts) >= 4 and parts[3] == "gate":
            return frozenset({_ATTENTION_ROLE})
        return frozenset({_FFN_ROLE})
    return frozenset({_ATTENTION_ROLE})


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
    num_hidden_layers: int,
    window_ffn_kind: str | None = None,
    compute_gate_on_attention: bool = False,
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
        if role in _checkpoint_weight_roles(
            name,
            compute_gate_on_attention=compute_gate_on_attention,
        ):
            if role == _FFN_ROLE and window_ffn_kind is not None:
                is_shared = ".shared_experts." in f".{normalized}"
                if window_ffn_kind == "shared" and not is_shared:
                    continue
                if window_ffn_kind == "routed" and is_shared:
                    continue
            yield name, loaded_weight


class AFDDeepseekV41RemoteMoEProxy(RemoteFFNProxy):
    """Parameter-free V4.1 MoE stage executed by the remote FFN role."""

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
        self.top_k = int(config.num_experts_per_tok)
        self.use_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.compute_gate_on_attention = bool(afd_config.compute_gate_on_attention)
        self.is_moe_layer = True
        self.is_window_afd = afd_config.connector == "WindowAFDConnector"
        if self.compute_gate_on_attention and not self.is_window_afd:
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
            if self.compute_gate_on_attention:
                self.mlp.gate = native.ReplicatedLinear(
                    config.hidden_size,
                    config.n_routed_experts,
                    bias=False,
                    quant_config=None,
                    prefix=f"{prefix}.mlp.gate",
                )
                self.mlp.gate.precast_fp32_weight = True
                self.mlp.gate.bias_vl = None
                if getattr(config, "vision_n_layers", 0) > 0:
                    self.mlp.gate.bias_vl = nn.Parameter(
                        torch.empty(config.n_routed_experts, dtype=torch.float32),
                        requires_grad=False,
                    )
                if self.layer_idx < int(getattr(config, "num_hash_layers", 0)):
                    self.mlp.gate.tid2eid = nn.Parameter(
                        torch.zeros(
                            config.vocab_size,
                            config.num_experts_per_tok,
                            dtype=torch.int32,
                        ),
                        requires_grad=False,
                    )
                    self.mlp.gate.e_score_correction_bias = None
                else:
                    self.mlp.gate.tid2eid = None
                    self.mlp.gate.e_score_correction_bias = nn.Parameter(
                        torch.empty(config.n_routed_experts, dtype=torch.float32),
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
            if self.is_window_afd:
                from afd_plugin.model_executor.models.npu.deepseek_v41_window import (
                    AFDDeepseekV41RoutedMoE,
                    AFDDeepseekV41SharedMoE,
                )

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
                        vllm_config=vllm_config,
                        config=config,
                        quant_config=vllm_config.quant_config,
                        prefix=f"{prefix}.mlp",
                        expert_layout=expert_layout,
                    )
            else:
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
        x, continuation = self.forward_attention_to_remote_ffn(
            positions,
            hidden_states,
            pre_mix,
            llama_4_scaling,
            input_ids=input_ids,
        )
        return self.complete_remote_ffn(x, continuation)

    # Patch reason: layer-major U2 must split V4.1 before the remote FFN HC
    # post while preserving the delayed ``pre_mix`` for the next layer.
    # Patch functionality: expose the Attention-side prefix and return the
    # connector-backed F2A tensor plus its V4.1 continuation state.
    # Signature: matches the adapter-local forward contract; no new caller
    # parameters beyond the native ``input_ids`` side channel.
    def forward_attention_to_remote_ffn(
        self,
        positions,
        hidden_states,
        pre_mix,
        llama_4_scaling=None,
        *,
        input_ids=None,
    ):
        """Run Attention and dispatch the normalized MoE input.

        The returned continuation keeps the FFN-side mHC state alive until
        the matching F2A receive.  This is the V4.1 equivalent of the V4
        layer-major U2 split and is also used by the ordinary U1 path.
        """
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
        send_kwargs: dict[str, torch.Tensor] = {}
        if self.compute_gate_on_attention:
            from afd_plugin.model_executor.models.npu.deepseek_v41_window import (
                compute_v41_window_gate,
            )

            topk_weights, topk_ids = compute_v41_window_gate(
                gate=self.mlp.gate,
                config=self.config,
                hidden_states=x,
                input_ids=input_ids,
                top_k=self.top_k,
            )
            send_kwargs["expert_ids"] = topk_ids
            send_kwargs["expert_scales"] = topk_weights
        transfer = self.mlp.dispatch_remote_ffn(
            x,
            input_ids=input_ids if self.layer_idx == 0 else None,
            **send_kwargs,
        )
        ffn_output = self.mlp.receive_remote_ffn(transfer, layer_idx=self.layer_idx)
        return ffn_output, (residual, ffn_post, ffn_comb, ffn_pre)

    # Patch reason: U2 completes the remote F2A tensor after the connector
    # dependency is consumed by the current Attention stream.
    # Patch functionality: finish FFN HC post and return the next layer's
    # V4.1 ``pre_mix`` state.
    # Signature: adapter-local continuation helper.
    def complete_remote_ffn(self, ffn_output, continuation):
        residual, post, comb, pre_mix = continuation
        return self.hc_post(ffn_output, residual, post, comb), pre_mix

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | AFDF2ATransferPayload:
        if self.afd_role != _FFN_ROLE:
            raise RuntimeError("DeepSeek-V4.1 Attention role has no local MoE")
        if self.is_window_afd:
            from afd_plugin.model_executor.models.npu.deepseek_v41_window import (
                compute_v41_window_ffn,
            )

            if kwargs.get("group_list") is None:
                raise RuntimeError("DeepSeek-V4.1 Window FFN requires group_list")
            return compute_v41_window_ffn(
                self,
                hidden_states,
                kwargs["group_list"],
                kwargs.get("dynamic_scales"),
                int(kwargs.get("group_list_type", 1)),
                kwargs.get("topk_scales"),
                kwargs.get("expand_x_shared"),
                kwargs.get("dynamic_scales_shared"),
            )
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
        self._window_global_mxfp_weights = None
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
        self.use_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )
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
        speculative_config = getattr(vllm_config, "speculative_config", None)
        uses_draft_model = bool(
            speculative_config is not None
            and (
                getattr(speculative_config, "use_eagle", lambda: False)()
                or getattr(speculative_config, "uses_draft_model", lambda: False)()
            )
        )
        if (
            self.afd_role == _ATTENTION_ROLE
            and native.get_pp_group().is_last_rank
            and uses_draft_model
        ):
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_mult * config.hidden_size,
                dtype=vllm_config.model_config.dtype,
                device=self.device,
            )
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        self._dspark_target_state: DeepseekV41DSparkTargetState | None = None
        self._dspark_hidden_buffer: torch.Tensor | None = None
        if self.afd_role == _ATTENTION_ROLE and is_dspark_config(vllm_config):
            self._dspark_target_state = DeepseekV41DSparkTargetState.from_config(config)
            if not self._dspark_target_state.enabled:
                raise RuntimeError(
                    "DeepSeek-V4.1 AFD DSpark requires dspark_target_layer_ids"
                )
            self.aux_hidden_state_layers = self._dspark_target_state.aux_layer_ids
            self._dspark_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                len(self._dspark_target_state.target_layer_ids) * config.hidden_size,
                dtype=vllm_config.model_config.dtype,
                device=self.device,
            )
        self.has_engram = False
        self.engram_hash = None
        self.requires_uncompiled_fallback = False
        self.make_empty_intermediate_tensors = self._make_empty_intermediate_tensors

        if self.afd_role == _ATTENTION_ROLE and self.topk_indices_buffer is not None:
            topology = native.build_layer_plan(config)
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens

            def make_shared_attention_state(indices: torch.Tensor) -> Any:
                candidate = torch.full(
                    (max_tokens, 1, topology.candidate_topk_blocks),
                    -1,
                    dtype=torch.int32,
                    device=self.topk_indices_buffer.device,
                )
                return native.DeepseekV41SharedAttentionState(
                    indices,
                    candidate,
                    torch.zeros(
                        (max_tokens, 1),
                        dtype=torch.int32,
                        device=self.topk_indices_buffer.device,
                    ),
                    torch.zeros(
                        (max_tokens, 1),
                        dtype=torch.int32,
                        device=self.topk_indices_buffer.device,
                    ),
                )

            # DBO stages execute layer-major.  A single native shared-state
            # buffer lets stage 1 overwrite stage 0's sparse indices before
            # the next layer consumes them, so each stage owns one state.
            self._shared_attention_states = {
                0: make_shared_attention_state(self.topk_indices_buffer),
                1: make_shared_attention_state(
                    torch.empty_like(self.topk_indices_buffer),
                ),
            }
            self.shared_attention_states = self._shared_attention_states
            self.shared_attention_state = self._shared_attention_states[0]
            for layer in self.layers:
                if isinstance(layer, AFDDeepseekV41DecoderLayer):
                    layer.self_attn.shared_state = self.shared_attention_state
        else:
            self.shared_attention_state = None
            self._shared_attention_states = {}
            self.shared_attention_states = self._shared_attention_states

    def _activate_attention_shared_state(self, stage_idx: int) -> None:
        """Select the DSA state owned by the current eager/DBO stage."""
        state = self._shared_attention_states.get(int(stage_idx))
        if state is None:
            state = self.shared_attention_state
        for layer in self.layers:
            if isinstance(layer, AFDDeepseekV41DecoderLayer):
                layer.self_attn.shared_state = state

    def get_shared_attention_state(self, stage_idx: int = 0) -> Any:
        """Return the DSA handoff buffers for one eager/DBO stage."""
        return self._shared_attention_states.get(
            int(stage_idx), self.shared_attention_state
        )

    def _reset_attention_shared_states(self) -> None:
        """Reset native per-forward DSA state before a new request."""

        for state in self._shared_attention_states.values():
            reset = getattr(state, "reset", None)
            if callable(reset):
                reset()

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
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        else:
            if intermediate_tensors is None:
                raise RuntimeError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]
        self._reset_attention_shared_states()
        pre_mix = hidden_states.new_zeros(
            hidden_states.shape[0], self.hc_mult, dtype=torch.float32
        )
        pre_mix[:, 0] = 1.0
        if self._dspark_target_state is not None:
            self._dspark_target_state.clear()
        self._activate_attention_shared_state(
            int(getattr(get_forward_context(), "ubatch_idx", 0)),
        )
        for layer in self.layers[self.start_layer : self.end_layer]:
            if self._dspark_target_state is not None:
                self._dspark_target_state.collect(layer.layer_idx, hidden_states)
            hidden_states, pre_mix = layer(
                positions,
                hidden_states,
                pre_mix,
                None,
                input_ids=input_ids,
            )
        if not native.get_pp_group().is_last_rank:
            return native.IntermediateTensors({"hidden_states": hidden_states})
        if self._mtp_hidden_buffer is not None:
            self._mtp_hidden_buffer[: hidden_states.shape[0]].copy_(
                hidden_states.flatten(1),
            )
        if self._dspark_target_state is not None:
            assert self._dspark_hidden_buffer is not None
            self._dspark_target_state.store(self._dspark_hidden_buffer)
        hidden_states = native.DeepseekV41DecoderLayer.hc_collapse(
            hidden_states, pre_mix
        )
        output = self.norm(hidden_states)
        if self._dspark_target_state is not None:
            return output, [self._dspark_hidden_buffer[: hidden_states.shape[0], :]]
        return output

    # Patch reason: native V4.1 has no AFD layer-major execution entry point.
    # Patch functionality: run two P2P HCCL ubatches while carrying one
    # ``pre_mix`` tensor and one deferred F2A continuation per stage.
    # Signature: matches ``AscendUBatchWrapper``'s model hook.
    def forward_ubatches_layer_major(self, ubatch_metadata: list[Any]) -> list[Any]:
        """Run two V4.1 micro-batches layer-major with deferred F2A receives.

        V4.1's ``pre_mix`` is the output of the previous layer's FFN HC
        pre-stage, so each micro-batch carries its own value alongside the
        hidden state.  The connector receive is enqueued by the layer and is
        consumed through ``wait_for_attention_stage_receive`` before the
        pending HC post is completed.
        """
        if self.afd_role != _ATTENTION_ROLE:
            raise RuntimeError("DeepSeek-V4.1 FFN model execution is connector-driven")
        if len(ubatch_metadata) != 2:
            raise RuntimeError(
                "DeepSeek-V4.1 layer-major execution requires exactly two stages; "
                f"got {len(ubatch_metadata)}"
            )

        stage_contexts = [item.context.forward_context for item in ubatch_metadata]
        connectors = []
        for stage_idx, (item, forward_context) in enumerate(
            zip(ubatch_metadata, stage_contexts, strict=True)
        ):
            afd_metadata = (forward_context.additional_kwargs or {}).get("afd_metadata")
            connector = getattr(afd_metadata, "connector", None)
            if connector is None:
                raise RuntimeError(
                    "DeepSeek-V4.1 layer-major U2 requires AFD connector metadata"
                )
            if int(getattr(forward_context, "ubatch_idx", -1)) != stage_idx:
                raise RuntimeError(
                    "DeepSeek-V4.1 layer-major stage context order is invalid: "
                    f"expected={stage_idx} actual="
                    f"{getattr(forward_context, 'ubatch_idx', None)}"
                )
            forward_context.input_ids = item.input_ids
            forward_context.afd_layer_major_u2 = True
            connectors.append(connector)

        connector = connectors[0]
        if any(stage_connector is not connector for stage_connector in connectors[1:]):
            raise RuntimeError(
                "DeepSeek-V4.1 layer-major U2 stages must share one connector"
            )
        require_idle = getattr(connector, "require_attention_pipeline_idle", None)
        wait_for_receive = getattr(connector, "wait_for_attention_stage_receive", None)
        reset_pipeline = getattr(connector, "reset_attention_pipeline_state", None)
        if not all(callable(method) for method in (require_idle, wait_for_receive)):
            raise RuntimeError(
                "DeepSeek-V4.1 layer-major U2 requires the HCCL stream connector"
            )

        hidden_ubatches: list[torch.Tensor] = []
        pre_mix_ubatches: list[torch.Tensor] = []
        pending_layers: list[AFDDeepseekV41DecoderLayer | None] = [None, None]
        pending_continuations: list[tuple[torch.Tensor, ...] | None] = [None, None]
        dspark_states = (
            [
                DeepseekV41DSparkTargetState(self._dspark_target_state.target_layer_ids)
                for _ in ubatch_metadata
            ]
            if self._dspark_target_state is not None
            else []
        )
        self._reset_attention_shared_states()
        require_idle()
        try:
            for stage_idx, (item, forward_context) in enumerate(
                zip(ubatch_metadata, stage_contexts, strict=True)
            ):
                with override_forward_context(forward_context):
                    self._activate_attention_shared_state(stage_idx)
                    if native.get_pp_group().is_first_rank:
                        hidden_states = (
                            item.inputs_embeds
                            if item.inputs_embeds is not None
                            else self.embed_input_ids(item.input_ids)
                        )
                        hidden_states = hidden_states.unsqueeze(1).repeat(
                            1, self.hc_mult, 1
                        )
                    else:
                        if item.intermediate_tensors is None:
                            raise RuntimeError(
                                "pipeline stage requires intermediate tensors"
                            )
                        hidden_states = item.intermediate_tensors["hidden_states"]
                hidden_ubatches.append(hidden_states)
                pre_mix = hidden_states.new_zeros(
                    hidden_states.shape[0], self.hc_mult, dtype=torch.float32
                )
                pre_mix[:, 0] = 1.0
                pre_mix_ubatches.append(pre_mix)

            for layer_offset, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)
            ):
                for stage_idx, (item, forward_context) in enumerate(
                    zip(ubatch_metadata, stage_contexts, strict=True)
                ):
                    with override_forward_context(forward_context):
                        # Layer-major alternates stages after each receive;
                        # rebind the native DSA buffers before every layer so
                        # stage 0 cannot consume stage 1's sparse indices.
                        self._activate_attention_shared_state(stage_idx)
                        if layer_offset > 0:
                            wait_for_receive(
                                stage_idx=stage_idx,
                                tensor=hidden_ubatches[stage_idx],
                            )
                            pending_layer = pending_layers[stage_idx]
                            continuation = pending_continuations[stage_idx]
                            if pending_layer is None or continuation is None:
                                raise RuntimeError(
                                    "DeepSeek-V4.1 layer-major stage has no "
                                    f"pending layer: stage={stage_idx}"
                                )
                            hidden_states, pre_mix = pending_layer.complete_remote_ffn(
                                hidden_ubatches[stage_idx], continuation
                            )
                            hidden_ubatches[stage_idx] = hidden_states
                            pre_mix_ubatches[stage_idx] = pre_mix
                        if dspark_states:
                            dspark_states[stage_idx].collect(
                                layer.layer_idx,
                                hidden_ubatches[stage_idx],
                            )

                        hidden_states, continuation = (
                            layer.forward_attention_to_remote_ffn(
                                item.positions,
                                hidden_ubatches[stage_idx],
                                pre_mix_ubatches[stage_idx],
                                None,
                                input_ids=item.input_ids,
                            )
                        )
                        hidden_ubatches[stage_idx] = hidden_states
                        pending_layers[stage_idx] = layer
                        pending_continuations[stage_idx] = continuation

            for stage_idx, forward_context in enumerate(stage_contexts):
                with override_forward_context(forward_context):
                    wait_for_receive(
                        stage_idx=stage_idx,
                        tensor=hidden_ubatches[stage_idx],
                    )
                    pending_layer = pending_layers[stage_idx]
                    continuation = pending_continuations[stage_idx]
                    if pending_layer is None or continuation is None:
                        raise RuntimeError(
                            "DeepSeek-V4.1 layer-major stage has no final pending "
                            f"layer: stage={stage_idx}"
                        )
                    hidden_ubatches[stage_idx], pre_mix_ubatches[stage_idx] = (
                        pending_layer.complete_remote_ffn(
                            hidden_ubatches[stage_idx], continuation
                        )
                    )
        except BaseException:
            if callable(reset_pipeline):
                reset_pipeline()
            self._activate_attention_shared_state(0)
            raise

        if not native.get_pp_group().is_last_rank:
            self._activate_attention_shared_state(0)
            return [
                native.IntermediateTensors({"hidden_states": hidden_states})
                for hidden_states in hidden_ubatches
            ]

        outputs: list[Any] = []
        dspark_buffer_offset = 0
        mtp_buffer_offset = 0
        for stage_idx, forward_context in enumerate(stage_contexts):
            with override_forward_context(forward_context):
                pre_mix = pre_mix_ubatches[stage_idx]
                if self._mtp_hidden_buffer is not None:
                    mtp_hidden = hidden_ubatches[stage_idx].flatten(1)
                    next_mtp_offset = mtp_buffer_offset + mtp_hidden.shape[0]
                    self._mtp_hidden_buffer[mtp_buffer_offset:next_mtp_offset].copy_(
                        mtp_hidden,
                    )
                    mtp_buffer_offset = next_mtp_offset
                output = self.norm(
                    native.DeepseekV41DecoderLayer.hc_collapse(
                        hidden_ubatches[stage_idx], pre_mix
                    )
                )
                if dspark_states:
                    assert self._dspark_hidden_buffer is not None
                    flat = dspark_states[stage_idx].flatten()
                    next_offset = dspark_buffer_offset + flat.shape[0]
                    self._dspark_hidden_buffer[dspark_buffer_offset:next_offset].copy_(
                        flat
                    )
                    outputs.append(
                        (
                            output,
                            [
                                self._dspark_hidden_buffer[
                                    dspark_buffer_offset:next_offset
                                ]
                            ],
                        )
                    )
                    dspark_buffer_offset = next_offset
                else:
                    outputs.append(output)
        # Leave ordinary eager/U1 callers on the canonical stage-0 state.
        self._activate_attention_shared_state(0)
        return outputs

    def compute_ffn_output(
        self, hidden_states: torch.Tensor, layer_idx: int, **kwargs: Any
    ) -> torch.Tensor | AFDF2ATransferPayload:
        return self.layers[layer_idx].compute_ffn_output(hidden_states, **kwargs)

    def compute_window_global_ffn_output(
        self,
        hidden_states: torch.Tensor,
        group_list: torch.Tensor,
        actual_token_num: torch.Tensor,
    ) -> torch.Tensor:
        """Run asynchronous Window's layer-major packed FFN transaction."""

        if self._window_global_mxfp_weights is None:
            self.initialize_window_global_mxfp_weights()
        from afd_plugin.model_executor.models.npu.deepseek_v4_window_ffn import (
            compute_window_global_mxfp_ffn,
        )

        return compute_window_global_mxfp_ffn(
            hidden_states=hidden_states,
            compact_group_list=group_list,
            actual_token_num=actual_token_num,
            weights=self._window_global_mxfp_weights,
        )

    def initialize_window_global_mxfp_weights(self) -> None:
        """Pack all role-local V4.1 Window weights for layer-major GMM."""

        if self._window_global_mxfp_weights is not None:
            return
        from afd_plugin.model_executor.models.npu.deepseek_v4_window_ffn import (
            build_window_global_mxfp_weights,
        )

        local_layers = [
            layer
            for layer in self.layers
            if not isinstance(layer, native.PPMissingLayer)
        ]
        if len(local_layers) != int(self.config.num_hidden_layers):
            raise RuntimeError(
                "DeepSeek-V4.1 Window global FFN requires every decoder layer "
                f"on the FFN rank: local={len(local_layers)} "
                f"total={self.config.num_hidden_layers}"
            )
        self._window_global_mxfp_weights = build_window_global_mxfp_weights(
            local_layers
        )

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        """Return decoder layers whose FFN is owned by the remote role."""

        return tuple(
            layer_idx
            for layer_idx, layer in enumerate(self.layers)
            if isinstance(layer, AFDDeepseekV41DecoderLayer)
            and layer.is_moe_layer
        )

    def get_experts_routing_spec(self, layer_idx: int) -> AFDExpertRoutingSpec:
        """Return the native router shape for generic FFN runner paths."""

        from afd_plugin.connectors.metadata import AFDExpertRoutingSpec

        layer = self.layers[layer_idx]
        gate = getattr(getattr(layer, "mlp", None), "gate", None)
        if gate is None:
            raise RuntimeError(
                "DeepSeek-V4.1 layer has no Attention-side router; "
                "Window uses connector-driven metadata"
            )
        return AFDExpertRoutingSpec(
            router_logits_width=int(self.config.n_routed_experts),
            router_logits_dtype=getattr(gate, "out_dtype", None)
            or gate.weight.dtype,
        )

    def get_mtp_target_hidden_states(self):
        return self._mtp_hidden_buffer

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        if (
            self._dspark_target_state is not None
            and tuple(layers) != self._dspark_target_state.aux_layer_ids
        ):
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark auxiliary layers do not match target config: "
                "expected="
                f"{self._dspark_target_state.aux_layer_ids} "
                f"actual={tuple(layers)}"
            )
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
        self.config = native.normalize_deepseek_v41_config(
            vllm_config.model_config.hf_config
        )
        self.quant_config = vllm_config.quant_config
        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=native.maybe_prefix(prefix, "model")
        )
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
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.requires_uncompiled_fallback = False
        self.num_moe_layers = self.config.num_hidden_layers
        self.set_moe_parameters()
        self.moe_comm_methods = getattr(
            native.AscendDeepseekV41LLMForCausalLM, "moe_comm_methods", {}
        )

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

    # Patch reason: ``AscendUBatchWrapper`` is installed around the CausalLM
    # object, while the layer-major implementation belongs to the backbone.
    # Patch functionality: expose the backbone hook without duplicating the
    # U2 scheduler implementation.
    # Signature: matches the wrapper hook used by the NPU model runner.
    def forward_ubatches_layer_major(self, ubatch_metadata: list[Any]) -> list[Any]:
        return self.model.forward_ubatches_layer_major(ubatch_metadata)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.model.get_experts_layer_indices()

    def get_experts_routing_spec(self, layer_idx: int) -> AFDExpertRoutingSpec:
        return self.model.get_experts_routing_spec(layer_idx)

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
            from afd_plugin.model_executor.models.npu.deepseek_v41_window import (
                AFDDeepseekV41RoutedMoE,
                AFDDeepseekV41SharedMoE,
            )

            for layer in self.model.layers:
                mlp = getattr(layer, "mlp", None)
                if isinstance(mlp, (native.DeepseekV41MoE, AFDDeepseekV41RoutedMoE)):
                    example_moe = mlp
                    self.moe_mlp_layers.append(mlp)
                    self.moe_layers.append(mlp.experts)
                elif isinstance(mlp, AFDDeepseekV41SharedMoE):
                    # A dedicated Window shared rank has no routed-expert
                    # runner. Keep its metadata for native loader bookkeeping
                    # while leaving ``moe_layers`` empty for connector-driven
                    # execution.
                    if example_moe is None:
                        example_moe = mlp
                    self.moe_mlp_layers.append(mlp)
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        native.DeepseekV41MixtureOfExperts.extract_moe_parameters(self, example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor):
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_ffn_output(
        self, hidden_states: torch.Tensor, layer_idx: int, **kwargs: Any
    ) -> torch.Tensor | AFDF2ATransferPayload:
        return self.model.compute_ffn_output(hidden_states, layer_idx, **kwargs)

    def compute_window_global_ffn_output(
        self,
        hidden_states: torch.Tensor,
        group_list: torch.Tensor,
        actual_token_num: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.compute_window_global_ffn_output(
            hidden_states,
            group_list,
            actual_token_num,
        )

    def initialize_window_global_mxfp_weights(self) -> None:
        self.model.initialize_window_global_mxfp_weights()

    def get_mtp_target_hidden_states(self):
        return self.model.get_mtp_target_hidden_states()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        window_ffn_kind = None
        if (
            self.afd_role == _FFN_ROLE
            and self.afd_config.connector == "WindowAFDConnector"
        ):
            role_rank = resolve_role_rank(self.vllm_config, self.afd_config)
            window_ffn_kind = _window_expert_layout(
                self.vllm_config,
                role_rank,
            ).kind
        role_weights = _iter_role_weights(
            weights,
            role=self.afd_role,
            num_hidden_layers=int(self.config.num_hidden_layers),
            window_ffn_kind=window_ffn_kind,
            compute_gate_on_attention=bool(self.afd_config.compute_gate_on_attention),
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
