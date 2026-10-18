# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek-V4.1 Window routing and role-local MoE building blocks.

The Window connector owns dispatch/combine.  This module only contains the
V4.1-specific pieces which are absent from the V4 wrapper: the native
``FusedMoEFactory`` weight owner, the dedicated shared-expert MLP, and the
native V4.1 router contract (hash and vision-token routing).  Keeping these
pieces separate lets the model wrapper reuse the existing Window transport
and global FFN implementation without duplicating connector code.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from vllm_ascend.models.deepseek_v41 import model as native

from afd_plugin.connectors import AFDF2ATransferPayload
from afd_plugin.distributed.topology import AFDWindowExpertLayout


def selected_expert_num(config: Any) -> int:
    """Return the Window payload width (routed top-k plus shared slot)."""

    return int(config.num_experts_per_tok) + int(config.n_shared_experts)


def _install_local_expert_map(
    experts: nn.Module,
    *,
    routed_expert_num: int,
    layout: AFDWindowExpertLayout,
) -> None:
    """Install a global-to-local map on both vLLM MoE API generations.

    ``FusedMoEFactory`` returns an ``AscendMoERunner`` on the pinned
    vLLM-Ascend stack.  Older vLLM-Ascend revisions exposed the map directly
    on that runner, while current revisions keep it on ``routed_experts``.
    The map is intentionally CPU-created; vLLM's expert loader moves it with
    the rest of the module before the first dispatch.
    """

    holder = getattr(experts, "routed_experts", experts)
    manager = holder.expert_map_manager
    expert_map = torch.full((routed_expert_num,), -1, dtype=torch.int32)
    start = layout.local_expert_start
    end = start + layout.local_expert_count
    expert_map[start:end] = torch.arange(layout.local_expert_count, dtype=torch.int32)

    # ``ExpertMapManager`` and ``RoutedExperts`` keep separate cached values in
    # the current API.  Update both so checkpoint loading and execution agree.
    manager._expert_map = expert_map
    manager._local_num_experts = layout.local_expert_count
    manager.global_num_experts = routed_expert_num
    holder._expert_map = expert_map
    holder.local_num_experts = layout.local_expert_count
    holder.global_num_experts = routed_expert_num
    # The factory was intentionally built with local-sized tensors.  The
    # loader still has to enumerate global checkpoint IDs so the map above can
    # translate them into those local slots.
    moe_config = holder.moe_config
    moe_config.num_experts = routed_expert_num
    moe_config.num_logical_experts = routed_expert_num
    moe_config.num_local_experts = layout.local_expert_count
    if hasattr(holder, "n_local_physical_experts"):
        holder.n_local_physical_experts = layout.local_expert_count
    if hasattr(holder, "n_physical_experts"):
        holder.n_physical_experts = layout.local_expert_count
    if hasattr(holder, "physical_expert_start"):
        holder.physical_expert_start = start
        holder.physical_expert_end = end


class AFDDeepseekV41RoutedMoE(native.DeepseekV41MoE):
    """V4.1 routed experts restricted to one Window FFN rank.

    The native ``DeepseekV41MoE`` creates a router and all global experts. A
    Window FFN rank receives already-routed rows, so it only needs the native
    expert weight owner.  The wrapper therefore creates a local-sized
    ``FusedMoEFactory`` and maps global checkpoint IDs to local slots.
    """

    def __init__(
        self,
        *,
        vllm_config: Any,
        config: Any,
        quant_config: Any,
        prefix: str,
        expert_layout: AFDWindowExpertLayout,
    ) -> None:
        # Do not run native ``DeepseekV41MoE.__init__``: it allocates a global
        # router and all experts before Window assigns a rank-local map.
        nn.Module.__init__(self)
        if expert_layout.is_shared:
            raise ValueError("V4.1 routed Window MoE requires a routed layout")
        self.vllm_config = vllm_config
        self.config = config
        self.layer_idx = int(prefix.split(".")[-2])
        self.routed_scaling_factor = float(
            getattr(config, "routed_scaling_factor", 1.5)
        )
        self.swiglu_limit = getattr(config, "swiglu_limit", None)
        self.n_routed_experts = int(config.n_routed_experts)
        self.n_shared_experts = 0
        # Keep the native MixtureOfExperts metadata contract even though the
        # Window rank owns only a contiguous local slice of routed experts.
        # ``extract_moe_parameters`` reads these fields before the first
        # runner dispatch; omitting them makes role-local construction fail
        # before any checkpoint weight can be loaded.
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = 0
        self.n_physical_experts = self.n_logical_experts
        self.n_local_physical_experts = expert_layout.local_expert_count
        self.physical_expert_start = expert_layout.local_expert_start
        self.physical_expert_end = (
            expert_layout.local_expert_start + expert_layout.local_expert_count
        )
        self.enable_eplb = False
        self.is_sequence_parallel = bool(
            vllm_config.parallel_config.use_sequence_parallel_moe
        )
        self.experts = native.FusedMoEFactory(
            shared_experts=None,
            gate=None,
            num_experts=expert_layout.local_expert_count,
            top_k=int(config.num_experts_per_tok),
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.moe_intermediate_size),
            quant_config=quant_config,
            tp_size=1,
            dp_size=1,
            use_grouped_topk=True,
            num_expert_group=int(getattr(config, "n_group", 1)),
            topk_group=int(getattr(config, "topk_group", 1)),
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "softmax"),
            routed_scaling_factor=self.routed_scaling_factor,
            swiglu_limit=self.swiglu_limit,
            e_score_correction_bias=None,
            enable_eplb=False,
            num_redundant_experts=0,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=0,
            fuse_shared_experts=False,
        )
        _install_local_expert_map(
            self.experts,
            routed_expert_num=self.n_routed_experts,
            layout=expert_layout,
        )

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Execute local rows when a caller explicitly supplies native routing."""

        return self.experts(hidden_states=hidden_states, **kwargs)


class AFDDeepseekV41SharedMoE(nn.Module):
    """The V4.1 shared expert placed on the dedicated Window rank."""

    def __init__(
        self,
        *,
        config: Any,
        parallel_config: Any,
        quant_config: Any,
        prefix: str,
    ) -> None:
        super().__init__()
        # Keep the native MoE metadata surface available to the CausalLM
        # wrapper.  The dedicated shared rank has no routed ``experts``
        # runner, but native weight loading still reads these cardinalities.
        self.n_routed_experts = 0
        self.n_shared_experts = int(config.n_shared_experts)
        self.n_logical_experts = 0
        self.n_redundant_experts = 0
        self.n_physical_experts = 0
        self.n_local_physical_experts = 0
        self.routed_scaling_factor = float(
            getattr(config, "routed_scaling_factor", 1.5)
        )
        self.shared_experts = native.DeepseekV41MLP(
            hidden_size=int(config.hidden_size),
            intermediate_size=(
                int(config.moe_intermediate_size) * int(config.n_shared_experts)
            ),
            hidden_act=config.hidden_act,
            swiglu_limit=getattr(config, "swiglu_limit", None),
            quant_config=quant_config,
            is_sequence_parallel=bool(parallel_config.use_sequence_parallel_moe),
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.shared_experts(hidden_states)


def compute_v41_window_gate(
    *,
    gate: nn.Module,
    config: Any,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute V4.1 routed IDs/scales on Attention for Window A2F.

    Hash layers must use the native token-id table.  Vision rows use the
    native V4.1 ``bias_vl`` contract; regular text rows use the Ascend fused
    grouped selector.  The function deliberately returns only routed top-k
    columns because the Window connector reserves the final column for the
    dedicated shared expert.
    """

    from vllm_ascend.device.device_op import DeviceOperator

    if input_ids is None and getattr(gate, "tid2eid", None) is not None:
        raise RuntimeError("DeepSeek-V4.1 Window hash routing requires input_ids")
    router_logits = torch.nn.functional.linear(
        hidden_states.float(), gate.weight.float()
    )
    bias = getattr(gate, "e_score_correction_bias", None)
    if bias is not None and bias.dtype != router_logits.dtype:
        bias = bias.to(router_logits.dtype)
    tid2eid = getattr(gate, "tid2eid", None)
    bias_vl = getattr(gate, "bias_vl", None)
    scoring_func = getattr(config, "scoring_func", "softmax")
    if tid2eid is not None or bias_vl is not None:
        if input_ids is None:
            raise RuntimeError(
                "DeepSeek-V4.1 hash/vision Window routing requires input_ids"
            )
        input_ids = torch.where(input_ids == -1, 0, input_ids).to(torch.int64)
        if bias_vl is not None:
            from vllm_ascend.ops.fused_moe.router.fused_topk_router import (
                select_deepseek_v4_vision_experts,
            )

            return select_deepseek_v4_vision_experts(
                router_logits=router_logits,
                input_ids=input_ids,
                tid2eid=tid2eid,
                bias_vl=bias_vl,
                text_bias=bias,
                top_k=top_k,
                renormalize=bool(getattr(config, "norm_topk_prob", True)),
                routed_scaling_factor=1.0,
                image_sentinel_lo=int(
                    getattr(config, "image_sentinel_base_id", 129257)
                ),
            )
        topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
            x=router_logits,
            k=top_k,
            bias=bias,
            input_ids=input_ids,
            tid2eid=tid2eid.to(torch.int32),
            k_group=int(getattr(config, "topk_group", 1)),
            group_count=int(getattr(config, "n_group", 1)),
            routed_scaling_factor=1.0,
            eps=1e-20,
            group_select_mode=1,
            renorm=0,
            norm_type=2,
            out_flag=False,
        )
        if bool(getattr(config, "norm_topk_prob", True)):
            topk_weights = topk_weights / topk_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-20)
        return topk_weights.to(torch.float32), topk_ids

    norm_type = 0 if scoring_func == "softmax" else 1
    topk_weights, topk_ids, _ = DeviceOperator.moe_gating_top_k(
        router_logits,
        k=top_k,
        k_group=int(getattr(config, "topk_group", 1)),
        group_count=int(getattr(config, "n_group", 1)),
        group_select_mode=1,
        renorm=int(bool(getattr(config, "norm_topk_prob", True))),
        norm_type=norm_type,
        out_flag=False,
        routed_scaling_factor=1.0,
        eps=1e-20,
        bias_opt=bias,
    )
    return topk_weights.to(torch.float32), topk_ids


def v41_window_expert_holder(moe: nn.Module) -> nn.Module:
    """Return the quantized routed-weight owner across runtime API versions."""

    return getattr(moe.experts, "routed_experts", moe.experts)


def _cumulative_group_list(
    group_list: torch.Tensor,
    group_list_type: int,
) -> tuple[torch.Tensor, int]:
    """Normalize Window's count/type-2 table for native GMM MLP methods."""

    if group_list.ndim == 2:
        if group_list.shape[1] != 2:
            raise RuntimeError(
                "DeepSeek-V4.1 Window group_list must have shape [G, 2]"
            )
        group_list = group_list[:, 1]
        group_list_type = 1
    if group_list.ndim != 1:
        raise RuntimeError(
            "DeepSeek-V4.1 Window group_list must be one- or two-dimensional"
        )
    if int(group_list_type) == 0:
        return group_list, 0
    if int(group_list_type) != 1:
        raise RuntimeError(
            "DeepSeek-V4.1 Window supports group_list_type 0 or 1, "
            f"got {group_list_type}"
        )
    return torch.cumsum(group_list, dim=0), 0


def compute_v41_window_ffn(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    dynamic_scales: torch.Tensor | None,
    group_list_type: int,
    topk_scales: torch.Tensor | None = None,
    expand_x_shared: torch.Tensor | None = None,
    dynamic_scales_shared: torch.Tensor | None = None,
) -> AFDF2ATransferPayload:
    """Run one V4.1 Window FFN rank's local MLP and package F2A output.

    Attention computes routing and Window batching returns expert rows already
    ordered for this rank.  Calling ``AscendMoEScheme`` directly preserves the
    native V4.1 quantization implementation (including W4A8-MXFP and W8A8-
    MXFP), while bypassing its router and token dispatcher.  The shared rank
    uses its native dense MLP; routed ranks use the quant method's GMM hooks.
    """

    mlp = layer.mlp
    shared = getattr(mlp, "shared_experts", None)
    experts = getattr(mlp, "experts", None)
    if experts is None:
        if shared is None:
            raise RuntimeError(
                "DeepSeek-V4.1 Window FFN layer has no routed or shared expert"
            )
        output = shared(
            expand_x_shared if expand_x_shared is not None else hidden_states
        )
        return AFDF2ATransferPayload(routed_output=output, shared_output=None)

    normalized_group_list, normalized_type = _cumulative_group_list(
        group_list, group_list_type
    )
    holder = getattr(experts, "routed_experts", experts)
    quant_method = holder.quant_method
    from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
    from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
    from vllm_ascend.ops.fused_moe.dataclass.moe_quant import (
        MoEMxfpParams,
        MoEQuantParams,
    )
    from vllm_ascend.quantization.quant_type import QuantType

    try:
        weights = quant_method.get_mlp_weights(holder)
    except (AttributeError, NotImplementedError):
        weights = quant_method.get_fused_mc2_weights(holder)
    if isinstance(weights, tuple):
        weights = MoEWeights(w1=weights[0], w2=weights[1])

    quant_type = holder.quant_type
    mxfp = None
    if quant_type in (
        QuantType.W8A8MXFP,
        QuantType.W4A4MXFP,
        QuantType.W4A8MXFP,
        QuantType.W4A16MXFP,
    ):
        import torch_npu

        mxfp = MoEMxfpParams(
            act_quant_type=torch.float8_e4m3fn,
            weight_quant_type=(
                torch_npu.float4_e2m1fn_x2
                if quant_type
                in (QuantType.W4A4MXFP, QuantType.W4A8MXFP, QuantType.W4A16MXFP)
                else torch.float8_e4m3fn
            ),
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            use_bf16=hidden_states.dtype == torch.bfloat16,
        )
    compute_input = MoEMlpComputeInput(
        hidden_states=hidden_states,
        group_list=normalized_group_list,
        group_list_type=normalized_type,
        dynamic_scale=dynamic_scales,
        topk_scales=topk_scales,
        weights=weights,
        quant=MoEQuantParams(quant_type=quant_type, mxfp=mxfp),
        fusion=False,
        layer=holder,
        activation=holder.activation,
        dynamic_eplb=False,
        swiglu_limit=float(getattr(holder, "swiglu_limit", 0.0) or 0.0),
    )
    from vllm_ascend.ops.fused_moe.moe_mlp import apply_moe_mlp

    routed_output, _ = apply_moe_mlp(compute_input, quant_method)
    routed_output = routed_output * float(mlp.routed_scaling_factor)
    return AFDF2ATransferPayload(
        routed_output=routed_output,
        shared_output=None,
    )


__all__ = [
    "AFDDeepseekV41RoutedMoE",
    "AFDDeepseekV41SharedMoE",
    "compute_v41_window_gate",
    "compute_v41_window_ffn",
    "selected_expert_num",
    "v41_window_expert_holder",
]
