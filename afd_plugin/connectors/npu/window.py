# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Window-based AFD connector initialization for Ascend NPU.

The connector owns the communication resources and the synchronous A2F/F2A
data path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import cann_ops_transformer as cot
import torch
import torch_npu
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup
from vllm.logger import init_logger

from afd_plugin.config import AFDConfig
from afd_plugin.config_utils import coerce_extra_int, coerce_extra_positive_int
from afd_plugin.connectors.base import AFDConnectorBase, ConnectorExtraInfo
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDTransferMetadata,
    AFDTransferState,
    AFDTransferContext,
)
from afd_plugin.distributed import (
    build_window_expert_layout,
    build_window_rank_mapping,
    init_afd_process_group,
)

logger = init_logger(__name__)

_COMM_CONTEXT_WINDOW_ALIGNMENT = 2 * 1024 * 1024

@dataclass(frozen=True, slots=True)
class WindowLayerBatch:
    """One layer's contiguous token range in an asynchronous FFN batch."""

    layer_idx: int
    token_start: int
    token_end: int
    group_list: torch.Tensor


@dataclass(slots=True)
class WindowAFDTransferState(AFDTransferState):
    """Operator-produced routing metadata for one A2F exchange."""

    expert_scales: torch.Tensor
    group_list: torch.Tensor | None = None
    dynamic_scale: torch.Tensor | None = None
    session_ids: torch.Tensor | None = None
    micro_batch_ids: torch.Tensor | None = None
    token_ids: torch.Tensor | None = None
    expert_offsets: torch.Tensor | None = None
    actual_token_num: torch.Tensor | None = None
    layer_batches: tuple[WindowLayerBatch, ...] = ()


@dataclass(frozen=True, slots=True)
class WindowAFDExtraInfo(ConnectorExtraInfo):
    """Window protocol options.

    The current implementation requires one micro-batch. Scheduling is
    lock-step or independent per Attention session according to ``async_dp``.
    """

    micro_batch_num: int = 1
    quant_mode: int = 2

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> WindowAFDExtraInfo:
        raw = {} if raw is None else raw
        if not isinstance(raw, Mapping):
            raise TypeError(
                "WindowAFDConnector connector_extra_config must be a mapping",
            )
        allowed = {"micro_batch_num", "quant_mode"}
        unknown = sorted(str(key) for key in raw if key not in allowed)
        if unknown:
            raise ValueError(
                "unknown WindowAFDConnector connector_extra_config field(s): "
                + ", ".join(unknown),
            )
        quant_mode = coerce_extra_int(
            raw.get("quant_mode", 2),
            field_name="quant_mode",
        )
        if quant_mode not in (0, 2):
            raise ValueError(
                "WindowAFDConnector quant_mode must be 0 or 2, " f"got {quant_mode}",
            )
        return cls(
            micro_batch_num=coerce_extra_positive_int(
                raw.get("micro_batch_num", 1),
                field_name="micro_batch_num",
            ),
            quant_mode=quant_mode,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "micro_batch_num": self.micro_batch_num,
            "quant_mode": self.quant_mode,
        }


def _align_up(value: int, alignment: int = 512) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _window_sizes(
    *,
    attention_size: int,
    micro_batch_num: int,
    micro_batch_size: int,
    selected_expert_num: int,
    hidden_size: int,
    quant_mode: int,
) -> tuple[int, int, int, int]:
    """Return Window sizes and per-token byte strides for both directions.

    The result is ``(attn_window_bytes, ffn_window_bytes,
    a2f_token_bytes, f2a_token_bytes)``.  Window sizes include both metadata
    and token data, with the final capacity aligned to 2 MiB for the external
    CommContext Window.  The formulas mirror ref/local_window_utils.py and are
    evaluated once from the configured maximum batch capacity.
    """

    if quant_mode == 2:
        # H INT8 bytes plus one FP32 scale, padded to a 512-byte record.
        a2f_token_bytes = _align_up(hidden_size + 4, 512)
    elif quant_mode == 0:
        # H fp16/bfloat16 values, each occupying two bytes.
        a2f_token_bytes = hidden_size * 2
    else:
        raise ValueError(f"unsupported Window quant_mode={quant_mode}")
    # F2A always returns H fp16/bfloat16 values.
    f2a_token_bytes = hidden_size * 2

    attention_window_info_bytes = _align_up(
        4 * selected_expert_num * micro_batch_size * micro_batch_num,
    )
    attention_window_data_bytes = (
        f2a_token_bytes
        * selected_expert_num
        * micro_batch_size
        * micro_batch_num
    )

    ffn_window_info_bytes = _align_up(
        4
        * (selected_expert_num * micro_batch_size + 2)
        * micro_batch_num
        * attention_size,
    )
    ffn_window_data_bytes = (
        a2f_token_bytes
        * selected_expert_num
        * micro_batch_size
        * micro_batch_num
        * attention_size
    )
    # The external Window passed to CommContextManager must satisfy the same
    # 2 MiB capacity alignment used by the operators' ccl_buffer_size helpers.
    attention_window_bytes = _align_up(
        attention_window_info_bytes + attention_window_data_bytes,
        _COMM_CONTEXT_WINDOW_ALIGNMENT,
    )
    ffn_window_bytes = _align_up(
        ffn_window_info_bytes + ffn_window_data_bytes,
        _COMM_CONTEXT_WINDOW_ALIGNMENT,
    )
    return (
        attention_window_bytes,
        ffn_window_bytes,
        a2f_token_bytes,
        f2a_token_bytes,
    )


class WindowAFDConnector(AFDConnectorBase):
    """Create the M2N communication context, Window, and schedule context."""

    yield_after_attn_send = True
    supports_connector_driven_loop = True
    is_window_connector = True
    requires_lockstep_dp_sync = True

    @classmethod
    def parse_extra_config(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> WindowAFDExtraInfo:
        return WindowAFDExtraInfo.from_mapping(raw)

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: Any,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        self.mapping = build_window_rank_mapping(afd_config, role_rank)
        self.world_rank = self.mapping.world_rank
        self.world_size = self.mapping.world_size
        self.attn_size = self.mapping.attention_size
        self.ffn_size = self.mapping.ffn_size
        self.async_mode = bool(afd_config.async_dp)
        self.requires_lockstep_dp_sync = not self.async_mode
        self.process_group: ProcessGroup | None = None
        self.hccl_comm_name: str | None = None
        self.window_tensor: torch.Tensor | None = None
        self.comm_buffer: Any | None = None
        self.attn_window_size = 0
        self.ffn_window_size = 0
        self.window_size = 0
        self.window_addr = 0
        self.context_holder: Any | None = None
        self.schedule_context: torch.Tensor | None = None
        self.expert_rank_table: torch.Tensor | None = None
        self.attn_rank_table: torch.Tensor | None = None
        self.local_expert_num = 0
        self._pending_transfers: dict[tuple[int, int], AFDTransferContext] = {}
        self._initialized = False

        hf_config = vllm_config.model_config.hf_config
        self.num_layers = int(hf_config.num_hidden_layers)
        self.hidden_size = int(hf_config.hidden_size)
        routed_topk = int(hf_config.num_experts_per_tok)
        shared_expert_num = int(hf_config.n_shared_experts)
        self.shared_expert_num = shared_expert_num
        self.routed_expert_num = int(hf_config.n_routed_experts)
        self.selected_expert_num = routed_topk + shared_expert_num
        self.expert_num = self.routed_expert_num + shared_expert_num
        self.micro_batch_size = int(vllm_config.scheduler_config.max_num_batched_tokens)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def init_afd_connector(self) -> None:
        if self._initialized:
            return
        if not self.afd_config.compute_gate_on_attention:
            raise ValueError(
                "WindowAFDConnector requires compute_gate_on_attention=true "
                "for the ref-style Attention-to-FFN route",
            )
        if self.extra_info.micro_batch_num != 1:
            raise ValueError(
                "WindowAFDConnector stage one supports only micro_batch_num=1, "
                f"got {self.extra_info.micro_batch_num}",
            )
        if self.micro_batch_size > 512:
            raise ValueError(
                "WindowAFDConnector requires max_num_batched_tokens <= 512 "
                "for the current AttentionToFfn operator, "
                f"got {self.micro_batch_size}",
            )
        routed_topk = self.selected_expert_num - self.shared_expert_num
        if routed_topk > 16:
            raise ValueError(
                "WindowAFDConnector requires num_experts_per_tok <= 16, "
                f"got {routed_topk}",
            )
        if self.shared_expert_num != 1:
            raise ValueError(
                "WindowAFDConnector currently supports exactly one shared expert, "
                f"got {self.shared_expert_num}",
            )
        routed_ffn_size = self.ffn_size - self.shared_expert_num
        if routed_ffn_size <= 0:
            raise ValueError(
                "WindowAFDConnector requires at least one routed-expert FFN rank"
            )

        (
            self.attn_window_size,
            self.ffn_window_size,
            a2f_token_size,
            f2a_token_size,
        ) = _window_sizes(
            attention_size=self.attn_size,
            micro_batch_num=self.extra_info.micro_batch_num,
            micro_batch_size=self.micro_batch_size,
            selected_expert_num=self.selected_expert_num,
            hidden_size=self.hidden_size,
            quant_mode=self.extra_info.quant_mode,
        )

        timeout = timedelta(minutes=30)
        try:
            self.process_group = init_afd_process_group(
                backend="hccl",
                init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
                world_size=self.world_size,
                rank=self.world_rank,
                group_name="afd_window",
                timeout=timeout,
            )
            backend = self.process_group._get_backend(torch.device("npu"))
            getter = getattr(backend, "get_hccl_comm_name", None)
            if getter is None:
                getter = getattr(self.process_group, "get_hccl_comm_name", None)
            if getter is None:
                raise RuntimeError("HCCL ProcessGroup does not expose comm name API")
            self.hccl_comm_name = str(getter(self.world_rank))

            self.window_size = (
                self.attn_window_size
                if self.afd_config.role == "attention"
                else self.ffn_window_size
            )
            self.window_tensor = torch.zeros(
                self.window_size,
                dtype=torch.uint8,
                device=torch.device("npu"),
            )
            self.window_addr = int(self.window_tensor.data_ptr())

            ffn_info, ffn_data, attn_info, attn_data = self._operator_shapes()
            if self.afd_config.role == "attention":
                self.comm_buffer = cot.get_buffer_for_attention_to_ffn(
                    self.process_group,
                    self.world_size,
                    ffn_info,
                    ffn_data,
                    quant_mode=self.extra_info.quant_mode,
                    window_addr=self.window_addr,
                    window_size=self.window_size,
                )
            else:
                self.comm_buffer = cot.get_buffer_for_ffn_to_attention(
                    self.process_group,
                    self.world_size,
                    attn_info,
                    attn_data,
                    window_addr=self.window_addr,
                    window_size=self.window_size,
                )

            context_factory = torch_npu._afd.create_schedule_context_holder
            kwargs = {
                "schedule_mode": 1 if self.afd_config.role == "attention" else 0,
                "session_num": self.attn_size,
                "micro_batch_num": self.extra_info.micro_batch_num,
                "micro_batch_size": self.micro_batch_size,
                "selected_expert_num": self.selected_expert_num,
                "expert_num": self.expert_num,
                "attn_to_ffn_token_size": a2f_token_size,
                "ffn_to_attn_token_size": f2a_token_size,
            }
            if self.afd_config.role == "attention":
                kwargs.update(
                    attention_window=self.window_addr,
                    attention_window_size=self.window_size,
                )
            else:
                kwargs.update(
                    ffn_window=self.window_addr,
                    ffn_window_size=self.window_size,
                )
            self.context_holder = context_factory(**kwargs)
            self.schedule_context = self.context_holder.get_schedule_context_tensor()
            self._build_rank_tables()
            ffn_kind = ""
            if self.afd_config.role == "ffn":
                ffn_kind = build_window_expert_layout(
                    routed_expert_num=self.routed_expert_num,
                    ffn_size=self.ffn_size,
                    ffn_rank=self.role_rank,
                ).kind
            print(
                "[Window][init] "
                f"role={self.afd_config.role} "
                f"role_rank={self.role_rank} "
                f"world_rank={self.world_rank} "
                f"attn_size={self.attn_size} "
                f"ffn_size={self.ffn_size} "
                f"micro_batch_num={self.extra_info.micro_batch_num} "
                f"micro_batch_size={self.micro_batch_size} "
                f"selected_expert_num={self.selected_expert_num} "
                f"expert_num={self.expert_num} "
                f"local_expert_num={self.local_expert_num} "
                f"ffn_kind={ffn_kind} "
                f"window_size={self.window_size}",
                flush=True,
            )
            self._initialized = True
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._pending_transfers.clear()
        holder = self.context_holder
        self.context_holder = None
        self.schedule_context = None
        if holder is not None:
            try:
                holder.stop_schedule()
            except Exception:
                pass
        comm_buffer = self.comm_buffer
        self.comm_buffer = None
        if comm_buffer is not None:
            try:
                comm_buffer.destroy()
            except Exception:
                pass
        self.window_tensor = None
        self.window_size = 0
        self.window_addr = 0
        self.attn_window_size = 0
        self.ffn_window_size = 0
        group = self.process_group
        self.process_group = None
        if group is not None:
            try:
                dist.destroy_process_group(group)
            except Exception:
                pass
        self.hccl_comm_name = None
        self._initialized = False

    def _build_rank_tables(self) -> None:
        """Build a balanced routed table plus one shared-first FFN rank."""
        if self.schedule_context is None:
            raise RuntimeError("ScheduleContext must be created before rank tables")
        device = self.schedule_context.device
        table = torch.zeros(
            (1, self.expert_num, 3),
            dtype=torch.int32,
            device=device,
        )
        for ffn_rank in range(self.ffn_size):
            layout = build_window_expert_layout(
                routed_expert_num=self.routed_expert_num,
                ffn_size=self.ffn_size,
                ffn_rank=ffn_rank,
            )
            if layout.is_shared:
                table[0, self.routed_expert_num, 0] = 1
                table[0, self.routed_expert_num, 1] = ffn_rank
                table[0, self.routed_expert_num, 2] = 0
                continue
            for local_id in range(layout.local_expert_count):
                expert_id = layout.local_expert_start + local_id
                table[0, expert_id, 0] = 1
                table[0, expert_id, 1] = ffn_rank
                table[0, expert_id, 2] = local_id
        self.expert_rank_table = table
        if self.afd_config.role == "ffn":
            self.local_expert_num = build_window_expert_layout(
                routed_expert_num=self.routed_expert_num,
                ffn_size=self.ffn_size,
                ffn_rank=self.role_rank,
            ).local_expert_count
        else:
            self.local_expert_num = 0
        self.attn_rank_table = (
            torch.arange(
                self.attn_size,
                dtype=torch.int32,
                device=device,
            )
            + self.ffn_size
        )

    def _operator_shapes(self) -> tuple[list[int], list[int], list[int], list[int]]:
        batch_size = self.micro_batch_size
        quant_mode = self.extra_info.quant_mode
        # This is the last dimension of one A2F token record in the FFN
        # Window, not its byte size in every mode.  For H=7168 it is 7168
        # fp16/bfloat16 elements in mode 0, or 7680 one-byte storage elements
        # in mode 2: align(7168 INT8 bytes + 4 scale bytes, 512) = 7680.
        a2f_token_data_dim = (
            _align_up(self.hidden_size + 4, 512)
            if quant_mode == 2
            else self.hidden_size
        )
        ffn_info = [self.attn_size, 1, 2 + batch_size * self.selected_expert_num]
        ffn_data = [
            self.attn_size,
            1,
            batch_size,
            self.selected_expert_num,
            a2f_token_data_dim,
        ]
        attn_info = [1, batch_size, self.selected_expert_num]
        attn_data = [1, batch_size, self.selected_expert_num, self.hidden_size]
        return ffn_info, ffn_data, attn_info, attn_data

    def _token_dtype(self) -> int:
        if self.extra_info.quant_mode == 2:
            return 2
        return 1 if self.vllm_config.model_config.dtype == torch.bfloat16 else 0

    @staticmethod
    def _token_dtype_for_tensor(tensor: torch.Tensor) -> int:
        if tensor.dtype == torch.bfloat16:
            return 1
        if tensor.dtype == torch.float16:
            return 0
        raise RuntimeError(
            "Window combine requires float16 or bfloat16 reference tensor, "
            f"got {tensor.dtype}",
        )

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        self._require_data_path()
        expert_ids = kwargs.get("expert_ids")
        expert_scales = kwargs.get("expert_scales")
        if expert_ids is None or expert_scales is None:
            raise RuntimeError("Window A2F requires expert_ids and expert_scales")
        batch_size = int(hidden_states.shape[0])
        if batch_size <= 0:
            raise RuntimeError("Window A2F requires at least one token")
        if batch_size > self.micro_batch_size:
            raise RuntimeError(
                "Window A2F batch exceeds the configured capacity: "
                f"batch={batch_size} capacity={self.micro_batch_size}",
            )
        expert_ids = expert_ids.to(torch.int32).reshape(batch_size, -1)
        expert_scales = expert_scales.to(torch.float32).reshape(batch_size, -1)

        # A2F receives only routed top-k IDs. The shared-expert slot is
        # represented by selected_expert_num (K + shared) in the Window
        # layout and rank table, not by an extra expert_ids column.
        routed_topk = self.selected_expert_num - self.shared_expert_num
        if expert_ids.shape[1] != routed_topk:
            raise RuntimeError(
                "Window A2F received an unexpected routed expert ID width: "
                f"got {expert_ids.shape[1]}, expected {routed_topk}",
            )
        if expert_scales.shape[1] != routed_topk:
            raise RuntimeError(
                "Window A2F received an unexpected routed expert scale width: "
                f"got {expert_scales.shape[1]}, expected {routed_topk}",
            )
        # The synchronous Window operators reuse one ScheduleContext and
        # therefore run with the fixed capacity shape used by the ref path.
        # Repeat valid inputs into padding slots so dynamic quantization never
        # receives artificial all-zero rows; padded results are discarded.
        repeat_indices = torch.arange(
            self.micro_batch_size,
            dtype=torch.long,
            device=hidden_states.device,
        ) % batch_size
        x = hidden_states[repeat_indices].reshape(
            1,
            self.micro_batch_size,
            self.hidden_size,
        )
        padded_expert_ids = expert_ids[repeat_indices].reshape(
            1,
            self.micro_batch_size,
            routed_topk,
        )
        active_mask = torch.ones(
            (1, self.micro_batch_size),
            dtype=torch.bool,
            device=hidden_states.device,
        )
        combine_scales = expert_scales.new_zeros(
            (self.micro_batch_size, routed_topk),
        )
        combine_scales[:batch_size].copy_(expert_scales)
        _, _, attn_info, _ = self._operator_shapes()
        session_id = torch.tensor([self.role_rank], dtype=torch.int32, device=x.device)
        micro_batch_id = torch.tensor(
            [int(kwargs.get("micro_batch_id", 0))],
            dtype=torch.int32,
            device=x.device,
        )
        model_layer_idx = int(context.metadata.layer_idx)
        if model_layer_idx < 0 or model_layer_idx >= self.num_layers:
            raise RuntimeError(
                "Window A2F received an out-of-range model layer: "
                f"layer={model_layer_idx} num_layers={self.num_layers}"
            )
        layer_id = torch.tensor(
            [model_layer_idx if self.async_mode else 0],
            dtype=torch.int32,
            device=x.device,
        )
        cot.attention_to_ffn(
            self.comm_buffer,
            x,
            session_id,
            micro_batch_id,
            layer_id,
            padded_expert_ids,
            self.expert_rank_table,
            attn_info,
            self.routed_expert_num,
            # The Ascend 950 AttentionToFfn V2 contract uses 0 for async and
            # 1 for wait-all sync.  FfnWorkerBatching V2 uses the opposite
            # value convention for its own sync_flag attribute.
            sync_flag=0 if self.async_mode else 1,
            ffn_start_rank_id=0,
            active_mask=active_mask,
        )
        logger.debug(
            "Window A2F sent layer=%d stage=%d batch=%d topk=%d",
            context.metadata.layer_idx,
            context.metadata.stage_idx,
            batch_size,
            expert_ids.shape[-1],
        )
        transfer_key = (
            int(context.metadata.stage_idx),
            int(context.metadata.layer_idx),
        )
        self._pending_transfers[transfer_key] = context
        state = WindowAFDTransferState(expert_scales=combine_scales)
        context.states = state

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._require_data_path()
        key = (int(ubatch_idx), int(kwargs.get("layer_idx", 0)))
        context = self._pending_transfers.pop(key, None)
        if context is None or not isinstance(context.states, WindowAFDTransferState):
            raise RuntimeError(f"Window F2A has no pending transfer for {key}")
        output, _ = torch_npu.npu_attention_worker_combine(
            self.schedule_context,
            context.states.expert_scales,
            torch.tensor([key[1]], dtype=torch.int32, device=ref_tensor.device),
            self.hidden_size,
            # ``token_dtype=2`` is only the INT8 payload mode of
            # ``ffn_worker_batching``.  ``attention_worker_combine`` accepts
            # only the output dtype modes: 0=FP16 and 1=BF16.  Use the same
            # dtype as the Attention continuation/residual, as P2P does.
            token_dtype=self._token_dtype_for_tensor(ref_tensor),
            need_schedule=1,
        )
        return output[: ref_tensor.shape[0]].reshape_as(ref_tensor)

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        self._require_data_path()
        batch_size = self.micro_batch_size
        # The operator expects the logical dimensions [A, BS, K+1, H].
        # Its tiling validates K+1 independently (currently <= 64); the
        # product A*BS*(K+1) is computed internally for the output rows.
        max_out_shape = [
            self.attn_size,
            batch_size,
            self.selected_expert_num,
            self.hidden_size,
        ]
        batching_expert_num = self.local_expert_num * (
            self.num_layers if self.async_mode else 1
        )
        if batching_expert_num > 8192:
            raise RuntimeError(
                "Window batching expert count exceeds the operator limit: "
                f"expert_num={batching_expert_num}"
            )
        if self.async_mode:
            outputs = cot.ffn_worker_batching(
                self.schedule_context,
                batching_expert_num,
                max_out_shape,
                token_dtype=self._token_dtype(),
                need_schedule=1,
                layer_num=self.num_layers,
                sync_flag=1,
            )
        else:
            outputs = torch_npu.npu_ffn_worker_batching(
                self.schedule_context,
                batching_expert_num,
                max_out_shape,
                token_dtype=self._token_dtype(),
                need_schedule=1,
                layer_num=0,
            )
        (
            hidden_states,
            group_list,
            session_ids,
            micro_batch_ids,
            token_ids,
            expert_offsets,
            dynamic_scale,
            actual_token_num,
        ) = outputs
        if actual_token_num.numel() != 1:
            raise RuntimeError(
                "Window batching returned actual_token_num with unexpected "
                f"shape {tuple(actual_token_num.shape)}",
            )
        actual_num = int(actual_token_num.item())
        if actual_num < 0 or actual_num > hidden_states.shape[0]:
            raise RuntimeError(
                "Window batching returned invalid actual_token_num: "
                f"actual={actual_num} capacity={hidden_states.shape[0]}",
            )
        if group_list.shape != (batching_expert_num, 2):
            raise RuntimeError(
                "Window batching returned group_list with unexpected shape: "
                f"got={tuple(group_list.shape)} "
                f"expected={(batching_expert_num, 2)}",
            )
        # On A3 the batching kernel writes a compact type-2 group list followed
        # by one [0, 0] sentinel, but does not clear the rest of the fixed-size
        # output.  Locate the valid prefix using actual_token_num, then convert
        # it to the dense cumulative type-0 form consumed by the native P2P
        # W8A8 MoE MLP path.  This also discards the stale fixed-buffer suffix.
        if actual_num == 0:
            group_list = torch.zeros(
                (batching_expert_num,),
                dtype=group_list.dtype,
                device=group_list.device,
            )
        else:
            group_counts = group_list[:, 1]
            cumulative_counts = torch.cumsum(group_counts, dim=0)
            prefix_ends = torch.nonzero(
                cumulative_counts == actual_num,
                as_tuple=False,
            ).flatten()
            if prefix_ends.numel() == 0:
                raise RuntimeError(
                    "Window batching group_list has no valid prefix matching "
                    f"actual_token_num={actual_num}",
                )
            valid_row_num = int(prefix_ends[0].item()) + 1
            if bool(torch.any(group_counts[:valid_row_num] <= 0).item()):
                raise RuntimeError(
                    "Window batching valid group_list prefix contains a "
                    "non-positive expert token count",
                )
            valid_expert_ids = group_list[:valid_row_num, 0]
            if bool(
                torch.any(
                    (valid_expert_ids < 0)
                    | (valid_expert_ids >= batching_expert_num)
                ).item()
            ):
                raise RuntimeError(
                    "Window batching valid group_list prefix contains an "
                    "out-of-range local expert ID",
                )
            if valid_row_num > 1 and bool(
                torch.any(valid_expert_ids[1:] <= valid_expert_ids[:-1]).item()
            ):
                raise RuntimeError(
                    "Window batching valid group_list expert IDs are not "
                    "strictly increasing",
                )
            expert_counts = torch.zeros(
                (batching_expert_num,),
                dtype=group_list.dtype,
                device=group_list.device,
            )
            expert_counts.scatter_(
                0,
                valid_expert_ids.to(torch.long),
                group_counts[:valid_row_num],
            )
            group_list = torch.cumsum(expert_counts, dim=0)

        group_sum = int(group_list[-1].item()) if group_list.numel() else 0
        if group_sum != actual_num:
            raise RuntimeError(
                "Window batching cumulative group_list does not match "
                f"actual_token_num: group_sum={group_sum} actual={actual_num}",
            )
        layer_batches: tuple[WindowLayerBatch, ...] = ()
        if self.async_mode and actual_num > 0:
            experts_per_layer = self.local_expert_num
            layer_ends = group_list[experts_per_layer - 1 :: experts_per_layer]
            layer_starts = torch.cat((layer_ends.new_zeros(1), layer_ends[:-1]))
            layer_bounds = (
                torch.stack((layer_starts, layer_ends), dim=1).cpu().tolist()
            )
            batches = []
            for layer_idx, (token_start, token_end) in enumerate(layer_bounds):
                if token_start == token_end:
                    continue
                expert_start = layer_idx * experts_per_layer
                expert_end = expert_start + experts_per_layer
                batches.append(
                    WindowLayerBatch(
                        layer_idx=layer_idx,
                        token_start=int(token_start),
                        token_end=int(token_end),
                        group_list=group_list[expert_start:expert_end] - token_start,
                    )
                )
            layer_batches = tuple(batches)
            covered_token_num = sum(
                batch.token_end - batch.token_start for batch in layer_batches
            )
            if covered_token_num != actual_num:
                raise RuntimeError(
                    "Window batching layer slices do not cover actual_token_num: "
                    f"layers={layer_batches} actual={actual_num}"
                )
        logger.debug(
            "Window FFN batching completed layer=%d stage=%d",
            int(kwargs.get("layer_idx", 0)),
            ubatch_idx,
        )
        context = AFDTransferContext(
            metadata=AFDTransferMetadata.create_ffn_metadata(
                layer_idx=int(kwargs.get("layer_idx", 0)),
                stage_idx=int(ubatch_idx),
                seq_lens=[int(hidden_states.shape[0])],
            ),
            states=WindowAFDTransferState(
                expert_scales=torch.empty(
                    (0,),
                    dtype=torch.float32,
                    device=hidden_states.device,
                ),
                group_list=group_list,
                dynamic_scale=dynamic_scale,
                session_ids=session_ids,
                micro_batch_ids=micro_batch_ids,
                token_ids=token_ids,
                expert_offsets=expert_offsets,
                actual_token_num=actual_token_num,
                layer_batches=layer_batches,
            ),
        )
        # Keep the static batching capacity Y.  The cumulative group_list and
        # actual_token_num describe the valid prefix consumed by grouped
        # matmul and F2A.  In particular, dynamic_scale must retain the same Y
        # as hidden_states for token-wise dynamic dequantization.
        return AFDA2FTransferPayload(
            hidden_states=hidden_states,
            context=context,
        )

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        self._require_data_path()
        if not isinstance(context.states, WindowAFDTransferState):
            raise RuntimeError("Window F2A requires batching state")
        state = context.states
        if any(
            value is None
            for value in (
                state.session_ids,
                state.micro_batch_ids,
                state.token_ids,
                state.expert_offsets,
                state.actual_token_num,
            )
        ):
            raise RuntimeError(
                "Window batching did not return complete routing metadata"
            )

        # FFNWorkerBatching returns fixed-capacity tensors.  actual_token_num
        # identifies their valid prefix; FfnToAttention consumes the same
        # capacity Y and ignores the suffix after that prefix.
        actual_token_num = state.actual_token_num.reshape(-1)
        if actual_token_num.numel() != 1:
            raise RuntimeError(
                "Window batching returned actual_token_num with unexpected "
                f"shape {tuple(state.actual_token_num.shape)}"
            )
        actual_num = int(actual_token_num.item())
        if actual_num < 0:
            raise RuntimeError(
                f"Window batching returned negative actual_token_num={actual_num}"
            )

        routed_output = getattr(ffn_output, "routed_output", ffn_output)
        if routed_output.dim() != 2 or routed_output.shape[0] < actual_num:
            raise RuntimeError(
                "Window F2A output capacity is smaller than actual token count: "
                f"output_shape={tuple(routed_output.shape)} actual_num={actual_num}"
            )

        metadata = (
            state.session_ids,
            state.micro_batch_ids,
            state.token_ids,
            state.expert_offsets,
        )
        if any(
            value.dim() != 1 or value.shape[0] != routed_output.shape[0]
            for value in metadata
        ):
            raise RuntimeError(
                "Window F2A output and metadata capacities do not match: "
                f"output_shape={tuple(routed_output.shape)} metadata_shapes="
                f"{[tuple(value.shape) for value in metadata]}"
            )

        cot.ffn_to_attention(
            self.comm_buffer,
            routed_output,
            state.session_ids,
            state.micro_batch_ids,
            state.token_ids,
            state.expert_offsets,
            actual_token_num,
            attn_rank_table=self.attn_rank_table,
        )
        logger.debug(
            "Window F2A sent stage=%d actual_tokens=%s",
            context.metadata.stage_idx,
            state.actual_token_num,
        )

    def _require_data_path(self) -> None:
        if not self._initialized:
            raise RuntimeError("WindowAFDConnector data path is not initialized")

    def select_experts(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts

        return select_experts(**kwargs)


__all__ = ["WindowAFDConnector", "WindowAFDExtraInfo", "WindowAFDTransferState"]
