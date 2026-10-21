# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Plugin-owned NPU profiler helpers for AFD Ascend runners."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Final, Literal, Protocol

logger = logging.getLogger(__name__)

AFDNPUProfilerRole = Literal["attention", "ffn"]

_ENV_PREFIX: Final[dict[AFDNPUProfilerRole, str]] = {
    "attention": "AFD_NPU_ATTENTION_PROFILER",
    "ffn": "AFD_NPU_FFN_PROFILER",
}
_DEFAULT_DIR: Final[dict[AFDNPUProfilerRole, str]] = {
    "attention": "/tmp/profile/attn",
    "ffn": "/tmp/profile/ffn",
}
_DEFAULT_ACTIVE_STEPS: Final[dict[AFDNPUProfilerRole, int]] = {
    "attention": 10,
    "ffn": 20,
}
_DEFAULT_WAIT_STEPS: Final[int] = 2
_DEFAULT_WARMUP_STEPS: Final[int] = 1
_DEFAULT_REPEAT: Final[int] = 1
_DEFAULT_SKIP_FIRST_STEPS: Final[int] = 1500
# An unset rank filter must collect every rank.  AFD runs one profiler per
# worker process, so selecting rank 0 by default silently drops the other
# cards in a multi-card role.
_DEFAULT_PROFILE_RANKS: Final[frozenset[int] | None] = None
_VLLM_TORCH_PROFILER_DIR_ENV: Final[str] = "VLLM_TORCH_PROFILER_DIR"


@dataclass(frozen=True)
class AFDNPUProfilerConfig:
    enabled: bool
    wait: int
    warmup: int
    active: int
    repeat: int
    skip_first: int
    trace_dir: str
    with_stack: bool
    ranks: frozenset[int] | None


class AFDNPUProfiler(Protocol):
    def step(self) -> None: ...

    def stop(self) -> None: ...


def afd_npu_profiler_config(role: AFDNPUProfilerRole) -> AFDNPUProfilerConfig:
    """Read plugin-owned profiler settings for an AFD NPU runner role."""

    prefix = _ENV_PREFIX[role]
    config = AFDNPUProfilerConfig(
        enabled=_env_bool(f"{prefix}_ENABLE", default=False),
        wait=_env_int(f"{prefix}_WAIT", default=_DEFAULT_WAIT_STEPS),
        warmup=_env_int(f"{prefix}_WARMUP", default=_DEFAULT_WARMUP_STEPS),
        active=_env_int(f"{prefix}_ACTIVE", default=_DEFAULT_ACTIVE_STEPS[role]),
        repeat=_env_int(f"{prefix}_REPEAT", default=_DEFAULT_REPEAT),
        skip_first=_env_int(
            f"{prefix}_SKIP_FIRST",
            default=_DEFAULT_SKIP_FIRST_STEPS,
        ),
        trace_dir=_env_dir(f"{prefix}_DIR", default=_DEFAULT_DIR[role]),
        with_stack=_env_bool(f"{prefix}_WITH_STACK", default=False),
        ranks=_env_ranks(
            f"{prefix}_RANKS",
            default=_DEFAULT_PROFILE_RANKS,
        ),
    )
    _validate_schedule_config(role, config)
    return config


def create_afd_npu_profiler(
    role: AFDNPUProfilerRole,
    *,
    role_rank: int = 0,
) -> AFDNPUProfiler | None:
    """Create a torch-npu profiler for an enabled role-local rank."""

    config = afd_npu_profiler_config(role)
    if not config.enabled or (
        config.ranks is not None and int(role_rank) not in config.ranks
    ):
        return None

    import torch_npu

    # ### PATCH START: detailed NPU metrics
    # Level1 with PipeUtilization emits the pipeline-utilization counters used
    # by Trace View.  AiCoreNone only records the basic timeline and makes a
    # multi-card AFD comparison appear complete while omitting the key NPU
    # pipeline metrics.
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )
    # ### PATCH END: detailed NPU metrics
    logger.info(
        "AFD NPU %s profiler enabled for role rank %d. Traces will be saved "
        "to: %s; with_stack=%s",
        role,
        role_rank,
        config.trace_dir,
        config.with_stack,
    )
    profiler = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=config.wait,
            warmup=config.warmup,
            active=config.active,
            repeat=config.repeat,
            skip_first=config.skip_first,
        ),
        with_stack=config.with_stack,
        with_modules=config.with_stack,
        record_shapes=True,
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            config.trace_dir,
        ),
    )
    profiler.start()
    return profiler


def step_afd_npu_profiler(profiler: AFDNPUProfiler | None) -> None:
    if profiler is not None:
        profiler.step()


def stop_afd_npu_profiler(profiler: AFDNPUProfiler | None) -> None:
    if profiler is not None:
        # ### PATCH START: schedule finalization
        # Advance once beyond ACTIVE so torch-npu invokes the trace handler
        # before stop().  This is intentionally done only at shutdown, after
        # the runner has stopped producing work.
        profiler.step()
        # ### PATCH END: schedule finalization
        profiler.stop()


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _env_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _validate_schedule_config(
    role: AFDNPUProfilerRole,
    config: AFDNPUProfilerConfig,
) -> None:
    """Reject invalid schedules before torch-npu creates a profiler."""

    non_negative = {
        "WAIT": config.wait,
        "WARMUP": config.warmup,
        "REPEAT": config.repeat,
        "SKIP_FIRST": config.skip_first,
    }
    invalid = [name for name, value in non_negative.items() if value < 0]
    if config.active <= 0:
        invalid.append("ACTIVE")
    if invalid:
        prefix = _ENV_PREFIX[role]
        raise ValueError(
            f"{prefix}_{invalid[0]} must be non-negative"
            if invalid[0] != "ACTIVE"
            else f"{prefix}_ACTIVE must be greater than zero"
        )


def _env_dir(name: str, *, default: str) -> str:
    return os.getenv(name) or os.getenv(_VLLM_TORCH_PROFILER_DIR_ENV) or default


def _env_ranks(
    name: str,
    *,
    default: frozenset[int] | None,
) -> frozenset[int] | None:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized == "all":
        return None
    if not normalized:
        raise ValueError(f"{name} must be 'all' or comma-separated ranks")
    try:
        ranks = frozenset(int(rank.strip()) for rank in normalized.split(","))
    except ValueError as exc:
        raise ValueError(
            f"{name} must be 'all' or comma-separated non-negative ranks, got {value!r}"
        ) from exc
    if any(rank < 0 for rank in ranks):
        raise ValueError(f"{name} must contain non-negative ranks, got {value!r}")
    return ranks


__all__ = [
    "AFDNPUProfilerConfig",
    "AFDNPUProfiler",
    "afd_npu_profiler_config",
    "create_afd_npu_profiler",
    "step_afd_npu_profiler",
    "stop_afd_npu_profiler",
]
