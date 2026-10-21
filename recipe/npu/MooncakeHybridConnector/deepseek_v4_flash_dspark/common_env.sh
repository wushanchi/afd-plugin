#!/usr/bin/env bash
# Shared configuration and helpers for the DeepSeek-V4 Mooncake PD baseline.
# This file is meant to be sourced by the role launchers.

set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the DeepSeek-V4-Flash-DSpark model directory}"
: "${NIC_NAME:?Set NIC_NAME to the HCCL communication interface}"
: "${LOCAL_IP:?Set LOCAL_IP to the communication IP of this node}"

VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_NAME="${MODEL_NAME:-dsv4}"
MODEL_HOST="${MODEL_HOST:-0.0.0.0}"
DP_SIZE="${DP_SIZE:-4}"
TP_SIZE="${TP_SIZE:-1}"
SEED="${SEED:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-24}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
HIXL_COMM_RES_PATH="${HIXL_COMM_RES_PATH:-/etc/hixlep}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
PREFILL_PORT="${PREFILL_PORT:-8900}"
DECODE_PORT="${DECODE_PORT:-8901}"
DP_RPC_PORT="${DP_RPC_PORT:-13346}"
PREFILL_KV_PORT="${PREFILL_KV_PORT:-30000}"
DECODE_KV_PORT="${DECODE_KV_PORT:-30100}"
PREFILL_HCCL_BASE_PORT="${PREFILL_HCCL_BASE_PORT:-40000}"
DECODE_HCCL_BASE_PORT="${DECODE_HCCL_BASE_PORT:-50000}"
HCCL_PORT_COUNT="${HCCL_PORT_COUNT:-32}"

export_common_env() {
  export HCCL_IF_IP="$LOCAL_IP"
  export VLLM_HOST_IP="${VLLM_HOST_IP:-$LOCAL_IP}"
  export GLOO_SOCKET_IFNAME="$NIC_NAME"
  export TP_SOCKET_IFNAME="$NIC_NAME"
  export HCCL_SOCKET_IFNAME="$NIC_NAME"
  export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-3600000}"
  export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-30000}"
  export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-600}"
  export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"
  export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
  export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
  export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
  export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command not found: $1" >&2
    exit 127
  }
}

validate_model_path() {
  [[ -d "$MODEL_PATH" ]] || {
    echo "MODEL_PATH is not a directory: $MODEL_PATH" >&2
    exit 2
  }
}

validate_nic() {
  [[ -d "/sys/class/net/$NIC_NAME" ]] || {
    echo "NIC_NAME does not exist: $NIC_NAME" >&2
    exit 2
  }

  require_command ip
  ip -4 -o addr show dev "$NIC_NAME" | awk -v expected="$LOCAL_IP" '
    { split($4, address, "/"); if (address[1] == expected) found = 1 }
    END { exit !found }
  ' || {
    echo "LOCAL_IP is not assigned to NIC_NAME: $LOCAL_IP / $NIC_NAME" >&2
    exit 2
  }
}

validate_hixl_path() {
  [[ -d "$HIXL_COMM_RES_PATH" ]] || {
    echo "HIXL_COMM_RES_PATH is not a directory: $HIXL_COMM_RES_PATH" >&2
    exit 2
  }
}

validate_visible_devices() {
  local visible_devices="$1" expected_count device
  local -a devices
  local -A seen=()
  [[ "$DP_SIZE" =~ ^[1-9][0-9]*$ && "$TP_SIZE" =~ ^[1-9][0-9]*$ ]] || {
    echo "DP_SIZE and TP_SIZE must be positive integers" >&2
    exit 2
  }
  expected_count=$((DP_SIZE * TP_SIZE))
  [[ "$visible_devices" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || {
    echo "Visible devices must be comma-separated non-negative integers" >&2
    exit 2
  }
  IFS=',' read -r -a devices <<< "$visible_devices"
  for device in "${devices[@]}"; do
    [[ -z "${seen[$device]:-}" ]] || {
      echo "Duplicate visible device: $device" >&2
      exit 2
    }
    seen[$device]=1
  done
  [[ ${#devices[@]} -eq "$expected_count" ]] || {
    echo "Visible device count (${#devices[@]}) must equal DP_SIZE*TP_SIZE ($expected_count)" >&2
    exit 2
  }
}

validate_ports() {
  local name value
  for name in PREFILL_PORT DECODE_PORT DP_RPC_PORT PREFILL_KV_PORT DECODE_KV_PORT PREFILL_HCCL_BASE_PORT DECODE_HCCL_BASE_PORT; do
    value="${!name}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] && (( value <= 65535 )) || {
      echo "$name must be an integer between 1 and 65535" >&2
      exit 2
    }
  done
  [[ "$HCCL_PORT_COUNT" =~ ^[1-9][0-9]*$ ]] &&
    (( PREFILL_HCCL_BASE_PORT + HCCL_PORT_COUNT - 1 <= 65535 && DECODE_HCCL_BASE_PORT + HCCL_PORT_COUNT - 1 <= 65535 )) || {
    echo "HCCL_PORT_COUNT must be positive and HCCL port ranges must fit in 1..65535" >&2
    exit 2
  }
}

hccl_port_ranges() {
  printf '%s-%s %s-%s\n' "$PREFILL_HCCL_BASE_PORT" "$((PREFILL_HCCL_BASE_PORT + HCCL_PORT_COUNT - 1))" \
    "$DECODE_HCCL_BASE_PORT" "$((DECODE_HCCL_BASE_PORT + HCCL_PORT_COUNT - 1))"
}

reserve_hccl_ports() {
  local current range updated
  current="$(sysctl -n net.ipv4.ip_local_reserved_ports 2>/dev/null || true)"
  updated="$current"

  for range in $(hccl_port_ranges); do
    if [[ ",${updated}," != *",${range},"* ]]; then
      updated="${updated:+${updated},}${range}"
    fi
  done

  if [[ "$updated" != "$current" ]]; then
    sysctl -w "net.ipv4.ip_local_reserved_ports=$updated" >/dev/null
  fi
}

prepare_runtime() {
  validate_ports
  validate_model_path
  validate_nic
  validate_hixl_path
  export_common_env

  if [[ "${RESERVE_HCCL_PORTS:-0}" == 1 ]]; then
    require_command sysctl
    reserve_hccl_ports
  fi
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_command() {
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    print_command "$@"
    return 0
  fi
  exec "$@"
}
