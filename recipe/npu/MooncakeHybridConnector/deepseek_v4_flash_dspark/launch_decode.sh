#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "$SCRIPT_DIR/common_env.sh"

DECODE_PORT="${DECODE_PORT:-8901}"
DP_RPC_PORT="${DP_RPC_PORT:-13346}"
DECODE_VISIBLE_DEVICES="${DECODE_VISIBLE_DEVICES:-4,5,6,7}"
DECODE_HCCL_BASE_PORT="${DECODE_HCCL_BASE_PORT:-50000}"
DECODE_KV_PORT="${DECODE_KV_PORT:-30100}"
DECODE_ENGINE_ID="${DECODE_ENGINE_ID:-1}"

export ASCEND_RT_VISIBLE_DEVICES="$DECODE_VISIBLE_DEVICES"
export HCCL_IF_BASE_PORT="$DECODE_HCCL_BASE_PORT"
validate_visible_devices "$DECODE_VISIBLE_DEVICES"
prepare_runtime

KV_TRANSFER_CONFIG='{
  "kv_connector": "MooncakeHybridConnector",
  "kv_role": "kv_consumer",
  "kv_port": "'"$DECODE_KV_PORT"'",
  "engine_id": "'"$DECODE_ENGINE_ID"'",
  "kv_connector_extra_config": {
    "prefill": {"dp_size": '"$DP_SIZE"', "tp_size": '"$TP_SIZE"'},
    "decode": {"dp_size": '"$DP_SIZE"', "tp_size": '"$TP_SIZE"'},
    "ascend_local_comm_res_path": "'"$HIXL_COMM_RES_PATH"'"
  }
}'

ADDITIONAL_CONFIG='{
  "ascend_compilation_config": {
    "enable_npugraph_ex": false,
    "enable_static_kernel": false
  },
  "enable_cpu_binding": true,
  "multistream_overlap_shared_expert": true,
  "recompute_scheduler_enable": true
}'

COMPILATION_CONFIG='{"cudagraph_mode": "FULL_DECODE_ONLY"}'

CMD=(
  "$VLLM_BIN" serve "$MODEL_PATH"
  --host "$MODEL_HOST"
  --port "$DECODE_PORT"
  --data-parallel-size "$DP_SIZE"
  --data-parallel-rpc-port "$DP_RPC_PORT"
  --tensor-parallel-size "$TP_SIZE"
  --enable-expert-parallel
  --seed "$SEED"
  --served-model-name "$MODEL_NAME"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --async-scheduling
  --block-size "$BLOCK_SIZE"
  --no-disable-hybrid-kv-cache-manager
  --no-enable-prefix-caching
  --safetensors-load-strategy prefetch
  --trust-remote-code
  --tokenizer-mode deepseek_v4
  --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 128}'
  --tool-call-parser deepseek_v4
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v4
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --compilation-config "$COMPILATION_CONFIG"
  --kv-transfer-config "$KV_TRANSFER_CONFIG"
  --additional-config "$ADDITIONAL_CONFIG"
)

if [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
  CMD+=(--speculative-config "$SPECULATIVE_CONFIG")
fi

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  print_command env "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES" "HCCL_IF_BASE_PORT=$HCCL_IF_BASE_PORT" "${CMD[@]}"
else
  run_command "${CMD[@]}"
fi
