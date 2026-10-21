#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "$SCRIPT_DIR/common_env.sh"

PREFILL_PORT="${PREFILL_PORT:-8900}"
PREFILL_VISIBLE_DEVICES="${PREFILL_VISIBLE_DEVICES:-0,1,2,3}"
PREFILL_HCCL_BASE_PORT="${PREFILL_HCCL_BASE_PORT:-40000}"
PREFILL_KV_PORT="${PREFILL_KV_PORT:-30000}"
PREFILL_ENGINE_ID="${PREFILL_ENGINE_ID:-0}"

export ASCEND_RT_VISIBLE_DEVICES="$PREFILL_VISIBLE_DEVICES"
export HCCL_IF_BASE_PORT="$PREFILL_HCCL_BASE_PORT"
validate_visible_devices "$PREFILL_VISIBLE_DEVICES"
prepare_runtime

KV_TRANSFER_CONFIG='{
  "kv_connector": "MooncakeHybridConnector",
  "kv_role": "kv_producer",
  "kv_port": "'"$PREFILL_KV_PORT"'",
  "engine_id": "'"$PREFILL_ENGINE_ID"'",
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
  "enable_shared_expert_dp": true,
  "enable_dsa_cp": false
}'

CMD=(
  "$VLLM_BIN" serve "$MODEL_PATH"
  --host "$MODEL_HOST"
  --port "$PREFILL_PORT"
  --data-parallel-size "$DP_SIZE"
  --tensor-parallel-size "$TP_SIZE"
  --enable-expert-parallel
  --seed "$SEED"
  --served-model-name "$MODEL_NAME"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --no-disable-hybrid-kv-cache-manager
  --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 128}'
  --no-enable-prefix-caching
  --safetensors-load-strategy prefetch
  --trust-remote-code
  --block-size "$BLOCK_SIZE"
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v4
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --enforce-eager
  --additional-config "$ADDITIONAL_CONFIG"
  --kv-transfer-config "$KV_TRANSFER_CONFIG"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  print_command env "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES" "HCCL_IF_BASE_PORT=$HCCL_IF_BASE_PORT" "${CMD[@]}"
else
  run_command "${CMD[@]}"
fi
