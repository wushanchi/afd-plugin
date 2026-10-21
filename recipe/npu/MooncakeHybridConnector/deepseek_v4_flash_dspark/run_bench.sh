#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the DeepSeek-V4-Flash-DSpark model directory}"
: "${PROXY_IP:?Set PROXY_IP to the proxy host IP}"

VLLM_BIN="${VLLM_BIN:-vllm}"
PROXY_PORT="${PROXY_PORT:-8000}"
MODEL_NAME="${MODEL_NAME:-dsv4}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
RESULT_DIR="${RESULT_DIR:-./results}"
RESULT_FILENAME="${RESULT_FILENAME:-dsv4_pd_random.json}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-2000}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-48}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.0}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
BENCH_SEED="${BENCH_SEED:-${SEED:-1024}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

for name in NUM_PROMPTS MAX_CONCURRENCY RANDOM_INPUT_LEN RANDOM_OUTPUT_LEN MAX_MODEL_LEN PROXY_PORT; do
  value="${!name}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$name must be a positive integer" >&2
    exit 2
  }
done
(( PROXY_PORT <= 65535 )) || { echo "PROXY_PORT must be in 1..65535" >&2; exit 2; }
(( RANDOM_INPUT_LEN + RANDOM_OUTPUT_LEN <= MAX_MODEL_LEN )) || {
  echo "RANDOM_INPUT_LEN + RANDOM_OUTPUT_LEN exceeds MAX_MODEL_LEN; update service and benchmark settings together" >&2
  exit 2
}

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  mkdir -p "$RESULT_DIR"
fi

CMD=(
  "$VLLM_BIN" bench serve
  --backend openai
  --host "$PROXY_IP"
  --port "$PROXY_PORT"
  --endpoint /v1/completions
  --model "$MODEL_NAME"
  --tokenizer "$TOKENIZER_PATH"
  --tokenizer-mode deepseek_v4
  --dataset-name random
  --random-input-len "$RANDOM_INPUT_LEN"
  --random-output-len "$RANDOM_OUTPUT_LEN"
  --random-range-ratio "$RANDOM_RANGE_RATIO"
  --num-prompts "$NUM_PROMPTS"
  --request-rate "$REQUEST_RATE"
  --max-concurrency "$MAX_CONCURRENCY"
  --seed "$BENCH_SEED"
  --ignore-eos
  --percentile-metrics ttft,tpot,e2el,itl
  --metric-percentiles 50,90,99
  --save-result
  --save-detailed
  --result-dir "$RESULT_DIR"
  --result-filename "$RESULT_FILENAME"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  exec "${CMD[@]}"
fi
