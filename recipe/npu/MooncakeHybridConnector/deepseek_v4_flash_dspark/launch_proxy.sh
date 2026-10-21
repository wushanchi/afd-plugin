#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PROXY_SCRIPT="${PROXY_SCRIPT:-$SCRIPT_DIR/../../../../tools/proxy_server.py}"
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-8000}"
PREFILL_HOSTS="${PREFILL_HOSTS:-127.0.0.1}"
PREFILL_PORTS="${PREFILL_PORTS:-${PREFILL_PORT:-8900}}"
DECODE_HOSTS="${DECODE_HOSTS:-127.0.0.1}"
DECODE_PORTS="${DECODE_PORTS:-${DECODE_PORT:-8901}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

[[ -f "$PROXY_SCRIPT" ]] || {
  echo "Proxy script does not exist: $PROXY_SCRIPT" >&2
  exit 2
}

read -r -a PREFILL_HOST_ARGS <<< "$PREFILL_HOSTS"
read -r -a PREFILL_PORT_ARGS <<< "$PREFILL_PORTS"
read -r -a DECODE_HOST_ARGS <<< "$DECODE_HOSTS"
read -r -a DECODE_PORT_ARGS <<< "$DECODE_PORTS"

[[ ${#PREFILL_HOST_ARGS[@]} -eq ${#PREFILL_PORT_ARGS[@]} ]] || {
  echo "PREFILL_HOSTS and PREFILL_PORTS must have the same number of entries" >&2
  exit 2
}
[[ ${#DECODE_HOST_ARGS[@]} -eq ${#DECODE_PORT_ARGS[@]} ]] || {
  echo "DECODE_HOSTS and DECODE_PORTS must have the same number of entries" >&2
  exit 2
}

CMD=(
  "$PYTHON_BIN" "$PROXY_SCRIPT"
  --host "$PROXY_HOST"
  --port "$PROXY_PORT"
  --prefiller-hosts "${PREFILL_HOST_ARGS[@]}"
  --prefiller-ports "${PREFILL_PORT_ARGS[@]}"
  --decoder-hosts "${DECODE_HOST_ARGS[@]}"
  --decoder-ports "${DECODE_PORT_ARGS[@]}"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  print_command "${CMD[@]}"
else
  exec "${CMD[@]}"
fi
