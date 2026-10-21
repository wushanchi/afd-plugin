#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "$SCRIPT_DIR/common_env.sh"

export RESERVE_HCCL_PORTS="${RESERVE_HCCL_PORTS:-1}"
prepare_runtime
if [[ "$RESERVE_HCCL_PORTS" == 1 ]]; then
  echo "Preflight passed. Reserved HCCL ranges: $(hccl_port_ranges)"
else
  echo "Preflight passed. HCCL port reservation was skipped"
fi
echo "Model: $MODEL_PATH"
echo "NIC/IP: $NIC_NAME / $LOCAL_IP"
