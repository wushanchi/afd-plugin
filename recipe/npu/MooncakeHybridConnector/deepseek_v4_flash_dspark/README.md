# DeepSeek-V4-Flash-DSpark Mooncake PD baseline

This recipe normalizes the supplied Prefill/Decode/Proxy commands into three
role launchers, a benchmark client, and one preflight script. It is a **baseline PD deployment**;
it does not enable the AFD plugin or an `additional-config.afd` section.

The commands retain the supplied topology:

| Role | API port | Visible NPUs | DP/TP | HCCL base port | Mooncake role | KV port | Engine ID |
| --- | ---: | --- | --- | ---: | --- | ---: | ---: |
| Prefill | 8900 | `0,1,2,3` | 4/1 | 40000 | `kv_producer` | 30000 | 0 |
| Decode | 8901 | `4,5,6,7` | 4/1 | 50000 | `kv_consumer` | 30100 | 1 |
| Proxy | 8000 | — | — | — | — | — | — |

The Prefill and Decode processes must use the same model path, model name,
DP/TP sizes, NIC/IP, HIXL resource path, and Mooncake topology values. The
default values for ports and visible devices match the supplied command, while
machine-specific values are required environment variables.

## Runtime source

The command was supplied as a vLLM-Ascend baseline. The repository does not
pin the vLLM-Ascend commit or the `load_balance_proxy_server_example.py`
revision used by that command, so this recipe is a normalized launch reference,
not a claim of a current hardware validation result. Record the exact vLLM,
vLLM-Ascend, CANN, Mooncake, and model revisions before reporting results.

## Prerequisites

- Linux with the Ascend runtime, vLLM, vLLM-Ascend, and Mooncake installed.
- The model directory is readable by the launch user.
- `NIC_NAME` exists and carries `LOCAL_IP`.
- The launch user can update `net.ipv4.ip_local_reserved_ports`, or sets
  `RESERVE_HCCL_PORTS=0` after reserving the ranges through system setup.
- The repository copy of the load-balance proxy and its Python dependencies are
  available. Set `PROXY_SCRIPT` explicitly when using a pinned vLLM-Ascend copy
  instead.

Set the machine-specific variables once in each shell:

```bash
export MODEL_PATH=/home/models/DeepSeek-V4-Flash-DSpark
export NIC_NAME=enp34s0f1
export LOCAL_IP=7.246.125.33
export HIXL_COMM_RES_PATH=/etc/hixlep
```

`MODEL_NAME`, `DP_SIZE`, `TP_SIZE`, and the port/device variables are optional
and have defaults matching the baseline. Override them when deploying a
different topology; both role processes must receive the same `DP_SIZE` and
`TP_SIZE`, and each role's visible-device list must contain `DP_SIZE*TP_SIZE`
entries.

Port reservations follow `PREFILL_HCCL_BASE_PORT` and `DECODE_HCCL_BASE_PORT`,
with `HCCL_PORT_COUNT=32` ports per range. Proxy backend ports default to the
matching `PREFILL_PORT` and `DECODE_PORT` values. Share these values with all
affected roles when changing ports. `BLOCK_SIZE` defaults to 32 for both stages.

## Launch order

Run `preflight.sh` once before starting either server. It checks the model,
NIC/IP assignment, and HIXL resource directory, then appends the two HCCL
ranges only when they are absent, so repeated launches do not duplicate
entries:

```bash
cd /path/to/afd-plugin
bash recipe/npu/MooncakeHybridConnector/deepseek_v4_flash_dspark/preflight.sh
```

Start Prefill and Decode in separate shells. The scripts use `exec`, so each
shell is replaced by its vLLM server process:

```bash
bash recipe/npu/MooncakeHybridConnector/deepseek_v4_flash_dspark/launch_prefill.sh
bash recipe/npu/MooncakeHybridConnector/deepseek_v4_flash_dspark/launch_decode.sh
```

After both API servers are ready, start the proxy. The default is the checked-in
copy of the vLLM-Ascend load-balance proxy. To use an external pinned copy,
set `PROXY_SCRIPT=/path/to/load_balance_proxy_server_example.py`. Host and port lists are
space-separated so a multi-prefiller deployment can be expressed without
changing the script:

```bash
bash recipe/npu/MooncakeHybridConnector/deepseek_v4_flash_dspark/launch_proxy.sh
```

For a remote two-node deployment, set the proxy lists to routable addresses,
for example:

```bash
PREFILL_HOSTS=10.0.0.11 DECODE_HOSTS=10.0.0.12 \
bash recipe/npu/MooncakeHybridConnector/deepseek_v4_flash_dspark/launch_proxy.sh
```

`127.0.0.1` is appropriate only when the two vLLM servers and proxy share a
host. Check the backend readiness before sending traffic:

```bash
curl -fsS http://127.0.0.1:8900/health
curl -fsS http://127.0.0.1:8901/health
```

Use `DRY_RUN=1` to print the fully expanded role command without starting
vLLM. The model path and NIC checks still run, which catches typos before a
real launch:

```bash
DRY_RUN=1 RESERVE_HCCL_PORTS=0 \
bash recipe/npu/MooncakeHybridConnector/deepseek_v4_flash_dspark/launch_prefill.sh
```

## Benchmark

After both backends and the proxy are ready, run:

```bash
PROXY_IP=127.0.0.1 bash recipe/npu/MooncakeHybridConnector/deepseek_v4_flash_dspark/run_bench.sh
```

The default workload matches the supplied AFD comparison client: input 2000,
output 48, 64 prompts, concurrency 64, seed 1024, request rate `inf`. Override
`RANDOM_INPUT_LEN`, `RANDOM_OUTPUT_LEN`, `NUM_PROMPTS`, `MAX_CONCURRENCY`,
`REQUEST_RATE`, `BENCH_SEED`, `PROXY_PORT`, `RESULT_DIR`, and `RESULT_FILENAME`
as needed. Set the same `MAX_MODEL_LEN` on the client and servers (default
4096); the input plus output must fit this limit. Use a routable `PROXY_IP`
when the benchmark runs on another node.

## Stage-specific behavior preserved

- Prefill keeps eager execution and disables `npugraph_ex` and static kernels.
- Decode keeps `--async-scheduling` and `FULL_DECODE_ONLY` compilation mode.
- Both stages keep the supplied DeepSeek-V4 tokenizer, tool-call parser,
  reasoning parser, multithreaded model loading, and Mooncake connector JSON.
- The proxy is an independent process; it is no longer appended after a
  server command that uses `exec`.

The scripts intentionally do not add AFD connector settings, U2, async-DP, or
performance claims. Those are separate deployment choices and must be tested
with their own pinned recipe.
