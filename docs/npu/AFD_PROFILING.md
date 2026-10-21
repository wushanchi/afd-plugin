# AFD NPU profiling

AFD Attention and FFN workers each create one `torch_npu.profiler` instance.
The profiler step is an engine or connector transaction count; it is not an
HTTP request count and the two roles do not share a step counter.

## Configuration

Use the symmetric environment variable prefixes
`AFD_NPU_ATTENTION_PROFILER_*` and `AFD_NPU_FFN_PROFILER_*`:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `ENABLE` | `0` | Create and start the profiler |
| `DIR` | `/tmp/profile/attn` or `/tmp/profile/ffn` | TensorBoard trace directory |
| `RANKS` | `all` | `all` or a comma-separated list of role-local ranks |
| `WAIT` | `2` | Steps skipped before warmup |
| `WARMUP` | `1` | Warmup steps |
| `ACTIVE` | `10` / `20` | Recorded steps for Attention / FFN |
| `REPEAT` | `1` | Number of schedule cycles; `0` follows torch-npu semantics |
| `SKIP_FIRST` | `1500` | Steps skipped before the first cycle |
| `WITH_STACK` | `0` | Include Python stack and module metadata |

Unset `RANKS` means every worker in that role is collected. A rank list is
role-local: rank 0 on Attention is distinct from rank 0 on FFN. The runner
passes `connector.role_rank`, so it remains correct for FFN-first Window and
P2P topologies.

The experimental NPU configuration uses profiler Level 1 with
`AiCMetrics.PipeUtilization`, in addition to CPU/NPU activities and recorded
shapes. This is the detailed pipeline metric set used by Trace View. Enabling
stack/module recording increases runtime and trace size.

## Step boundaries and shutdown

Attention advances the schedule only inside a live `execute_model` call.
FFN advances it only after warmup or graph capture has completed. Startup
dummy runs and ACL graph capture therefore do not consume the serving window.
Window FFN remains connector-driven, so one step is one Window transaction;
it is not guaranteed to align one-for-one with an Attention engine step.

At shutdown the helper performs one final `profiler.step()` before
`profiler.stop()` so the active cycle can flush its trace handler. This flush
is performed once by each runner after the FFN connector loop has stopped.

For comparable A/F traces, record the runner logs and align transfer metadata
or request/session markers. Identical `WAIT`, `WARMUP`, and `ACTIVE` values do
not establish a shared wall-clock start time.
