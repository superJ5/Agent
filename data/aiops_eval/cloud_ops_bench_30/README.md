# Cloud-OpsBench 30-case subset

This directory contains a selected subset of 30 cases from
[LLM4Ops/Cloud-OpsBench](https://github.com/LLM4Ops/Cloud-OpsBench).

- Upstream commit: `03c415e5709297432282fbbfd499f1bca0f8c347`
- Downloaded: 2026-09-04
- Upstream license: MIT; see `LICENSE`
- System represented by the data: Online Boutique on Kubernetes

## Selection

| Ground-truth root cause | Cases |
|---|---:|
| `pod_cpu_overload` | 8 |
| `code_memory_leak` | 4 |
| `node_network_delay` | 4 |
| `node_network_packet_loss` | 4 |
| `containerd_unavailable` | 5 |
| `kubelet_unavailable` | 5 |
| **Total** | **30** |

## Files retained

Each case keeps the files useful for deterministic Agent evaluation:

- `metadata.json`: question, difficulty and ground-truth root cause.
- `tool_cache.json`: cached results returned by the benchmark diagnostic tools.
- `raw_data/alert.json`: alert and anomaly evidence when provided upstream.
- `raw_data/metrics.csv`: time-series metrics when provided upstream.

The very large `raw_data/logs.json`, Kubernetes state snapshots and application
source trees are intentionally omitted. One raw log file can exceed 20 MB, while
the retained `tool_cache.json` already contains the official deterministic tool
responses used for diagnosis replay.

## Important boundary

These files are original Kubernetes microservice benchmark cases. They have now
been converted into the compact `cases.jsonl` evidence format, but they have not
been converted to this project's live CLS and local Monitor MCP schemas. Their
presence must not be described as a completed 30-case evaluation until all cases
have actually been run and a valid result summary has been produced.

## Prepare and evaluate

Create the compact normalized input:

```bash
uv run python scripts/prepare_cloud_ops_bench_eval.py
```

Run one case first to confirm the model and output format:

```bash
uv run python scripts/evaluate_cloud_ops_bench.py --limit 1
```

Run all 30 cases:

```bash
uv run python scripts/evaluate_cloud_ops_bench.py
```

Outputs are written to `results.jsonl` and `results.summary.json`. This first
version evaluates evidence-to-diagnosis reasoning (root cause and fault object),
not MCP tool-selection accuracy. The benchmark evidence is replayed offline and
must not be described as live production telemetry.

Model/API failures are recorded as unfinished calls. Accuracy is calculated only
over completed diagnoses; if every call fails, accuracy is `null` rather than a
misleading `0`.

The command above is an LLM-only evidence-to-diagnosis baseline. It bypasses the
Planner, Executor, Replanner and tool selection, so it is not an Agent score.

To evaluate the actual Plan-Execute-Replan Agent with per-case replay tools:

```bash
# Smoke-test one complete Agent run
uv run python scripts/evaluate_cloud_ops_bench_agent.py --limit 1

# Evaluate all 30 cases
uv run python scripts/evaluate_cloud_ops_bench_agent.py
```

This produces `agent_results.jsonl` and `agent_results.summary.json`, including
the plan, executed steps and tool-call trace. The tools replay the official
cached observations; they do not modify a live Kubernetes cluster and do not
test the HTTP transport of this project's production MCP servers.
