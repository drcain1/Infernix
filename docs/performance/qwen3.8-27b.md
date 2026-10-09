# Qwen3.8-27B serving performance

[Performance index](../performance.md) · [Measurement and publication rules](methodology.md)

On this page: [run records](#scope-and-run-records), [context profile](#no-speculation-context-profile),
[single-request decode](#single-request-speculative-decode), [corpus makespan](#corpus-makespan),
[decode saturation](#decode-saturation), [completion outcomes](#completion-outcomes),
[reproduction](#reproduction-and-reports), [V100 comparison](#v100-comparison-with-ninfer).

## Scope and run records

Measured on September 28–29, 2026 (Asia/Shanghai), using one RTX 5090, NVIDIA driver 617.14,
CUDA 13.4 compile/runtime/driver API, and Infernix revision
`7f6aafedb5f20200def820cfe51ab81c09c20eeb`. The serving binary was fixed for the entire campaign.
All runs use FP8 E4M3 row-256 KV, a 1,024-token prefill chunk, CUDA Graphs, disabled prefix reuse,
and the [common stochastic sampling profile](methodology.md#common-serving-profile).
MTP3 and DFlash2 K=7 use the optimized proposal head. Startup and warmup are outside measurement.

The artifacts are `out/qwen3_8_27b.ninfer` (`groupwise-int`) and
`out/qwen3_8_27b_nvfp4.ninfer` (`nvfp4`), including their selected speculative companion weights.
All 20 configurations and 820 formal requests completed without request, CUDA, or allocation errors.
The following RTX 5090 tables contain this campaign's measurements only. The separate
[V100 comparison](#v100-comparison-with-ninfer) records its own hardware and workload.

| Run | Weights ID | Measurement | Context ceiling | C | Actual KV capacity (tokens, in C order) |
|---|---|---|---|---|---|
| G0 | `groupwise-int` | MTP0 | 262,144 | 1 | 262,144 |
| G3 | `groupwise-int` | MTP3 corpus | 131,072 | 1, 2, 4, 8 | 131,072, 262,144, 347,200, 318,592 |
| GD | `groupwise-int` | DFlash2 K=7 corpus | 131,072 | 1 | 131,072 |
| GS | `groupwise-int` | MTP3 saturation | 16,384 | 1, 2, 4, 8 | 16,384; 32,768; 65,536; 131,072 |
| N0 | `nvfp4` | MTP0 | 262,144 | 1 | 262,144 |
| N3 | `nvfp4` | MTP3 corpus | 131,072 | 1, 2, 4, 8 | 131,072, 262,144, 253,632, 225,024 |
| ND | `nvfp4` | DFlash2 K=7 corpus | 131,072 | 1 | 131,072 |
| NS | `nvfp4` | MTP3 saturation | 16,384 | 1, 2, 4, 8 | 16,384; 32,768; 65,536; 131,072 |

MTP0 reserves its KV capacity explicitly; corpus and saturation points use `--kv-capacity auto`.
The workspace arena is 243.3 MiB in every configuration. Resident weight arenas depend on the selected backend:

| Backend | groupwise-int weights (GiB) | nvfp4 weights (GiB) |
|---|---|---|
| No speculation | 15.920 | 18.976 |
| MTP3 | 16.672 | 19.729 |
| DFlash2 K=7 | 18.326 | 21.383 |

## No-speculation context profile

G0/N0 use the [single-request method](methodology.md#single-request-phases): five fixed seeds per fixture, thinking disabled, and a 128-token output budget.

### groupwise-int

| Prompt tokens | Samples | Prefill phase (tok/s) | Server TTFT (ms) | Decode phase (tok/s) |
|---|---|---|---|---|
| 7,680 | 5 | 3,331.9 ± 7.6 | 2,308.5 ± 5.2 | 84.1 ± 0.3 |
| 64,512 | 5 | 2,964.4 ± 3.6 | 21,798.1 ± 28.1 | 76.8 ± 0.1 |
| 130,048 | 5 | 2,623.8 ± 0.6 | 49,643.7 ± 10.3 | 69.7 ± 0.2 |
| 260,096 | 5 | 2,139.4 ± 0.4 | 121,723.1 ± 19.7 | 59.2 ± 0.1 |

### nvfp4

| Prompt tokens | Samples | Prefill phase (tok/s) | Server TTFT (ms) | Decode phase (tok/s) |
|---|---|---|---|---|
| 7,680 | 5 | 12,819.1 ± 16.8 | 602.7 ± 1.1 | 74.1 ± 0.3 |
| 64,512 | 5 | 8,658.2 ± 44.8 | 7,487.4 ± 37.7 | 68.2 ± 0.2 |
| 130,048 | 5 | 6,198.5 ± 19.6 | 21,055.7 ± 67.2 | 62.5 ± 0.3 |
| 260,096 | 5 | 4,016.4 ± 10.4 | 64,910.0 ± 171.8 | 53.4 ± 0.6 |

## Single-request speculative decode

These phase statistics come directly from the C=1 corpus points, G3/N3 and GD/ND.
Each reasoning fixture has five samples and a 65,536-token output budget; each other category
pools three fixtures × five seeds with a 4,096-token budget. Values are arithmetic mean ± sample
standard deviation of per-request rates and ratios. Category deviation also includes differences
between fixtures. Actual output lengths vary; [completion outcomes](#completion-outcomes) apply
to all the means below. In particular, all C=1 AIME 15 samples reach the output limit.

### groupwise-int

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode phase (tok/s) | MTP3 acceptance | MTP3 tokens/round |
|---|---|---|---|---|---|
| `long_decode_aime26_01` | 5 | 1,440.8 ± 290.3 | 196.1 ± 4.7 | 77.9% ± 2.6% | 3.34 ± 0.08 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 149.2 ± 1.3 | 54.9% ± 0.7% | 2.65 ± 0.02 |
| `long_decode_aime26_30` | 5 | 49,543.0 ± 9,464.5 | 155.1 ± 3.2 | 57.3% ± 2.0% | 2.72 ± 0.06 |

#### MTP3 cross-scenario decode

| Category | Samples | Decode phase (tok/s) | MTP3 acceptance | MTP3 tokens/round |
|---|---|---|---|---|
| Code | 15 | 193.5 ± 5.9 | 76.4% ± 3.3% | 3.29 ± 0.10 |
| Story | 15 | 123.4 ± 10.3 | 36.6% ± 5.9% | 2.10 ± 0.18 |
| Translation | 15 | 190.3 ± 10.6 | 74.6% ± 6.1% | 3.24 ± 0.18 |
| Structured | 15 | 214.7 ± 13.7 | 88.4% ± 7.8% | 3.65 ± 0.23 |

#### DFlash2 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode phase (tok/s) | DFlash2 acceptance | DFlash2 tokens/round |
|---|---|---|---|---|---|
| `long_decode_aime26_01` | 5 | 1,572.2 ± 585.1 | 305.7 ± 17.7 | 65.9% ± 4.7% | 5.61 ± 0.33 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 177.3 ± 4.8 | 33.9% ± 1.3% | 3.37 ± 0.09 |
| `long_decode_aime26_30` | 5 | 41,378.4 ± 7,618.2 | 189.0 ± 8.2 | 36.4% ± 2.3% | 3.55 ± 0.16 |

#### DFlash2 cross-scenario decode

| Category | Samples | Decode phase (tok/s) | DFlash2 acceptance | DFlash2 tokens/round |
|---|---|---|---|---|
| Code | 15 | 258.6 ± 24.6 | 53.6% ± 6.5% | 4.75 ± 0.45 |
| Story | 15 | 115.0 ± 28.7 | 15.9% ± 7.5% | 2.11 ± 0.53 |
| Translation | 15 | 241.6 ± 49.1 | 49.1% ± 12.9% | 4.44 ± 0.91 |
| Structured | 15 | 348.9 ± 51.4 | 77.3% ± 13.6% | 6.41 ± 0.95 |

### nvfp4

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode phase (tok/s) | MTP3 acceptance | MTP3 tokens/round |
|---|---|---|---|---|---|
| `long_decode_aime26_01` | 5 | 1,559.4 ± 727.2 | 207.4 ± 3.6 | 77.2% ± 1.8% | 3.32 ± 0.05 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 161.7 ± 3.8 | 57.2% ± 2.1% | 2.72 ± 0.06 |
| `long_decode_aime26_30` | 5 | 37,978.0 ± 7,474.4 | 170.0 ± 2.3 | 59.9% ± 1.5% | 2.80 ± 0.05 |

#### MTP3 cross-scenario decode

| Category | Samples | Decode phase (tok/s) | MTP3 acceptance | MTP3 tokens/round |
|---|---|---|---|---|
| Code | 15 | 205.3 ± 9.5 | 76.8% ± 5.1% | 3.30 ± 0.15 |
| Story | 15 | 132.6 ± 13.1 | 37.6% ± 7.1% | 2.13 ± 0.21 |
| Translation | 15 | 202.9 ± 12.3 | 75.2% ± 6.6% | 3.26 ± 0.20 |
| Structured | 15 | 231.7 ± 10.6 | 90.6% ± 5.6% | 3.72 ± 0.17 |

#### DFlash2 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode phase (tok/s) | DFlash2 acceptance | DFlash2 tokens/round |
|---|---|---|---|---|---|
| `long_decode_aime26_01` | 5 | 1,921.6 ± 947.3 | 323.6 ± 28.8 | 62.8% ± 6.9% | 5.40 ± 0.48 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 200.4 ± 13.9 | 35.3% ± 3.4% | 3.47 ± 0.24 |
| `long_decode_aime26_30` | 5 | 38,535.8 ± 3,520.6 | 214.9 ± 3.4 | 38.1% ± 0.8% | 3.66 ± 0.05 |

#### DFlash2 cross-scenario decode

| Category | Samples | Decode phase (tok/s) | DFlash2 acceptance | DFlash2 tokens/round |
|---|---|---|---|---|
| Code | 15 | 285.7 ± 25.2 | 53.9% ± 6.0% | 4.77 ± 0.42 |
| Story | 15 | 129.2 ± 33.8 | 16.5% ± 8.1% | 2.15 ± 0.56 |
| Translation | 15 | 264.0 ± 64.1 | 48.6% ± 15.3% | 4.40 ± 1.07 |
| Structured | 15 | 377.3 ± 57.7 | 75.7% ± 13.8% | 6.30 ± 0.96 |

## Corpus makespan

Each point is one uninterrupted run of the same 75-request corpus, using shuffle seed `20260811`
and ordered HTTP sends. Makespan includes prefill, decode, admission waits, transitions, and drain.
Corpus rates divide token totals by full makespan; acceptance is the ratio of summed accepted and
drafted tokens. Average batch covers the entire run. Stochastic continuations differ between points.

### MTP3 groupwise-int

| C | Requests | Computed prefill tokens | Decode tokens | Makespan (s) | Requests/s | Corpus prefill (tok/s) | Corpus decode (tok/s) | Avg batch | MTP acceptance |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 75 | 15,460 | 768,038 | 4,918.67 | 0.0152 | 3.1 | 156.1 | 1.00 | 58.0% |
| 2 | 75 | 15,460 | 728,617 | 2,676.67 | 0.0280 | 5.8 | 272.2 | 1.92 | 58.4% |
| 4 | 75 | 15,460 | 779,134 | 2,023.84 | 0.0371 | 7.6 | 385.0 | 3.47 | 59.0% |
| 8 | 75 | 15,460 | 755,705 | 1,719.23 | 0.0436 | 9.0 | 439.6 | 4.31 | 60.0% |

### MTP3 nvfp4

| C | Requests | Computed prefill tokens | Decode tokens | Makespan (s) | Requests/s | Corpus prefill (tok/s) | Corpus decode (tok/s) | Avg batch | MTP acceptance |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 75 | 15,460 | 693,701 | 4,115.22 | 0.0182 | 3.8 | 168.6 | 1.00 | 59.6% |
| 2 | 75 | 15,460 | 709,989 | 2,394.23 | 0.0313 | 6.5 | 296.5 | 1.90 | 58.9% |
| 4 | 75 | 15,460 | 722,202 | 1,640.17 | 0.0457 | 9.4 | 440.3 | 3.13 | 58.5% |
| 8 | 75 | 15,460 | 708,589 | 1,443.37 | 0.0520 | 10.7 | 490.9 | 3.52 | 59.5% |

At C=8, the full-corpus average batch is 4.31 for groupwise-int and 3.52 for NVFP4.
Their shared KV capacities are 318,592 and 225,024 tokens; the sampled maximum waiting counts
are four and five requests. NVFP4 C=4 also records one waiting request. These are measured admission
limits, and the full makespans retain their effect. No spill, owner degradation/eviction, or
search-exhaustion event was recorded.

### DFlash2 K=7

| Weights ID | Requests | Completion tokens | Decode tokens | Makespan (s) | Requests/s | Corpus decode (tok/s) | DFlash2 acceptance |
|---|---|---|---|---|---|---|---|
| `groupwise-int` | 75 | 725,080 | 725,005 | 3,896.89 | 0.0192 | 186.0 | 35.8% |
| `nvfp4` | 75 | 697,637 | 697,562 | 3,354.86 | 0.0224 | 207.9 | 36.6% |

Each DFlash2 point computes 15,460 prefill tokens, has average decode batch 1.00, and uses 131,072 KV tokens.

## Decode saturation

GS/NS use the [sustained-wave method](methodology.md#decode-saturation): one wave at each C,
`long_decode_aime26_15`, a 335-token prompt, thinking enabled, and an 8,192-token output budget
per request. All 30 requests reach that budget. Steady throughput selects complete full-batch
intervals; acceptance covers the entire wave, and wave makespan includes ramp-up and drain.

### groupwise-int

| C | Steady (s) | Avg batch | Steady decode (tok/s) | MTP acceptance (wave) | Wave makespan (s) |
|---|---|---|---|---|---|
| 1 | 58.00 | 1.00 | 136.5 | 44.4% | 59.89 |
| 2 | 63.00 | 2.00 | 253.3 | 45.2% | 65.25 |
| 4 | 80.00 | 4.00 | 398.1 | 46.1% | 83.32 |
| 8 | 108.00 | 8.00 | 582.4 | 46.4% | 115.39 |

### nvfp4

| C | Steady (s) | Avg batch | Steady decode (tok/s) | MTP acceptance (wave) | Wave makespan (s) |
|---|---|---|---|---|---|
| 1 | 54.00 | 1.00 | 147.7 | 46.2% | 55.34 |
| 2 | 54.00 | 2.00 | 291.0 | 48.7% | 56.52 |
| 4 | 60.00 | 4.00 | 522.2 | 45.8% | 63.11 |
| 8 | 69.00 | 8.00 | 922.4 | 46.1% | 72.00 |

## Completion outcomes

Token accounting and termination reasons were checked for every formal request. All 40 MTP0
requests terminate at a stop token. The table below gives stop-token / output-limit counts for
the C=1 speculative corpus; every sample remains in the published statistics.

| Workload | Samples per profile | Groupwise MTP3 | NVFP4 MTP3 | Groupwise DFlash2 | NVFP4 DFlash2 |
|---|---|---|---|---|---|
| `long_decode_aime26_01` | 5 | 5 / 0 | 5 / 0 | 5 / 0 | 5 / 0 |
| `long_decode_aime26_15` | 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 |
| `long_decode_aime26_30` | 5 | 5 / 0 | 5 / 0 | 5 / 0 | 5 / 0 |
| Code | 15 | 0 / 15 | 5 / 10 | 0 / 15 | 5 / 10 |
| Story | 15 | 10 / 5 | 10 / 5 | 9 / 6 | 11 / 4 |
| Translation | 15 | 15 / 0 | 15 / 0 | 15 / 0 | 15 / 0 |
| Structured | 15 | 1 / 14 | 0 / 15 | 3 / 12 | 1 / 14 |
| **Total** | **75** | 36 / 39 | 40 / 35 | 37 / 38 | 42 / 33 |

Across MTP3 C=1/2/4/8, groupwise-int stop/limit counts are 36/39, 35/40, 36/39, and 37/38;
NVFP4 counts are 40/35, 41/34, 40/35, and 42/33. Output-limit samples measure generation through
the configured budget and do not establish completed reasoning or task success.

Screening the complete C=1 responses for obvious large verbatim repetitions found no candidates.
Concurrent C>1 runs retain token and termination records, but no complete-response repetition audit
was performed. Answer accuracy, prompt compliance, and perplexity were not measured in this campaign.

## Reproduction and reports

Reports are local under `profiles/bench/qwen3_8_27b_fp8kv_20260928/`.
`campaign.json` records the build, artifact identities, commands, and execution status.
`run.jsonl`, `points/*.json`, `server/*.jsonl`, and the existing summaries retain the measured evidence;
C=1 corpus responses and phase summaries are in `corpus/<point>/`.

| Runs | Report directories relative to campaign root |
|---|---|
| G0 / N0 | `groupwise-int/mtp0/`, `nvfp4/mtp0/` |
| G3 / N3 | `groupwise-int/mtp3-corpus/c{1,2,4,8}/`, `nvfp4/mtp3-corpus/c{1,2,4,8}/` |
| GD / ND | `groupwise-int/dflash2-corpus/`, `nvfp4/dflash2-corpus/` |
| GS / NS | `groupwise-int/mtp3-saturation/c{1,2,4,8}/`, `nvfp4/mtp3-saturation/c{1,2,4,8}/` |

Build `infernix-serve`, then run from the repository root with Python 3.11 and a fresh output directory:

```bash
export INFERNIX_BENCH_PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
export INFERNIX_PERF_OUTPUT=profiles/bench/qwen3_8_27b_fp8kv_rerun

for perf_profile in nvfp4 groupwise-int; do
  case "$perf_profile" in
    nvfp4) perf_artifact=out/qwen3_8_27b_nvfp4.ninfer ;;
    groupwise-int) perf_artifact=out/qwen3_8_27b.ninfer ;;
  esac
  perf_common=(--serve build/apps/infernix-serve
    --artifact "qwen3_8_27b=$perf_artifact"
    --kv-dtype fp8 --sampling stochastic --port 18080)

  "$INFERNIX_BENCH_PYTHON" tools/bench/run_serve_corpus.py "${perf_common[@]}" \
    --mode mtp0 --output "$INFERNIX_PERF_OUTPUT/$perf_profile/mtp0"

  for perf_c in 1 2 4 8; do
    "$INFERNIX_BENCH_PYTHON" tools/bench/run_serve_concurrency.py "${perf_common[@]}" \
      --mode mtp3 --suite decode-saturation --concurrency "$perf_c" \
      --decode-tokens 8192 --max-context 16384 --kv-capacity auto --prefill-chunk 1024 \
      --output "$INFERNIX_PERF_OUTPUT/$perf_profile/mtp3-saturation/c$perf_c"
  done

  for perf_c in 1 2 4 8; do
    "$INFERNIX_BENCH_PYTHON" tools/bench/run_serve_concurrency.py "${perf_common[@]}" \
      --mode mtp3 --suite corpus-makespan --concurrency "$perf_c" \
      --max-context 131072 --kv-capacity auto --prefill-chunk 1024 \
      --output "$INFERNIX_PERF_OUTPUT/$perf_profile/mtp3-corpus/c$perf_c"
  done

  "$INFERNIX_BENCH_PYTHON" tools/bench/run_serve_concurrency.py "${perf_common[@]}" \
    --mode dflash2_7 --suite corpus-makespan --concurrency 1 \
    --max-context 131072 --kv-capacity auto --prefill-chunk 1024 \
    --output "$INFERNIX_PERF_OUTPUT/$perf_profile/dflash2-corpus"
done
```

## V100 comparison with NInfer

This section preserves the initial qualified port comparison. The subsequent
[MTP optimization](#v100-mtp-activation-staging-optimization) raises measured mean decode
throughput to 63.84 tok/s, 20.27% above the same NInfer baseline.

Measured October 9, 2026 (Europe/Stockholm) on one Tesla V100-PCIE-32GB, 80 SMs,
ECC off, with the existing **170 W power limit**. Driver: 580.178.04. Infernix uses
CUDA 12.9.86 and GCC 14.3.1 on AlmaLinux 10.2; the existing NInfer container uses CUDA 12.8.
This is a separate, short comparison against the installed NInfer V100 service, not the
RTX 5090 corpus campaign above. The acceptance criterion was to match or beat that service.

Both servers used the Swift-1.5-Qwen3.8-27B-OrcaRouter-NVFP4-DFlash2 weights, including
its original chat template and sampling resources. The source NInfer artifact is v2;
the Infernix fixture has the same numeric payloads in a v3 container with the exact offline
Volta layout permutation. Its mathematical bindings were checked against the repository's
v2 upgrader. The generic upgrader replaces the chat template, so its default output alone
would not reproduce this comparison's prompt token counts.

| Setting | Both servers |
|---|---|
| Route | Persistent loopback OpenAI Chat Completions, non-streaming, one request at a time |
| Context / KV capacity | 240,000 / auto; INT8 group-64 KV |
| Prefill chunk | 2,048 tokens |
| Speculation | MTP, four draft tokens, optimized proposal head |
| CUDA Graphs / vision | Enabled / enabled |
| Prefix cache | Original cache, one device state, eight host states, 8,192 MiB host KV budget |
| Request | Bicycle maintenance notes repeated 72 times, then a request for a detailed guide |
| Sampling | Temperature 0.7, thinking off; remaining defaults from the same artifact |
| Repetitions | One excluded warmup, then three measured requests; none discarded |
| Tokens | Every measured request: 2,061 prompt, 256 completion, zero cache hits |

Rates below are the HTTP `timings.prompt_per_second` and `timings.predicted_per_second`
fields; decode divides 255 decode intervals by their elapsed time. Wall time is curl's
complete request time. Means are arithmetic means of per-request values; deviations are
sample standard deviations. Acceptance pools accepted and drafted tokens across the three
requests. This unseeded stochastic sample does not establish a universal speed advantage.

| Trial | NInfer prompt tok/s | Infernix prompt tok/s | NInfer decode tok/s | Infernix decode tok/s | NInfer wall s | Infernix wall s |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 926.54 | 945.86 | 48.34 | 59.32 | 7.502 | 6.506 |
| 1 | 928.79 | 943.62 | 54.94 | 51.40 | 6.866 | 7.172 |
| 2 | 923.48 | 941.35 | 55.96 | 56.11 | 6.795 | 6.760 |
| Mean ± SD | 926.27 ± 2.67 | 943.61 ± 2.25 | 53.08 ± 4.14 | 55.61 ± 3.98 | 7.054 ± 0.389 | 6.813 ± 0.336 |

The final means meet the criterion: **+1.87% prompt rate, +4.77% decode rate, and
3.43% less wall time**. Trial 1 is an exception: decode was 6.44% slower and wall time
4.46% longer. Pooled draft acceptance was 36.29% for NInfer and 40.24% for Infernix;
changed draft acceptance contributes to the observed decode results. The reserved 240K
context was exercised at startup, not filled by this 2K-prompt workload. Longer-context,
multi-request, vision and DFlash throughput are not established by these measurements.

### Development findings and costs

Earlier complete versions missed the same baseline; their measurements remain part of the
record. These are successive implementations, not repetitions pooled into the final mean.

| Version | Mean prompt tok/s | Mean decode tok/s | Disposition |
|---|---:|---:|---|
| Initial complete port | 755.75 | 52.44 | 18.4% prompt regression; profiled the whole HTTP request |
| Vectorized staging / SIMT gating / Q4 head | 899.55 | 52.89 | Still below both baseline means |
| Wider GEMM / consistent Q8 decode | 922.45 | 55.05 | Prompt rate still 0.4% below baseline |
| Bounded wide Q8 GEMM | 943.61 | 55.61 | Final measured implementation |

The first warmup aborted because a cooperative GDN launch used Blackwell occupancy assumptions;
actual kernel occupancy fixed the launch, and the production Volta prefill route now uses SIMT.
Numerical qualification also caught and fixed batched BF16 KV writes, attention masking and
workspace mismatches, insufficient FP16 attention accumulation, NVFP4 divisor overflow,
and record-workspace and Q8 launch-limit errors. CUDA compilation/device-link issues were
resolved before qualification.

Two later wide-Q8 attempts failed startup before any measured request: unconstrained staging
exceeded the matched memory budget, and a general 2,048-token bound still exceeded it by
30.3 MB. The final query and execution share a 512-token bound for the vision merger's
`[5120,4608]` geometry and a 2,048-token bound elsewhere. Context, vision and automatic headroom
settings were retained. Staging uses caller-owned temporary FP16 operands and FP32 outputs;
there is no extra persistent dequantized weight copy.

A 16-token SIMT GDN tile was rejected: it passed the oracle but was 6.3–27.5% slower than the
8-token tile across T=64,128,512,1024,2048. A Q8 switch for T=1 alone was rejected because it
broke exact column invariance. The retained consistent small-batch policy improves the five
measured MTP Q8 shapes at T=1 and T=4, but at **T=8** the `[14336,5120]` and `[34816,5120]`
shapes take **231.424 and 538.624 µs**, versus **205.824 and 467.968 µs** with the earlier QPN
route: **12.4% and 15.1% slower**. The other three T=8 shapes improved 15.9–19.4%.
These are cold, single-Op CUDA Graph medians (five warmups, 30 samples), not multi-request
inference results. This tradeoff preserves column invariance and prioritizes the measured
single-request workload.

The fixed Volta graph/driver reservation increased from 64 to 256 MiB, plus 4 MiB per executable.
That reserves 192 MiB more in startup planning. Earlier DFlash runs used 218–250 MiB against
an insufficient allowance; the final forced 32-node tree used 234 MiB within its 296 MiB
allowance. Final ordinary/MTP/tree smoke runs each used a 740 MiB workspace. The CUDA 13 /
Blackwell build and GPU routes were not requalified on this V100-only machine.

Wide Q8 staging also increases the vision smoke's workspace from 780.0 to 839.5 MiB
(+59.5 MiB). This is temporary planned memory, additional to the graph reservation above.
The matched server still starts with vision and the original 240K context settings.

### V100 qualification

Qualification used the same V100 and CUDA 12.9 toolchain as the comparison. Independent
FP32/FP64 Op oracles passed for native and packed projections, fused input/residual/SwiGLU
routes, embeddings and proposal-head scores, GDN state transitions, speculative trees,
and BF16/INT8/FP8 attention and cache writes. Exact codec/container checks passed (21 Python
tests), as did GPU artifact materialization and writer interoperability. Tests exercise
CUDA Graphs, masked/ragged attention, workspace boundaries and canaries. The final Q8 suite
also covers exact column invariance on five real MTP geometries and staging tile tails:
T=2048/2049/4097 for MTP and T=511/512/513/1025 for the vision merger.

Through the public Engine, final ordinary, MTP4 and forced 32-node DFlash2-tree runs produced
identical greedy 64-token outputs with INT8 KV, context 4096 and prefill chunk 2048. The
answer reached its token cap; this checks route consistency, not completion quality. The
tree ran 20 rounds, including seven side rounds, with zero fallback steps. The chart vision
fixture correctly returned its title, three bars and the leftmost tallest bar.

A short CausalScoring smoke used 169 input tokens (168 scored), one window, context 256,
stride 128 and prefill chunk 256. Final perplexity was **7.291306 with BF16 KV** and
**7.288551 with INT8 KV**. INT8 versus BF16 had top-1 agreement **98.81%** and mean KL
lower bound **0.00048767** over the reference's top 32 tokens plus an other-token bucket.
Earlier port results on this smoke were 7.288158/7.3185 (BF16/INT8) with KL 0.00042934;
the final BF16 perplexity and KL are slightly higher, while INT8 perplexity is lower.
These changes are retained in the record; this small sample does not establish corpus-level
quality or numerical identity with NInfer. Long-context and concurrent end-to-end quality
remain unmeasured.

### V100 reproduction

Build and pack using the [V100 instructions](../maintainer/build-system.md#configuration).
The measured local artifact is
`out/Swift-1.5-Qwen3.8-27B-OrcaRouter-NVFP4-DFlash2-volta.infernix`.
Start a fresh server for the four-request sequence:

```sh
build-v100/apps/infernix-serve \
  out/Swift-1.5-Qwen3.8-27B-OrcaRouter-NVFP4-DFlash2-volta.infernix \
  --model-id qwen3.8-27b --host 127.0.0.1 --port 8080 \
  --max-context 240000 --max-concurrency 1 --kv-capacity auto --prefill-chunk 2048 \
  --kv-dtype int8 --spec mtp --draft-tokens 4 --lm-head-draft \
  --use-original-prefix-caching --device-state-slots 1 --vision --preserve-thinking \
  --pending-timeout-ms 600000
```

Local response JSON, wall times, earlier results, failed-startup logs and the exact client script
are retained under `profiles/bench/v100-port-2026-10-09/` (generated reports, not source-controlled).
In another terminal, pass a fresh output directory to the retained script:

```sh
bash profiles/bench/v100-port-2026-10-09/http_compare.sh /tmp/infernix-v100-repeat
```

The script sends labels `warmup`, `measured-0`, `measured-1`, `measured-2`. Each user message is
`Notes LABEL: ` followed by 72 repetitions of
`The bicycle requires regular maintenance. Inspect tire pressure, brake pads, chain wear, spoke tension, and bearing play before a long ride. `
and then a newline followed by
`Write a detailed guide to maintaining a bicycle, covering diagnosis, tools, repair steps, and preventive checks. Continue until the output limit.`
It sets `max_tokens:256`, `enable_thinking:false`, `temperature:0.7` and model `qwen3.8-27b`.
Verify the actual token and cache-hit counts when using another artifact or template.

### V100 DFlash2 versus MTP

A follow-up on the same October 9 build compared Infernix MTP4, DFlash2 with seven
draft tokens and chain verification, and DFlash2 with seven draft tokens and
`--draft-tree-nodes auto`. All retained `--lm-head-draft`, vision, INT8 KV, the original
prefix cache, a 2,048-token prefill chunk and one execution lane. Hardware and HTTP workload
were the same as above. Each mode used a fresh server, one excluded warmup and three measured
requests; none were discarded. Every measured request had 2,061 prompt tokens, 256 output
tokens and zero cache hits. Sampling remained unseeded at temperature 0.7.

DFlash2 **could not start at the original 240,000-token context reservation**: minimum runtime
plus automatic headroom required 11,956,537,344 bytes after weights, but only 10,488,119,296
were available (a 1.37 GiB shortfall). This failed attempt produced no measured requests.
The comparison therefore used **32,768 context tokens for all three modes**, including a
fresh MTP measurement. No model code, power limit, headroom or vision setting changed.
Loaded weights were 21.7 GiB for DFlash2 versus 20.0 GiB for MTP; their default pinned host
context allocations were 9.46 and 9.15 GiB respectively.

| Mode | Decode trials 0 / 1 / 2, tok/s | Mean decode ± sample SD | Mean prompt tok/s | Mean wall s | Pooled draft acceptance |
|---|---|---:|---:|---:|---:|
| MTP4 | 57.33 / 54.99 / 56.58 | 56.30 ± 1.19 | 937.74 | 6.755 | 41.13% |
| DFlash2 chain, K=7 | 26.32 / 23.13 / 21.98 | 23.81 ± 2.25 | 828.40 | 13.294 | 23.51% |
| DFlash2 auto tree, K=7 | 26.16 / 25.00 / 24.65 | 25.27 ± 0.79 | 824.82 | 12.628 | 25.70% |

DFlash2 chain and auto-tree mean decode rates are **57.72% and 55.11% lower** than MTP;
mean wall times are **96.78% and 86.93% longer**. Every DFlash2 decode trial is slower
than every MTP trial. Auto trees improve the DFlash2 mean by 6.16%, but their first trial
is slightly slower than the chain's first trial. MTP is the measured choice for this workload.
Tree draft counts include side candidates, so acceptance percentages alone do not isolate
the cause of the throughput difference. No kernel attribution was collected in this comparison.
Other draft lengths, fixed tree widths, prompts and long contexts remain unmeasured; this
does not establish DFlash2 performance against NInfer's own DFlash2 implementation.

Exact startup scripts, all response JSON and wall times, the failed 240K startup log, and
`dflash2-comparison.json` are retained in `profiles/bench/v100-port-2026-10-09/`.
The measured directories are `mtp-32k/`, `dflash2-chain-32k/` and `dflash2-auto-32k/`;
their scripts are `start_compare_32k_mtp.sh`, `start_compare_32k_chain.sh` and
`start_compare_32k_auto.sh`. Reuse `http_compare.sh` with a fresh output directory.
All benchmark servers were stopped afterward; NInfer remains stopped.

### V100 MTP activation-staging optimization

The follow-up optimization targets the same dense Qwen27B MTP4 workload, with a goal of
at least 15% more decode throughput than the installed NInfer baseline. It retains the
original **240,000-token context reservation**, vision, INT8 KV, 2,048-token prefill chunk,
optimized proposal head, prefix-cache settings and 170 W power limit. No model weights,
sampling settings or request text changed. Each measured request still has 2,061 prompt
tokens, 256 output tokens and zero cache hits. There is one excluded warmup and three
measured requests, with none discarded; the earlier results above remain separate baselines.

A warmed whole-request Nsight Systems capture identified the small-batch FP8/NVFP4
projections as the main remaining decode cost. They repeatedly converted the same BF16
activations to FP16 in each output-row block. The workspace-taking route now converts once
per Op for T <= 32 and reuses that FP16 buffer. The tensor-core math, FP32 accumulation,
scale application and output policies retain their existing boundaries. Resource queries
include the buffer; no persistent weight expansion or hidden device allocation was added.
The workspace-free public overload continues to convert within its kernel.

| Trial | Prompt tok/s | Decode tok/s | Wall s |
|---|---:|---:|---:|
| 0 | 951.14 | 63.90 | 6.184 |
| 1 | 951.03 | 67.04 | 5.998 |
| 2 | 947.51 | 60.57 | 6.412 |
| Mean ± sample SD | 949.89 ± 2.06 | 63.84 ± 3.24 | 6.198 ± 0.208 |

Against NInfer's 926.27 prompt / 53.08 decode tok/s and 7.054 s wall means, this is
**+2.55% prompt, +20.27% decode and 12.14% less wall time**. Against the initial qualified
port's 943.61 / 55.61 tok/s and 6.813 s, it is **+0.67% prompt, +14.79% decode and 9.02%
less wall time**. Every optimized decode rate exceeds every initial-port and NInfer rate.
The weakest numbered trial comparison against NInfer is trial 2 at +8.22%; the 15% target
is met by the mean, not every individual trial. Pooled draft acceptance is 39.93%, versus
40.24% for the initial port and 36.29% for NInfer. These remain short, unseeded samples
on one prompt family; no universal or long-context speedup is claimed.

Public Linear microbenchmarks used independently packed Volta weights, cold CUDA Graphs,
five warmups and 30 measured samples per point. At each of T=1,4,5,8,16, the four FP8
geometries `[16384,5120]`, `[14336,5120]`, `[5120,6144]`, `[5120,17408]` and NVFP4
`[5120,17408]` improved: all 25 median latencies fell by **20.3–48.7%**. The unchanged
NVFP4 SwiGLU `[34816,5120]` control ranged from 0.7% faster to **0.6% slower**, with T=16
unchanged. These Op results are separate from the HTTP throughput claim.

All seven affected numerical suites passed: packed/native Linear, attention input, GDN
input, GDN snapshot and record, FP8 residual, and NVFP4 residual. One initial GDN test
failed its workspace assertion because it expected the workspace-free overload to consume
the policy overload's new scratch allocation; the assertion now distinguishes those calls,
and the rerun passed. Numerical oracles, graph checks and canary checks were retained.
The extra temporary buffer is `2*K*T` bytes, arena aligned, and an extra cast launch is
captured per affected Op. Final plain/MTP/tree smokes retain the 740 MiB planned workspace.
Measured CUDA Graph memory increases from 6 to **8 MiB** for plain/MTP and from 234 to
**244 MiB** for the forced 32-node DFlash2 tree; all remain within the existing allowances.
Their greedy 64-token outputs match both each other and the earlier build exactly.
The chart vision output also matches, with its workspace still 839.5 MiB and graph memory
8 MiB (previously 6 MiB). Both 168-position scoring summaries are exactly unchanged:
BF16/INT8 perplexity 7.291306/7.288551, mean KL lower bound 0.00048767 and top-1 agreement
98.81%. These are functional/numerical smoke checks, not corpus-quality qualification.
Blackwell code is not selected by this change; its build/GPU
regression testing remains unavailable on this machine.

Evidence is retained in `profiles/bench/v100-mtp-optimization-2026-10-09/`: the whole-request
profile, exact packed projection probe, before/after CSVs, oracle logs, and `staging-http/`
with all HTTP responses and timings. Reproduce using the original 240K startup command and
HTTP client above. The DFlash2 comparison predates this shared-kernel optimization and has
not been remeasured; it must not be interpreted as current DFlash2 throughput.

### V100 broader MTP comparison and second optimization

This follow-up compares the staged-input build above, a further optimized build, and a fresh
run of the installed NInfer V100 service. It uses the same V100 PCIe 32 GB at 170 W, model
weights and original template, INT8 KV, MTP4, optimized proposal head, vision reservation,
240,000-token context limit, 2,048-token prefill chunk and one resident request. Infernix uses
CUDA 12.9; the installed NInfer binary uses CUDA 12.8, on the same driver. These are engine
comparisons, not a controlled comparison of compiler versions.

Requests were frozen before comparing results: bicycle maintenance (2,061 prompt tokens),
a Python cache review (1,558), and a maintenance archive (15,889). Each generates 256 tokens
at temperature 0.7 with thinking disabled. Each workload has one excluded warmup (seed 11)
and three retained measurements (seeds 17, 29, 43). All have zero prefix-cache hits. Per-request
labels prevent reuse; request JSON is identical across engines and builds. Seeds control
request inputs but do not guarantee identical sampled output or draft acceptance across
engines. Reported means are arithmetic means of the three runs; the aggregate is the geometric
mean of the three workload speed ratios, with each workload weighted equally.

The fresh staged-build comparison revealed a long-prompt weakness: its prefill rate was
749.67 tok/s versus NInfer's 892.32, and total latency was 25.580 s versus 22.896 s, despite
faster decode. This result is retained; the earlier single-prompt, unseeded +20.27% decode
result is not substituted for this broader comparison.

Whole-request profiles attributed 40.3% of long-request GPU time to wide GEMM, 23.6% to
prefill attention and 7.0% to recurrent GDN. Short-request profiling also identified the
packed NVFP4 SwiGLU projections as a substantial decode cost. Profiled times are used for
attribution; throughput comparisons use unprofiled HTTP requests.

The selected implementation combines the adjacent gate/up projections into one grid for
NVFP4 T <= 32 and FP8 T <= 16. FP32 gate/up results still feed the same SwiGLU epilogue and
final BF16 cast. FP8 T=17–32 retains separate grids because the combined T=32 experiment
was 9.38% slower. The 24-query-head attention geometry uses 32 query columns per tile
instead of 16, retaining FP32 weighted-value accumulation, causal masks and stream-K fixup.
The 16-head geometry retains its existing tile. No public interface, artifact representation,
model binding, persistent weight expansion or hidden allocation is introduced. Scratch and
metadata remain owned by the existing workspace queries and execution arena.

Rejected alternatives remain relevant: prefetching two or four packed NVFP4 weight groups
slowed the important T=1–8 private-kernel probes by roughly 7–17%; gains at T=16 did not justify
adopting it. Three alternative wide-GEMM tile/swizzle schedules produced mixed results and
no consistent improvement over the existing 128x256 tile. Neither experiment entered production.

Public INT8 append-and-attend and cached-only attention benchmarks use cold CUDA Graphs,
five warmups and 30 samples, with T=64/256/2048 and history lengths 0/2048/14336. The larger
tile reduces median latency by 12.53–55.35% at the 12 points that select the affected route;
the six other points range from 0.05% faster to 0.10% slower. At T=2048 the reduction is
12.53–23.23%. Reported workspace capacities and measured peaks are identical at all 18 points.
These are public Op measurements, not end-to-end speedups.

A further 64-key attention tile was rejected: it slowed all 12 affected points by
5.28–40.97% relative to the selected 32-key tile, despite fitting the same one-block occupancy.
It was reverted before end-to-end measurement. Final public SwiGLU medians at T=1/4/5/8/16/32
fall by 6.12–12.36% for NVFP4 and 2.16–11.90% for the changed FP8 T<=16 route; the unchanged
FP8 T=32 control is 0.40% faster. The original combined FP8 T=32 regression remains the reason
for retaining its separate-grid implementation. SwiGLU workspace capacities are unchanged.

Unprofiled HTTP results (mean, with the full three-run min–max range in parentheses):

| Workload / engine | Decode tok/s | Prefill tok/s | Wall s | Draft acceptance |
|---|---:|---:|---:|---:|
| Bicycle / NInfer | 58.40 (56.63–61.55) | 926.15 (920.64–929.76) | 6.603 (6.365–6.744) | 42.63% |
| Bicycle / previous Infernix | 62.56 (60.82–64.36) | 947.81 (946.66–949.37) | 6.280 (6.165–6.391) | 38.44% |
| Bicycle / optimized Infernix | 68.17 (65.32–72.65) | 957.29 (956.22–958.76) | 5.929 (5.691–6.080) | 43.12% |
| Code / NInfer | 62.29 (60.15–65.92) | 873.40 (872.53–875.12) | 5.891 (5.661–6.027) | 47.03% |
| Code / previous Infernix | 69.68 (67.06–74.16) | 870.81 (869.12–872.22) | 5.483 (5.251–5.617) | 45.56% |
| Code / optimized Infernix | 67.36 (65.28–70.24) | 877.75 (877.24–878.33) | 5.591 (5.433–5.707) | 42.40% |
| 16K archive / NInfer | 50.47 (47.16–55.07) | 892.32 (889.21–895.56) | 22.896 (22.453–23.289) | 39.95% |
| 16K archive / previous Infernix | 58.61 (57.27–60.50) | 749.67 (748.35–751.37) | 25.580 (25.453–25.717) | 42.14% |
| 16K archive / optimized Infernix | 59.34 (57.33–62.65) | 792.40 (791.41–793.89) | 24.388 (24.117–24.557) | 42.22% |

Against previous Infernix, the bicycle/code/archive decode means change by
**+8.97% / -3.34% / +1.24%**, prefill by **+1.00% / +0.80% / +5.70%**, and wall latency by
**-5.59% / +1.98% / -4.66%**. Equal-workload geometric means give **+2.16% decode,
+2.47% prefill and 2.82% less wall time** (2.90% faster inverse wall time). The code-review
regression is retained: its worst paired seed loses 11.97% decode speed and takes 8.67%
longer. Across all nine paired requests, decode changes range from -11.97% to +12.88%,
prefill from +0.70% to +5.75%, and wall latency from -7.69% to +8.67%.

Against fresh NInfer, decode means improve by **16.72% / 8.13% / 17.56%**; the geometric
mean improvement is **14.06%**. This broader result falls below a 15% aggregate target.
Prefill changes by **+3.36% / +0.50% / -11.20%** and wall latency by
**-10.21% / -5.09% / +6.52%**. The archive request remains slower overall. Aggregate prefill
is 2.65% lower and wall latency 3.18% lower (3.28% faster inverse wall time). All nine paired
decode rates exceed NInfer, by 0.96–27.38%; paired prefill changes range from -11.35% to
+4.14%, and wall latency from -10.59% to +9.08%. Faster mean decode does not establish a
win for every request or latency metric.

Draft acceptance changed between builds, increasing on bicycle and decreasing on code.
These sampled throughput differences include that effect; they cannot all be attributed to
kernel latency. Three runs per workload do not establish a universal gain or statistical
confidence across applications. No Claude-produced patch or comparable benchmark was available,
so these results do not establish a comparison against that implementation.

All nine affected numerical suites passed: packed Volta, native FP8 and native NVFP4 SwiGLU,
plus the main and wide BF16/INT8/FP8 attention suites. They retain their independent FP64
oracles and existing error criteria; no tolerance was relaxed. SwiGLU qualification uses
independently decoded stored weights/scales at the real 34816x5120 geometry, including
T=1/5/8/16/32/33/65/129 and graph cases. Attention qualification covers independently decoded
KV, long histories, fragmented pages, append/cached routes, graph envelopes, masked-prefix
invariance, partial batches and canaries. The rejected 64-key variant was performance-tested
only and is absent from the final source.

Final 64-token greedy plain/MTP/tree and chart-vision outputs match the preceding build byte
for byte. Planned/measured workspace remains 740 MiB for text and 839.5 MiB for vision.
CUDA Graph memory remains 8 MiB for plain/MTP/vision and falls from 244 to 242 MiB for the
forced 32-node tree. Tree throughput was not remeasured. Blackwell compilation and GPU testing
remain unavailable; these changes select Volta implementations only.

Evidence is retained in `profiles/bench/v100-mtp-round2/`: `baseline/`, `ninfer/` and
`candidate/` contain every request, response, timing and summary; `suite.py` defines the
frozen requests. Run the original 240K server command above, then
`.venv/bin/python3.11 profiles/bench/v100-mtp-round2/suite.py <fresh-output-directory>`.
The same directory retains short/long Nsight captures, public-Op CSVs and probe sources,
rejected-experiment results, `candidate-oracles.log`, and `final_smokes.sh` with its outputs.

The final 168-position BF16/INT8 scoring summaries are exactly unchanged:
perplexity 7.291306250865034 / 7.288551281435343, INT8 mean KL lower bound
0.0004876700137516895 against BF16, and top-1 agreement 98.81%. Numerical KL fields also
match; only the reference-file path differs. These short checks do not establish corpus-wide
quality or long-context task accuracy. All benchmark/smoke GPU processes exited, and the
NInfer service remains stopped.
