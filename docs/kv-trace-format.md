# The `.memkv` KV trace format (Product Phase 3)

A minimal, versioned, offline trace format that lets
`tools/membrane-quant-policy-bench` benchmark real captured KV-cache
block data via `--trace FILE` without depending on llama.cpp or model
weights at benchmark run time. Implemented in
`tools/membrane-workload-core/trace_format.h`/`.c`.

## Why a new format

Existing maintained trace formats (`membrane/kvtrace.h`,
`membrane/attntrace{,2,3}.h`) store *derived statistics* — per-step KV
byte-growth deltas, or per-block attention scores — never raw
quantizable tensor values. `membrane/kvdump.h` does store raw K/V
tensor payloads, but as a general multi-record container (arbitrary
tensor dims, one record per layer/K-or-V, model name, token range) —
more structure than a quantization benchmark needs, and not shaped as
a flat sequence of fixed-size blocks. `.memkv` is the smallest format
that is exactly that: a real capture, already flattened into
`elements_per_block`-sized blocks ready for `membrane/quant_simd.h`.

## Pipeline

```text
real model run (MEMBRANE_ENABLE_LLAMA=ON, existing tools, unchanged)
        |
        v
membrane-kv-capture  -->  foo.kvdump   (existing format, raw K/V tensors)
        |
        v
membrane-kv-trace-export  -->  foo.memkv   (new, this format; unconditional
        |                                   build, never touches llama.cpp)
        v
membrane-quant-policy-bench --trace foo.memkv --policy adaptive|q4-only|q8-only
```

Only the first step needs `MEMBRANE_ENABLE_LLAMA=ON` and a `.gguf`
model. Everything downstream of a `.kvdump` file is llama-free.

## Wire layout

Little-endian throughout. Fixed 64-byte header, then metadata, then
the raw block payload:

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0  | 8 | `magic` | `"MEMKV01"` + 1 NUL byte |
| 8  | 4 | `format_version` | currently `1` |
| 12 | 4 | `dtype` | `0` = F16 (only supported value) |
| 16 | 4 | `elements_per_block` | must be > 0 |
| 20 | 4 | `metadata_length` | bytes of UTF-8 metadata that follow the header |
| 24 | 8 | `block_count` | must be > 0 |
| 32 | 8 | `payload_bytes` | must equal `block_count * elements_per_block * 2` |
| 40 | 8 | `created_unix_time` | informational only |
| 48 | 4 | `payload_checksum` | CRC32 (same algorithm as `membrane_block_checksum`) of the raw block payload |
| 52 | 4 | `header_checksum` | CRC32 of bytes `[0, 52)` |
| 56 | 8 | reserved | must be `0` |

Followed by `metadata_length` bytes of UTF-8 metadata (a short safe
label — see Privacy below), then `payload_bytes` bytes of raw
little-endian block data, block-major (block 0's `elements_per_block`
values, then block 1's, ...).

Every length is validated with overflow-safe arithmetic before any
allocation: unsupported `dtype`, zero/oversized `elements_per_block`
or `block_count`, an inconsistent `payload_bytes`, or an oversized
`metadata_length` are all rejected before the payload is ever read.
`tools/membrane-workload-core/test_trace_format.c` exercises all of
these against adversarial/malformed files. The reader
(`membrane_trace_read_block`) is genuinely streaming — one block per
call, never the whole payload buffered at once — while still verifying
the payload checksum incrementally across every block read.

## Privacy / provenance

A trace contains **only** numerical block data and a short metadata
label — never prompt or token text, and never model weights.
`membrane/kvdump.h`'s own header (what `membrane-kv-capture` produces)
has no prompt/token-text field to begin with, only a model name
string, a layer index, a K/V tag, and token *counts* — so there is
nothing prompt-derived for `membrane-kv-trace-export` to strip; its
metadata label is built solely from those already-safe fields (or
overridden with `--label`).

Because a trace file is untrusted input from the benchmark's
perspective (it may be handed a corrupt or adversarially-crafted
file), `metadata` and any trace-derived display name are treated as
data, not compile-time-safe strings, everywhere they reach JSON/CSV
output (`tools/membrane-quant-policy-bench/bench_core.h`'s
`membrane_bench_json_escape`/`membrane_bench_csv_escape`, shared by
single-trace and trace-set output alike).

## Producing a trace

```bash
cmake -S . -B build-llama -DCMAKE_BUILD_TYPE=Release -DMEMBRANE_ENABLE_LLAMA=ON
cmake --build build-llama -j --target membrane-kv-capture membrane-kv-trace-export

./build-llama/tools/membrane-kv-capture/membrane-kv-capture \
  --model models/<name>.gguf --prompt-file <prompt.txt> \
  --out capture.kvdump --n-tokens 64

./build-llama/tools/membrane-kv-trace-export/membrane-kv-trace-export \
  --input capture.kvdump --output trace.memkv \
  [--layer N] [--tensor k|v] [--elements-per-block N] [--label TEXT]
```

`membrane-kv-trace-export` picks the first F16 record in the
`.kvdump` file by default (`--layer`/`--tensor` to pick another) and
requires no `MEMBRANE_ENABLE_LLAMA` build flag itself.

## Consuming a trace

```bash
./build/tools/membrane-quant-policy-bench/membrane-quant-policy-bench \
  --trace trace.memkv --policy adaptive
```

See the main [README](../README.md#benchmark-a-captured-kv-trace) and
`membrane-quant-policy-bench --help` for the full `--trace` contract
(what `--blocks` means as a cap, why `--workload`/`--seed` are
rejected together with it, and the JSON/CSV trace fields).

## Batch: many layers/tensors from one capture (Product Phase 4)

One `.kvdump` capture already holds one record per layer/K-or-V (see
`membrane-kv-capture`'s own write order). `membrane-kv-trace-export
--output-dir DIR` walks every F16 record in the file (optionally
narrowed with `--layer-start`/`--layer-end`/`--tensor k|v|both`) and
writes one `.memkv` per record into `DIR` — already-existing directory,
never created by this tool — named deterministically
`layer-NNN-k.memkv` / `layer-NNN-v.memkv` (zero-padded to at least 3
digits). These names carry only a layer index and a K/V tag: no model
name, prompt, or other trace-derived string. A `.kvdump` containing two
F16 records for the same (layer, tensor) — corrupt/unexpected input —
fails the whole batch rather than silently overwriting one file with
the other.

`membrane-quant-policy-bench --trace-dir DIR --matrix` then discovers
every `.memkv` file directly inside `DIR` (non-recursive; symlinks are
rejected outright, not followed), runs all 3 policies against each
(3N result rows for N traces), and reports a block-weighted cross-trace
"adaptive" aggregate: storage bytes are summed once and divided (never
an average of per-trace percentages), q4/q8 mean errors are the exact
pooled mean over every underlying block across every trace, max errors
are the max of per-trace maxes, and aggregate throughput is total
processed blocks divided by total *measured* processing time — never a
sum of per-trace throughputs. A file named `layer-NNN-{k,v}.memkv`
surfaces its layer/tensor in the output; any other name still
benchmarks, just with `layer`/`tensor` reported as unknown (JSON
`null`). Two discovered files that resolve to the same (layer, tensor)
are rejected as a conflicting set before anything is benchmarked.

This answers "does Q4/Q8 suitability vary across layers/tensors on one
real model execution" for whichever one capture you supply — it is not
a claim about models or prompts in general.
