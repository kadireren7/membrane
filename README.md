# MEMBRANE

**Adaptive Computational Memory Engine for LLM Inference**

Turn limited AI memory into an adaptive hierarchy of compressed, quantized,
offloaded and recomputable blocks.

> **Status: Pre-alpha research prototype**
>
> Current milestone: lossless block store and compression benchmark
> (RAW + RLE codecs). No performance claims have been validated yet.

---

## 1. The memory wall

LLM inference servers have plenty of compute, but GPU VRAM/HBM capacity and
memory bandwidth are the actual bottleneck — mainly in the form of the
KV-cache, which grows linearly with context length and concurrent requests.

MEMBRANE's long-term goal: automatically keep, compress, quantize, move, or
recompute every block of LLM memory depending on how hot or cold it is,
instead of relying on fixed, statically configured tiers.

## 2. What MEMBRANE does (today)

Today, MEMBRANE is a standalone, dependency-free C11 library and CLI that:

- Splits arbitrary binary data into fixed-size blocks.
- Compresses each block with a pluggable codec (`RAW` passthrough, `RLE`
  today; more codecs later).
- Verifies every block round-trips losslessly via checksum.
- Reports compression ratio and throughput as JSON.

It does **not** yet talk to any LLM runtime, GPU, or quantization scheme —
see [docs/architecture.md](docs/architecture.md) for the explicit scope of
this milestone and the longer-term roadmap.

## 3. Architecture

```
┌─────────────────────────────────────────────┐
│                  LLM Runtime                │
│        llama.cpp / vLLM / custom engine      │  (future)
└──────────────────────┬──────────────────────┘
                        │ KV block API
                        ▼
┌─────────────────────────────────────────────┐
│           MEMBRANE DECISION ENGINE           │  (future)
│  KEEP · COMPRESS · QUANTIZE · MOVE · DROP    │
└──────────────────────┬───────────────────────┘
                        │
         ┌──────────────┼───────────────┐
         ▼               ▼               ▼
    Fast Tier        Warm Tier       Cold Tier      (future)
    GPU/VRAM          CPU RAM        NVMe/CXL
```

Today's scope is the bottom-most piece: a block store with pluggable
codecs, described in full in [docs/architecture.md](docs/architecture.md).

## 4. Current status

See [docs/architecture.md](docs/architecture.md) for exactly what exists,
what's stubbed out, and what's not started yet.

## 5. Building from source

Requires CMake >= 3.16 and a C11 compiler (gcc or clang).

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

To build with AddressSanitizer + UndefinedBehaviorSanitizer enabled:

```bash
cmake -S . -B build -DMEMBRANE_ENABLE_ASAN=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## 6. Roadmap

See [docs/architecture.md](docs/architecture.md) for the phased roadmap,
from the current lossless block store through KV-cache integration,
quantization, predictive prefetch, and eventual FPGA/CXL exploration.

## 7. Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## 8. Citation

A `CITATION.cff` will be added once the project reaches a citable milestone.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
