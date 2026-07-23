# MEMBRANE Architecture

## Scope of this document

This document describes what MEMBRANE actually is *right now* (v0.0.1,
"Phase 0" of the roadmap below), and where it's headed. It is written to be
honest about the gap between the two — nothing below claims a capability
that isn't implemented and tested.

## What v0.0.1 is

A standalone, dependency-free C11 library (`membrane_core`) plus a CLI
(`membrane-bench`) that:

- Defines a `membrane_block_t` — a unit of storage with metadata (id,
  original/stored size, access counters, state, codec, checksum).
- Defines a pluggable codec interface (`include/membrane/codec.h`) —
  a small vtable of `compress` / `decompress` / `bound` function pointers,
  looked up by `membrane_codec_t`.
- Ships two real codecs: `RAW` (passthrough, the correctness baseline) and
  `RLE` (run-length encoding, the first real compressor).
- Ships two placeholder codec IDs (`LZ4`, `BITPACK`) that are registered in
  the codec table but return "unimplemented" — reserved for later phases,
  not wired into anything yet.
- Verifies every block round-trips to the original bytes via a CRC32
  checksum computed over the uncompressed data.
- Benchmarks compression ratio and throughput (GB/s) via
  `membrane-bench`, reporting results as JSON.

## What v0.0.1 explicitly is NOT

- No LLM runtime integration (no llama.cpp, no vLLM).
- No GPU/CUDA code.
- No hot/warm/cold eviction policy engine.
- No NVMe or CXL storage tier.
- No lossy compression or quantization.
- No predictive prefetching.
- No distributed/multi-server memory.

These are not abandoned — they're future phases, listed below in order.

## Why this project exists

LLM inference servers are usually compute-rich but memory-poor: GPU VRAM
capacity and bandwidth — not FLOPs — cap how much context and how many
concurrent requests a given piece of hardware can serve, mostly because of
the KV-cache. Existing engines already support some fixed-configuration
answers to this (e.g. quantized KV-cache types, static CPU offload). The
long-term goal here is to turn "what to do with each block of memory" into
a real-time, per-block decision instead of a static, user-set knob.

The first validated claim we're aiming for: **reduce KV-cache memory usage
by at least 40% while the model keeps working correctly.** Everything past
that (2x effective capacity, longer context, more concurrent requests) is a
consequence to measure later, not something assumed up front.

## Roadmap

- **Phase 0 (this milestone):** Lossless block store + RAW/RLE codec
  benchmark, fully local, no LLM dependency.
- **Phase 1:** Smarter block store — hot/warm/cold state, LRU-style
  eviction, background compression, trace capture/replay, all still
  running against synthetic or file-backed data.
- **Phase 2:** Real KV-cache data — pull actual KV-cache tensors out of a
  running llama.cpp model and analyze their compressibility (per-layer,
  per-token-position, K vs V).
- **Phase 3:** llama.cpp integration — a MEMBRANE KV backend plugged into
  llama.cpp's block allocator.
- **Phase 4:** Lossy KV quantization (FP16 → Q8 → Q4) with a measured
  quality/memory tradeoff per block.
- **Phase 5:** Predictive prefetch — predict which blocks will be accessed
  next from access history, layer, token position, and attention pattern.
- **Phase 6:** FPGA offload for compression/decompression/prefetch.
- **Phase 7:** CXL memory pooling and ASIC research.

## Codec interface

Codecs are plain C structs of function pointers, registered by
`membrane_codec_t` in a small lookup table (`src/codecs/codec_registry.c`).
This keeps the block store and CLI codec-agnostic — adding a new codec
means writing a `compress`/`decompress`/`bound` triplet and registering it,
nothing else needs to change. See `include/membrane/codec.h` for the exact
function signatures.

## Memory safety

The build supports an `-DMEMBRANE_ENABLE_ASAN=ON` CMake option that turns
on AddressSanitizer + UndefinedBehaviorSanitizer. CI runs both a plain
Debug build and an ASan build on every push.
