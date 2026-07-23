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
- Makes its first per-block decision: if a codec's output is not
  smaller than the input, the block is stored RAW instead, so
  incompressible data never expands. (Measured on a 1 GiB 50/50
  zero/random dataset, this alone moves blind RLE from ~1.0x to ~2.0x
  effective compression.)
- Verifies every block round-trips to the original bytes via a CRC32
  checksum computed over the uncompressed data.
- Benchmarks compression ratio and throughput (GB/s) via
  `membrane-bench`, reporting results as JSON.

## What v0.0.1 explicitly is NOT

- No LLM runtime integration (no llama.cpp, no vLLM).
- No GPU/CUDA code.
- No multi-tier hot/warm/cold engine (the store has single-tier LRU
  eviction; see "Budget-aware block store" below).
- No NVMe or CXL storage backend (evicted blocks are dropped, not
  offloaded).
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

## Budget-aware block store (Phase 1.2)

`membrane_store_t` (`include/membrane/store.h`) holds compressed blocks
under a fixed byte budget and is the first piece that decides *which*
blocks stay resident. Design decisions:

- **Two physical states, not three tiers.** A block is either RESIDENT
  (compressed, counted against `budget_bytes`) or gone. There is no
  file/NVMe backend yet: an evicted block is simply dropped. The
  roadmap's hot/warm/cold hierarchy is deliberately *not* built as three
  physical pools here — that would be premature while only RAM exists.
- **Budget is never exceeded.** `put` compresses first, then evicts the
  least-recently-used blocks until the new block fits. If it still does
  not fit (e.g. a single block larger than the whole budget, or every
  other block pinned), `put` returns `MEMBRANE_ERR_BUDGET_FULL` and any
  existing data for that id is left intact. Fit checks are written as
  subtractions (`need <= budget - (resident - old_size)`) so they never
  overflow `size_t`.
- **Data structures.** An `id -> store_entry` chained hash index for
  O(1) lookup, plus an intrusive doubly-linked LRU list threaded through
  the same entries for O(1) touch/evict. Each `store_entry` owns one
  `membrane_block_t`, reusing its size/codec/checksum metadata rather
  than duplicating it.
- **Thread safety: one mutex + pin/refcount.** A single
  `pthread_mutex_t` guards the index, LRU list, and accounting. The slow
  decode runs *outside* the lock: `get` pins the entry (so eviction
  skips it), releases the lock, decodes with the pure `membrane_block_decode`,
  then re-locks to unpin. A `remove` or overwrite of a pinned block
  unlinks it immediately but defers the `free` to the last unpin, so
  concurrent `get`/`remove` cannot use-after-free. Verified under
  ThreadSanitizer with a multi-threaded get/put/remove stress test.
- **Accounting knows the difference between logical and physical.**
  `resident_bytes` (compressed footprint, the budgeted quantity) versus
  `logical_bytes` (sum of original sizes); their ratio is the
  `effective_capacity_ratio` — the headline number for "more logical
  data than physical budget."

Deliberately deferred (flagged as premature for this cut): async
compression workers, prefetch, a pluggable eviction-policy vtable, and
hash-index resizing.

## Persistent backend and promotion (Phase 1.4)

Evicted blocks are no longer dropped: with a backend configured, the store
writes them to a cold tier and loads them back on demand. This is what
pushes effective capacity *beyond* the RAM budget.

- **Backend seam.** `membrane_backend_t` (`include/membrane/backend.h`) is
  an opaque handle over an internal vtable (`store`/`load`/`remove`/
  `contains`/`used_bytes`/`destroy`), mirroring the codec vtable. All state
  lives in the handle — no globals. The only backend today is the file
  backend; RAM/CXL/FPGA backends would implement the same vtable. The store
  references but does not own the backend (the caller destroys it after).
- **File format.** One file per block, named only from the numeric id
  (16 hex digits — no caller string reaches the path, so there is no
  traversal surface). A 48-byte little-endian header carries a magic,
  version, id, sizes, both codecs, the payload checksum, and a CRC32 over
  the header itself; the payload follows. Writes go to a `.tmp` file that
  is `fsync`ed and atomically `rename`d into place. On load, a bad magic/
  version/header-CRC, an id mismatch, or a `stored_size` exceeding the file
  is rejected as `CORRUPT_DATA`; a corrupted payload is caught later by the
  block's own checksum at decode. Truncation fails the header read or the
  size check.
- **Eviction / promotion with I/O off the lock.** Entries gain a state:
  RESIDENT, EVICTING, EVICTED, LOADING. The heavy backend read/write runs
  with the store lock released; the entry is parked in a transient state
  (EVICTING/LOADING) that its owning thread holds exclusively. Any other
  thread that finds a transient entry waits on a condition variable and
  retries, so per-entry I/O is serialized without holding the lock across
  it, while different entries evict/promote concurrently. `get` on an
  evicted block loads it, decodes (verifying the checksum) *before*
  committing, then promotes it resident — evicting other LRU blocks to make
  room, or serving without promoting if the budget is fully pinned. Backend
  deletes (unlink) for `remove`/overwrite are done under the lock since they
  are cheap metadata operations; only the block payload read/write is moved
  off the lock.
- **Error safety.** A failed eviction write leaves the block resident
  (rolled back, counted as `backend_write_failures`). A failed load or a
  checksum mismatch never yields corrupt data — `get` returns the error and
  the block stays evicted. `remove` clears both the resident and backend
  copies; `destroy` removes every record and `.tmp` file. Validated by
  `membrane-store-bench`: 256 MiB logical held in a 64 MiB budget with a
  file backend keeps `resident_bytes` pinned to 64 MiB across thousands of
  evict/promote cycles, every block read back (in scrambled order) with
  integrity PASS, and no files left behind.

## Memory safety

The build supports `-DMEMBRANE_ENABLE_SANITIZERS=ON` (AddressSanitizer +
UndefinedBehaviorSanitizer) and `-DMEMBRANE_ENABLE_TSAN=ON`
(ThreadSanitizer; mutually exclusive with the ASan build). CI runs a
plain Debug build and an ASan build on every push. Note: running the
TSan build on recent kernels may need `setarch -R` to disable ASLR,
otherwise ThreadSanitizer aborts with "unexpected memory mapping" — this
is an environment quirk, not a data race.

## Memory safety

The build supports an `-DMEMBRANE_ENABLE_ASAN=ON` CMake option that turns
on AddressSanitizer + UndefinedBehaviorSanitizer. CI runs both a plain
Debug build and an ASan build on every push.
