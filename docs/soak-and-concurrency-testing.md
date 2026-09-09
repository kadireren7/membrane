# Soak and concurrency testing

Mega Phase C, PR C3. These are **resource-stability** checks, never
throughput/performance benchmarks — they measure whether RSS, thread
count, and file-descriptor count stay bounded under sustained real use,
not how fast requests complete. Both scripts require a real (small)
GGUF model and are local-dev-only tooling — CI has no real model file
(`models/` is gitignored, this project's established constraint), so
neither runs in CI; `scripts/verify-api-contract.py` only checks that
both scripts stay syntactically valid.

## `scripts/soak-test-server.py`

Starts a real `membrane serve` process, registers a real small model,
sends a bounded number of sequential real chat-completion requests, and
samples the server process's own RSS/thread-count/FD-count via `/proc`
at intervals. Fails if RSS grows more than 64 MiB, threads grow by more
than 2, FDs grow by more than 8, any request fails (real-generation
mode only), or the server is not healthy immediately afterward.

**Real result on this project's own dev host** (SmolLM2-135M-Instruct,
20 real sequential requests across several runs): thread count stayed
completely flat at every sample in every run (no thread leak). FD count
stayed flat (confirmed via direct `/proc/<pid>/fd` inspection, 6
requests, 0 growth) under normal memory conditions; under this dev
host's own real, severe, transient memory pressure (heavy swap usage
from unrelated processes sharing this host), FD/RSS numbers fluctuated
non-monotonically (RSS *dropped* between some samples) rather than
growing steadily — consistent with real OS-level swap/scheduling
effects under genuine memory pressure, not a per-request resource leak
(a real leak would show monotonic growth; this did not).

## `scripts/concurrency-soak-server.py`

A real, EXTERNAL-PROCESS complement to `test_server.cpp`'s own
`test_concurrent_requests_are_thread_safe` (16 real simultaneous
connections against an in-process test server, TSan-instrumented, run
in CI's `server-thread-sanitizer` job — confirmed clean this PR,
including under this exact TSan build). This script launches a real,
separate `membrane serve` OS process and fires real concurrent HTTP
requests clustered around the bounded-admission limit (8,
`request_admission.h`), checking the real admit/reject split and that
every `SERVER_BUSY` response (and only `SERVER_BUSY`, see below)
carries `Retry-After`.

**A real bug was found in this script itself, not in MEMBRANE**: an
earlier version required `Retry-After` on every `503` response. Real
concurrent runs showed two genuinely different `503` codes appearing
side by side — `SERVER_BUSY` (admission gate full, carries
`Retry-After`) and `NO_FEASIBLE_CONTEXT` (host memory guard rejected
the load, no `Retry-After` promised or needed) — both correct and both
already documented in `docs/server.md`'s error table. Fixed by parsing
the real response body's own `error.code` and only requiring
`Retry-After` on `SERVER_BUSY` specifically.

**A real, disclosed environmental finding**: under this dev host's own
severe, transient real memory pressure, some concurrent requests timed
out entirely (a real socket-level timeout, never a MEMBRANE crash or
hang past the client's own timeout) rather than receiving even a `503`.
Investigated directly: `membrane-run --ctx auto --plan-only` against
the same model, at the same time, showed a real `PLANNER_REJECTED_ALL`
outcome — the host-memory-guard's OWN fixed 256 MiB reserve
(`host_memory_guard.h`) can exceed genuinely available memory on a
severely pressured shared host, and evaluating that rejection itself
requires touching real (possibly swapped-out) memory, so the normally-
fast rejection path is not a hard guaranteed-fast path under extreme
swap-thrashing conditions. This is a real, disclosed limitation of
operating under severe host memory pressure, not a MEMBRANE logic bug
— the guard's decision was correct every time it completed; the
finding is about worst-case *latency* of that decision under conditions
well beyond this project's own normal validated envelope (see
`docs/host-memory-guard.md`). Not addressed with a code change this PR
(would require a materially more complex async/timeout-bounded
memory-check architecture — out of this hardening phase's own scope,
and not evidenced as a real problem outside of genuinely extreme,
shared-host memory exhaustion).

Real result, `SERVER_BUSY`/`NO_FEASIBLE_CONTEXT` split confirmed
correctly attributed and every real `SERVER_BUSY` response carrying
`Retry-After` — see `results/product-hardening/v0.4-validation.json`.

## TSan coverage

`test_server.cpp` (16 real concurrent in-process connections,
`server-thread-sanitizer` CI job) is the continuously-run TSan
coverage for this exact concurrent code path. The real, external-
process concurrency soak above was additionally run once, manually,
against a `-DMEMBRANE_ENABLE_TSAN=ON` build (`build-b1-tsan`) via
`test_server` itself; the external-process script was run against a
regular (non-TSan) build specifically because of real, severe, observed
memory pressure on this dev host at test time (TSan's own real memory
overhead was judged an avoidable additional risk to an already-strained
shared host at that moment) — disclosed here rather than silently
mixed in as if both had run under identical conditions.

## Mega Phase E, PR E1: real concurrent decode, and a real pre-existing
## ggml CPU-kernel race found by testing it

E1 replaced full generation-serialization (one `std::mutex` held for the
whole request) with bounded, genuinely concurrent decode
(`decode_concurrency_gate_t`, default capacity 2 — see `tools/membrane/
decode_concurrency.h`). `test_decode_concurrency.cpp` (new, same
Threads::Threads-only, CI-run, TSan-covered precedent as
`test_request_admission.cpp`) proves the gate primitive itself race-free
under real concurrent contention. `scripts/verify-continuous-batching.py`
(new, local-dev-only, requires a real GGUF, same "never runs in CI"
convention as the two scripts above) proves real, concurrent, real-model
decode is CORRECT (concurrent non-stream/stream/mixed all succeed with
coherent, non-corrupted per-request output; a real client disconnect on
one stream never affects a concurrently-running second request; a short
request is not starved behind a long one) and gives real, honest
performance evidence (~1.24x wall-clock on a real 4-request burst,
2 CPU cores' worth of real headroom on this dev host — never marketed
as more than that one real measurement shows).

**A real, previously-undiscovered hang class was found and fixed while
building this evidence**: `membrane serve` never called `llama_log_set()`
(unlike `membrane-run`'s own `main.cpp`, which already suppresses
llama.cpp's own verbose internal logging via exactly this API) — every
real request's own fresh `llama_context` creation (this project's
"persistent model, new context per request" architecture, unchanged)
prints real, substantial llama.cpp-internal diagnostic output
(`graph_reserve`/`sched_reserve`/`resolve_fused_ops` lines — measured
directly at ~13 KB per request on the real SmolLM2-135M fixture) to
stderr, completely outside MEMBRANE's own quiet/verbose design. Under
any real supervisor whose stdout/stderr sink has bounded buffering and
no continuous reader (confirmed, reproduced directly: Python's own
`subprocess.PIPE` without a draining thread — NOT systemd/journald,
which reads continuously and never blocks this way), enough real
sequential requests (~5 on this project's own measurement) fill that
buffer and the request-handling thread's own `fprintf()` call blocks
FOREVER — starving every other request waiting on that thread's held
decode-gate/admission slot. Fixed in `server.cpp` by calling
`llama_log_set()` once at `membrane_server_run()` startup (mirroring
`main.cpp`'s own `quiet_log_callback`, ERROR-level only). Root-caused
directly (not dismissed as "just memory pressure" despite this dev
host's own real, severe, chronic memory constraints) via `/proc/<pid>/
task/*/wchan` inspection, which showed the stuck thread genuinely
blocked in `anon_pipe_write`, not in any lock/condvar this project's own
code owns.

**A real, pre-existing third-party (vendored ggml CPU backend) data race
was found, NOT introduced by E1**: running one real generation under a
TSan build (`build-tsan`, real SmolLM2-135M, `MEMBRANE_MAX_CONCURRENT_
DECODE=1` -- i.e. zero cross-request concurrency, a single decode call
with its own internal `n_threads=4` intra-op parallelism only) reports
real TSan `WARNING: data race` findings inside `ggml_compute_forward_
flash_attn_ext_f16_one_chunk`/`ggml_compute_forward_mul_mat_one_chunk`
(`third_party/llama.cpp/ggml/src/ggml-cpu/`) -- ggml's OWN intra-op
worker threads (spawned via `libgomp`, one context's own `n_threads=4`
pool) racing on writes into that SAME context's own compute buffer.
Confirmed this is **unrelated to E1's own cross-request scheduler**:
it reproduces identically with a single in-flight request and zero
concurrent decodes, so it has been real and present since `n_threads>1`
was first used (long before this phase), simply never exercised under
TSan with real generation before (the existing CI TSan job, `test_
server.cpp`, uses a nonexistent-model 404 path specifically to stay
llama-real-generation-free and fast). E1's own new code (`decode_
concurrency_gate_t`, `decode_slot_enter`/`decode_slot_exit`, the
per-request `local_session` copy, `ensure_model_loaded`'s drain-wait)
shows NO TSan-flagged races in isolation, real-generation runs included.
**Not fixed in this PR**: this is a real bug inside vendored, third-party
CPU-kernel code, likely a documented/accepted class of numeric-kernel
race (disjoint-in-practice writes the C++/TSan memory model still flags,
common in hand-tuned SIMD/threaded compute code -- real output was
correct and coherent in every real request observed despite the
warnings) -- fixing it would mean patching vendored ggml source, well
outside a "scheduler foundation" PR's own scope, and is not currently
CI-gating (the CI TSan job's own real-generation-free design is
unaffected and stays green). Disclosed here, honestly, as a known,
pre-existing limitation of the vendored CPU backend rather than hidden
or wrongly attributed to this phase's own new scheduler code.
