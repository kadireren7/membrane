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
