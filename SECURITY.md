# Security Policy

MEMBRANE is a research prototype, not a production system. It has not
been audited for security and should not be exposed to untrusted input
or deployed as a network-facing service.

## Scope

- The C11 core library (block store, codecs, quantization) and CLI tools
  parse binary files it produces itself (traces, checkpoints, CSVs) and
  local model files. It has not been hardened against maliciously
  crafted input files.
- The file-backed block store (`include/membrane/backend.h`) derives
  file names only from a numeric block id, not caller-supplied strings,
  specifically to avoid path traversal — but this has not received an
  independent security review.
- No component in this project accepts network input.
- `third_party/llama.cpp` is a git submodule with its own upstream
  security policy; vulnerabilities in it should be reported upstream,
  not here (see [docs/licensing.md](docs/licensing.md)).

## Reporting a vulnerability

If you find a real security issue (memory safety bug reachable from
untrusted input, path traversal, etc.), please report it privately by
emailing kadirerenaltintas072@gmail.com rather than opening a public
issue. Include:

- The affected file(s)/commit.
- Steps to reproduce, or a minimal input that triggers it.
- Why you believe it's security-relevant (not just a correctness bug —
  see [CONTRIBUTING.md](CONTRIBUTING.md) for ordinary bug reports).

This is a single-maintainer research project with no formal SLA; expect
an acknowledgment on a best-effort basis, not a guaranteed response time.

## Memory safety verification

Every change is expected to pass AddressSanitizer + UndefinedBehaviorSanitizer
and ThreadSanitizer builds (`-DMEMBRANE_ENABLE_SANITIZERS=ON` /
`-DMEMBRANE_ENABLE_TSAN=ON`) before being considered done — see
[CONTRIBUTING.md](CONTRIBUTING.md) and [docs/reproduction.md](docs/reproduction.md)
Level 1.2. This substantially reduces, but does not eliminate, the risk
of memory-safety bugs.
