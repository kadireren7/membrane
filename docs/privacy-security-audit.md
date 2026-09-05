# Privacy and security audit

Mega Phase C, PR C3. A real, grep-based source audit of
`tools/membrane/server.cpp`, `tools/membrane-run/runtime_session.cpp`,
and `tools/membrane-llama-runtime/decode_loop.cpp` — not a re-statement
of intent, an actual check of what the code does. `docs/server.md`'s
own "Security scope" section already documented most of these claims;
this audit independently verifies them against the current source.

## Findings

- **No prompt/message content is ever logged.** Every `fprintf`/
  `std::cerr`/`std::cout` call in `server.cpp` was inspected: none
  embeds a request's `messages`/`content` field. The one content-
  adjacent string in `decode_loop.cpp` (`"membrane: kv-store prompt
  decode failed\n"`) is a fixed, static diagnostic message — it never
  contains the actual prompt text.
- **No external (non-loopback) network calls exist anywhere in this
  codebase.** Every `http://`/`https://` occurrence in
  `tools/membrane*` is either building the server's own local bind-
  address string (127.0.0.1 or the configured `listen_address`,
  loopback-only by default) or `doctor_cmd.cpp`'s own HTTP client
  talking to that same local server. No telemetry, no update-check
  ping, no analytics call of any kind.
- **Error responses never leak a filesystem path.** Every
  `send_json_error()` call site in `server.cpp` was inspected: none
  embeds a registry-internal path (confirmed directly, not just
  documented) — matches the existing claim in `docs/server.md`.
- **No request/response is written to any log file.** No `fopen`/
  `ofstream` targeting a log file exists in `server.cpp` or
  `runtime_session.cpp`. The `server_config.h` `log_level` field is
  defined but not currently read anywhere in `server.cpp` — an honest
  observation, not a finding requiring a fix: an unused field cannot
  cause content logging.
- **No authentication, by design, loopback-only by default**
  (unchanged from `docs/server.md`'s own existing "Security scope" —
  re-verified, not re-litigated, this PR).

## Scope

This audit covers the HTTP server and generation path specifically
(the surface a remote-ish client actually talks to). It does not cover
`third_party/llama.cpp` itself (an unmodified upstream dependency, two
narrow patches — `patches/`) or this project's own CLI tools' local
filesystem access (registry/config files, already covered by
`docs/model-registry.md`'s "Safety" section and
`docs/schema-versioning.md`).

## Regression guard

`scripts/verify-api-contract.py` includes a grep-based check that no
new `fprintf`/`cerr`/`cout` call in `server.cpp` embeds a `req.`/
`messages`/`content`/`prompt` reference, so a future change cannot
silently reintroduce content logging without at least this check
noticing.
