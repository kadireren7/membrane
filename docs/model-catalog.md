# Model catalog and download manager

Mega Phase D, PR D1. Turns MEMBRANE from "bring your own GGUF" into
"tell MEMBRANE what model you want" for a small, curated, evidence-
backed set of models — a curated catalog, never a web marketplace, and
never a scraper of arbitrary mirrors.

## Commands

```bash
membrane model search [QUERY]      # offline, no network -- browse the catalog
membrane model info NAME           # offline -- full detail for one family
membrane model install NAME [--quant Q4_K_M]   # real HTTPS download + register
membrane model uninstall NAME      # delete the downloaded file + unregister
```

`search`/`info` never touch the network — the catalog is compiled
directly into the `membrane` binary (`model_catalog.cpp`'s own
`MEMBRANE_CATALOG_JSON` constant), so browsing it works with zero
network dependency, the same principle this project already applies
to inference itself (`docs/server.md`'s own "no telemetry, no external
network calls"). Only `install` makes a real network call.

`install` with no `--quant` picks the **smallest available variant** —
an honest, conservative, non-hardware-aware default. Mega Phase D, PR
D2 (a separate, later PR) adds real hardware-aware variant selection;
this PR does not pretend to have it yet.

## Source policy

Every catalog entry is sourced from a real, reputable, named Hugging
Face repository — never scraped, never anonymous. Each family records:

- `provider` — the real Hugging Face org/user that published the GGUF
  files (not necessarily the original model author).
- `repo_url` — a real, browsable URL.
- `license` — read directly from the repo's own metadata tag.
- Per-variant `filename`, `download_url`, `size_bytes`, and `sha256` —
  every one individually verified this phase against the real,
  current upstream repo (`model_catalog.cpp`'s own top comment
  documents exactly how: a real `?blobs=true` API call for filenames/
  sizes/checksums, a real `curl -sIL` for the resolved download URL).

This phase's catalog covers 3 families (SmolLM2-135M-Instruct,
SmolLM2-360M-Instruct, Qwen2.5-1.5B-Instruct) — models this project
already has real compatibility evidence for
(`docs/compatibility.json`). Broader family expansion (Mistral, Phi,
Gemma) is real, separate work with its own evidence requirements
(Section 23 of the Mega Phase D task) — not invented here without
that same bar, tracked as a known limitation.

## Download safety

`download_manager.h`/`.cpp`, backed by libcurl (chosen over adding
OpenSSL directly to the existing vendored cpp-httplib specifically
because this dev host had no root access to install `libssl-dev`,
while libcurl's dev headers were already present — see that module's
own top comment):

- **HTTPS only** — rejected before libcurl is ever invoked (a plain
  `http://` URL never reaches the network layer at all), and
  `CURLOPT_PROTOCOLS_STR "https"` additionally prevents a malicious
  redirect chain from downgrading to plaintext.
- **Atomic**: writes to `<dest>.partial`, only `rename()`s to the real
  final path after every check below passes.
- **Resumable**: a pre-existing `.partial` file is resumed
  (`CURLOPT_RESUME_FROM_LARGE`), never re-downloaded from scratch.
- **Real file-size verification** against the catalog's own recorded
  size.
- **Real SHA-256 verification** against the catalog's own recorded
  digest — computed by shelling out to `sha256sum` (GNU coreutils, a
  Debian `Essential:yes` package, present on every real Debian-family
  install by definition — unlike `curl`, which this project already
  established it never assumes at runtime). If genuinely unavailable,
  this degrades to a disclosed, honest `checksum_verified: false`
  rather than a hard failure or a silent skip — the same graceful-
  degradation precedent `membrane doctor`'s own systemctl-availability
  check already established.
- **Real disk-space precheck** (`statvfs()`) with a 256 MiB safety
  margin above the exact byte count, before any byte is downloaded.
- **Never executes** a downloaded file — it is only ever opened for
  `stat()`/read (GGUF validation), written by `libcurl`'s own write
  callback, and eventually `mmap()`'d by llama.cpp's own model loader
  — no code path anywhere passes a downloaded path to `exec`/`system`.
- **Idempotent `install`**: a re-run against an already-correct,
  already-downloaded file skips the download and instead re-verifies
  its checksum (a real, if rare, same-size-but-corrupted file is still
  caught and refused, never silently accepted) — a real bug found and
  fixed while testing this end to end (an earlier version always
  re-downloaded, wasting real bandwidth and ~2 minutes on a real
  100 MiB file).
- **Cleanup on a real validation failure**: a size or checksum
  MISMATCH deletes the `.partial` file (retrying a resumable download
  against provably-wrong bytes would never converge); a plain transfer
  error (network drop, timeout) leaves it in place for a genuinely
  resumable retry.

## Storage

`~/.local/share/membrane/models/<family-name>/<upstream-filename>`
(honoring `XDG_DATA_HOME` first, same three-way fallback as the model
registry's own `models.json`, plus a `MEMBRANE_MODELS_INSTALL_DIR`
test-hook override) — organized by family, never a flat directory,
never inside the repository checkout.

`uninstall` refuses to delete a registry entry's file unless that
file's path is actually inside this managed directory (a real safety
boundary: a model registered via plain `membrane model add` against an
arbitrary user path is never touched by `uninstall` — use `remove`
for that, which only ever forgets the registry entry).

## Offline mode

Unchanged, and reinforced by this PR: real GGUF inference has zero
network dependency (`docs/server.md`'s own "Security scope"). A model
already installed (via `install` or plain `add`) needs no further
network access ever. `search`/`info` work fully offline too — only
`install` itself needs a real connection, and only for the one file
being downloaded.

## Real end-to-end evidence

See `results/model-catalog/validation.json` — a real install of
`smollm2-135m-instruct` (Q4_K_M, ~100 MiB) from its real Hugging Face
repo, real SHA-256 match, real GGUF validation, real idempotent re-run
(skips the download, re-verifies checksum), real corruption detection
(a same-size, tampered file is refused, never silently accepted), and
real uninstall (file + registry entry both removed).
