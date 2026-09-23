# Ollama runtime adapter (Milestone H2, read-only)

> **H2 does NOT send inference requests and does NOT change Ollama state.**
> The adapter discovers a local Ollama daemon, reads its version, lists its
> models, and reads one model's metadata. It never installs, starts or stops
> Ollama. It never pulls, deletes, creates, copies, loads or unloads models,
> never changes configuration or model parameters, and never applies a
> Planner v2 decision.

This builds on the runtime abstraction from Milestone H1
([runtime-abstraction.md](runtime-abstraction.md)).

## 1. Ollama API surface used

Audited against the upstream source and docs at **ollama/ollama v0.34.3**
(the latest release when H2 was written): `server/routes.go` (route
table), `api/types.go` (response structs), `docs/openapi.yaml`, `docs/api.md`,
`docs/api/errors.mdx`, `docs/api/authentication.mdx`, `envconfig/config.go`,
and `internal/modelref/modelref.go`.

| Call | Purpose | Response fields MEMBRANE reads |
|---|---|---|
| `GET /api/version` | discovery + health + version | `version` (string) |
| `GET /api/tags` | model inventory | `models[].name`, `.model`, `.size`, `.digest`, `.modified_at`, `.remote_host`, `.capabilities[]`, `.details.{format,family,families,parameter_size,quantization_level,context_length}` |
| `POST /api/show` with body `{"model": NAME}` | model metadata | `details.*` (as above), `model_info["general.architecture"]`, `model_info["general.parameter_count"]`, `model_info["<arch>.context_length"]`, `capabilities[]`, `parameters`, `modified_at`, `remote_host`; plus only whether `template`, `system` and `license` are present |

Nothing else is called. `/api/show` is a POST only because Ollama defines
it that way. For a local model it reads the local manifest and changes
nothing (`ShowHandler` → `GetModelInfo`). `verbose` is never sent, so the
large tokenizer arrays are not returned.

**Enforced allowlist.** `membrane_ollama_request_allowed()`
(`tools/membrane/runtime_ollama.h`) accepts exactly the three
method+path pairs above. The adapter's single HTTP helper checks it
before opening a socket. Everything else Ollama serves is unreachable
from this module, including:

- inference: `/api/generate`, `/api/chat`, `/api/embed`, `/api/embeddings`, and every `/v1/*` route
- mutation: `/api/pull`, `/api/push`, `/api/create`, `/api/copy`, `/api/delete`, `/api/blobs/*`
- cloud and account: `/api/me`, `/api/signout`, `/api/experimental/*`
- observation not needed in H2: `/api/ps`, `/api/status`

**Documented facts this adapter relies on:**

- Default endpoint: `127.0.0.1:11434` (`envconfig.Host()`; the OpenAPI
  `servers` entry is `http://localhost:11434`).
- Authentication: "The local API at `http://localhost:11434` does not
  require authentication." No credentials are ever sent.
- Errors: non-2xx status with a JSON body of the form `{"error": "..."}`. An unknown model on
  `/api/show` returns 404.
- **Cloud proxying:** for model names ending in `:cloud` or
  `:<tag>-cloud` (Ollama's `modelref.parseSourceSuffix`), `/api/show`
  forwards the request to ollama.com. The adapter mirrors that rule
  (`membrane_ollama_is_cloud_model_ref()`) and refuses such names **before
  any request is sent**. `/api/tags` reads local manifests only, but a
  pulled cloud stub appears there with `remote_host` set. MEMBRANE lists
  it with `"remote": true` and never contacts that host.

## 2. Discovery, availability and health

Discovery is one `GET /api/version` to the configured endpoint, with a
0.5 s connect timeout, a 2 s read timeout, a 64 KiB response cap, no
retries and no redirects. It never scans the network, probes other ports,
or shells out to an `ollama` binary. MEMBRANE does not look for the binary
at all, because "binary installed" and "daemon reachable" are different
facts. Only API reachability is reported.

| Probe outcome | `status` (availability) | `health` | Reason (human) |
|---|---|---|---|
| A. no HTTP response (refused, timeout, DNS) | `unavailable` | `unreachable` | `Ollama API is not reachable.` |
| B. 2xx + valid `{"version": "..."}` | `available` | `healthy` | – |
| C. non-2xx on `/api/version` (something else is listening) | `unavailable` | `incompatible` | `endpoint answered GET /api/version with HTTP N; not a compatible Ollama API.` |
| D. 2xx but malformed body / missing or invalid `version` / oversize | `unknown` | `unknown` | `endpoint answered, but its version response was malformed.` |
| invalid `MEMBRANE_OLLAMA_ENDPOINT` | `unavailable` | `not_probed` | `MEMBRANE_OLLAMA_ENDPOINT is invalid: ...` (nothing is probed) |

The adapter is always **listed**, because it is a known runtime.
"Adapter exists" never means "Ollama is running". `version` is filled
only in case B, with `version_provenance: "api_probe"`. It is never
guessed. Low-level socket errors are never printed.

## 3. Endpoint configuration

```
MEMBRANE_OLLAMA_ENDPOINT=http://127.0.0.1:11434   # the default
```

This follows the project's existing `MEMBRANE_*` environment-variable
convention (`MEMBRANE_MODELS_PATH`, `MEMBRANE_SERVER_CONFIG_PATH`, ...).
There is no new config file or config framework. Accepted form:
`http://HOST[:PORT][/]`, where HOST can be a bracketed IPv6 literal such as
`http://[::1]:11434`. The port defaults to 11434. The following are
rejected with a clear error:

- `https://` (this build has no TLS)
- other schemes
- user info (`user:pw@`)
- paths, queries and fragments
- unbracketed IPv6
- an invalid port

Ollama's own `OLLAMA_HOST` is deliberately **not** read. It is the
*server's bind address* (often `0.0.0.0`), not a client target.

**Execution mode reflects reality.** Only a loopback host (`localhost`,
`127.0.0.0/8`, `::1`) is `local_external`. An explicitly configured
non-loopback host (including `0.0.0.0`) is reported as `remote_external`,
never silently as local. The default is always loopback.

## 4. CLI

```
$ membrane runtime list                       # daemon not running
ID                 TYPE       STATUS       VERSION
membrane-native    native     available    -
ollama             external   unavailable  -

$ membrane runtime list                       # daemon reachable
ID                 TYPE       STATUS       VERSION
membrane-native    native     available    -
ollama             external   available    0.34.3

$ membrane runtime inspect ollama             # daemon not running
Runtime
  ID: ollama
  Type: external
  Execution mode: local_external
  Status: unavailable
  Health: unreachable
  Endpoint: http://127.0.0.1:11434

Reason
  Ollama API is not reachable.

Capabilities
  ... (matrix, see section 7) ...

Note
  Capabilities describe what this runtime's documented API exposes (static_contract).
  This build only READS its version, model list and model metadata:
  no inference, no model or configuration changes.

$ membrane runtime models ollama
NAME                     SIZE       FAMILY       PARAMS   QUANT
qwen2.5:7b               4.7 GB     qwen2        7.6B     Q4_K_M
llama3.2:3b              2.0 GB     llama        3.2B     Q4_K_M

$ membrane runtime model inspect ollama qwen2.5:7b
Model
  Runtime: ollama
  Name: qwen2.5:7b
  Architecture: qwen2
  Family: qwen2
  Format: gguf
  Parameter size: 7.6B
  Parameter count: 7615616512
  Quantization: Q4_K_M
  Context length (model maximum): 32768
  Modified: 2026-09-01T10:00:00+03:00
  Model capabilities: completion, tools
  Template: present   System prompt: none   License: present

Runtime default parameters
  stop                           "<|im_start|>"
  stop                           "<|im_end|>"
```

(The `models`/`model inspect` output above was produced by the real binary
against a loopback fixture server, not a real Ollama. See section 10.)

Every command accepts `--json`. Sizes use decimal units, as Ollama prints
them. The template, system prompt, license and Modelfile are never dumped:
only whether each is present is shown. Default parameters are capped at
1 KiB.

Exit codes (`product_cli.h`) and JSON error codes (`{"ok": false, "error": {"code", "message"}}`):

| Situation | JSON `error.code` | Exit code |
|---|---|---|
| unknown/reserved id, bad usage, `models membrane-native`, invalid model name, invalid endpoint | `CLI_ERROR` / `INVALID_MODEL_NAME` / `INVALID_ENDPOINT` | 2 `CLI_ERROR` |
| Ollama unreachable, non-2xx, malformed body, oversized body | `RUNTIME_UNAVAILABLE` / `RUNTIME_HTTP_ERROR` / `MALFORMED_RESPONSE` / `RESPONSE_TOO_LARGE` | 4 `RUNTIME_ERROR` |
| model not found; cloud model reference refused | `MODEL_NOT_FOUND` / `CLOUD_MODEL_REFUSED` | 3 `MODEL_ERROR` |

`membrane runtime models membrane-native` is intentionally a CLI error.
The native runtime's models *are* MEMBRANE's own registry
(`membrane model list`).

## 5. Model inventory schema

`membrane runtime models ollama --json`:

```json
{
  "schema_version": 1,
  "membrane_version": "1.0.0",
  "runtime_id": "ollama",
  "models": [
    {
      "runtime_id": "ollama",
      "runtime_model_id": "qwen2.5:7b",
      "display_name": "qwen2.5:7b",
      "family": "qwen2",
      "families": ["qwen2"],
      "format": "gguf",
      "parameter_size": "7.6B",
      "quantization": "Q4_K_M",
      "size_bytes": 4683087332,
      "digest": "845dbda0…",
      "modified_at": "2026-09-01T10:00:00+03:00",
      "context_length": 32768,
      "remote": false,
      "remote_host": null,
      "runtime_capabilities": ["completion", "tools"],
      "provenance": "api_probe"
    }
  ]
}
```

- Every value is as Ollama reported it. Anything not reported is `null`
  or `[]`: an empty string from Ollama becomes `null`, and nothing is
  inferred. `parameter_size` stays Ollama's own string and is not parsed.
- `context_length` is the model's trained maximum, and only when Ollama
  exposes it. It is **not** a configured or active context.
- `runtime_capabilities` lists the model's own features according to
  Ollama (`completion`, `tools`, `vision`, ...). This is unrelated to the
  runtime capability matrix.
- Models appear in Ollama's own order (most recently modified first), so
  the same response always produces the same JSON.
- The whole list is rejected with `MALFORMED_RESPONSE` if the top level is
  not `{"models": [...]}` or any entry is not an object with a name.
  Wrongly typed optional fields become unknown.

The C++ struct behind this (`membrane_external_model_t`,
`tools/membrane/runtime_adapter.h`) is backend-neutral so that a later
vLLM adapter can fill it too. **It is kept strictly separate from
MEMBRANE's own model registry:** nothing in H2 converts, merges, compares
or writes Ollama models into `registry_core`/`models.json`.

## 6. Model metadata

`membrane runtime model inspect ollama MODEL --json` returns
`{"schema_version", "membrane_version", "runtime_id", "model": {...}}`.
`model` has the inventory fields above plus:

- `architecture`: from `model_info["general.architecture"]`
- `parameter_count`: from `model_info["general.parameter_count"]`
- `parameters`: the runtime's default parameter text, capped at 1 KiB
- `parameters_truncated`
- `has_template`, `has_system`, `has_license`

`context_length` comes from `details.context_length`, or otherwise from
`model_info["<architecture>.context_length"]`. `/api/show` does not
return size or digest, so both are `null`: they are not copied over from
`/api/tags`. Request body: exactly `{"model": NAME}`, with no `verbose`
and no `options`. Response cap: 8 MiB. Read timeout: 10 s.

## 7. Capability matrix

This answers **"can MEMBRANE explicitly observe or control this through
Ollama's documented local HTTP API?"** It does not answer "does Ollama do
this internally". A setting that is only a server-side environment
variable, read once when `ollama serve` starts (`OLLAMA_KV_CACHE_TYPE`,
`OLLAMA_NUM_PARALLEL`, `OLLAMA_CONTEXT_LENGTH`, `CUDA_VISIBLE_DEVICES`,
...), is **not** an API control: MEMBRANE does not own that process.
Provenance is `static_contract`, meaning the matrix was derived from the
v0.34.3 docs and source rather than probed from the running daemon. H2
itself exercises only *enumeration* and *metadata*. Every other
`supported`/`partial` value describes the documented API that a future
adapter *could* use, not something H2 calls.

| Capability | Ollama | Basis |
|---|---|---|
| model_enumeration | supported | `GET /api/tags` (used by H2) |
| model_metadata | supported | `POST /api/show` (used by H2) |
| model_switch | supported | every request names its `model`; the scheduler loads on demand |
| model_load_unload | partial | only as documented side effects of `/api/generate` (empty prompt loads; `keep_alive: 0` unloads); the scheduler still evicts on its own (`OLLAMA_MAX_LOADED_MODELS`, memory pressure) |
| context_control | supported | per-request `options.num_ctx` (docs/api.md, OpenAPI, FAQ) |
| gpu_layer_control | partial | `options.num_gpu` is accepted (api/types.go, docs example) but its semantics are not specified in the OpenAPI document, and Ollama's memory fitting decides the final offload |
| quant_variant_control | partial | can only choose among variants already pulled (one tag = one quant); getting another variant is a pull, i.e. a mutation |
| kv_precision_control | unsupported | `OLLAMA_KV_CACHE_TYPE` is server-start env only |
| kv_placement_control | unsupported | no control exists |
| concurrency_control | unsupported | `OLLAMA_NUM_PARALLEL` is server-start env only |
| device_selection | unknown | `options.main_gpu` appears only in a docs example; device visibility is server-start env |
| memory_headroom_telemetry | unsupported | no free-memory or headroom endpoint |
| chat_completions | supported | `/api/chat`, `/v1/chat/completions` (**never called in H2**) |
| streaming | supported | NDJSON / SSE (**never called in H2**) |
| cancellation | unknown | no documented cancel endpoint or semantics |
| current_model | supported | `GET /api/ps` (not called in H2) |
| active_context | supported | `/api/ps` `context_length` per loaded model |
| ram_usage | partial | `/api/ps` `size` per loaded model; no host-wide view |
| vram_usage | partial | `/api/ps` `size_vram` per loaded model; no per-device breakdown |
| loaded_model_memory | partial | `/api/ps` `size` (scheduler's figure) |
| kv_cache_usage | unsupported | not exposed |
| live_kv_migration | unsupported | no KV export or import API |
| dynamic_reconfiguration | unsupported | changing `num_ctx` reloads the model; nothing is reconfigured live |

Under H1's negotiation rule (only `supported` satisfies a dimension), a
Planner v2 plan with decisions currently negotiates `unsupported` or
`partially_supported` against Ollama. That comparison is H3's job, not
H2's (see section 11).

## 8. HTTP behavior and security

- **HTTP stack:** reuses the cpp-httplib (v0.50.1) and nlohmann/json
  copies MEMBRANE already vendors. No libcurl, no new dependency.
- **Timeouts:** 0.5 s connect; read timeouts of 2 s (version), 5 s (tags)
  and 10 s (show), each with a hard overall cap.
- **Bounded responses:** 64 KiB for version, 16 MiB for tags, 8 MiB for
  show. The caps are enforced while reading, so an oversized response is
  never fully buffered.
- **Requests:** no retries, no keep-alive, no redirects followed, no
  proxy, no credentials, no telemetry, and no background monitoring.
- **Failures:** malformed JSON and non-2xx responses become structured
  errors. Nothing crashes, and no raw socket text is printed. Ollama's
  own `error` string is shown only as a short, printable excerpt of at
  most 200 characters.
- **Default target:** loopback only. No LAN scan and no port probing.
  Cloud model references are refused locally before any request.

## 9. Read-only guarantee

- **The allowlist** (section 1) is enforced in code and tested.
- **Proof in tests:** `test_runtime_ollama` runs every command against a
  mock that logs *every* request, including requests to unmatched routes.
  It asserts that each request is on the allowlist and that no inference
  or mutating route was reached. The mock registers trap handlers for all
  of those routes.
- **No file writes:** the same test points `MEMBRANE_MODELS_PATH`,
  `MEMBRANE_SERVER_CONFIG_PATH` and `MEMBRANE_SYSTEMD_USER_DIR` at an
  empty directory and asserts it is still empty afterwards. No registry,
  config or service file is written.
- **Planner v2 is untouched:** no `membrane_plan_t` mapping and no
  negotiation against Ollama.

## 10. Tests and real smoke

`tools/membrane/test_runtime_ollama.cpp` needs no Ollama. It starts an
in-process `httplib::Server` on an ephemeral 127.0.0.1 port, serving
fixtures shaped like v0.34.3's documented responses, and it always pins
`MEMBRANE_OLLAMA_ENDPOINT` to that server or to a closed port, so a real
daemon on the test host is never contacted. It covers:

- unavailable
- healthy / version
- model list, including deterministic JSON and a remote stub
- malformed / incompatible / oversized version response
- a bounded timeout
- malformed model list and non-2xx
- `/api/show` metadata, 404, and malformed responses
- cloud references refused without any request
- the capability matrix
- endpoint parsing and override
- read-only / no-inference behavior
- membrane-native regression

`test_runtime_cmd` was updated: list now shows 2 runtimes, and `vllm` is
the remaining reserved id.

**Real smoke:** not run. No Ollama binary or daemon was present on the
development host (`127.0.0.1:11434` refused the connection), and H2 does
not install one. Against that host, `membrane runtime list` shows
`ollama external unavailable`, `inspect ollama` shows the Reason block,
and `models ollama` exits 4 with `RUNTIME_UNAVAILABLE`.

## 11. Limitations and what is deferred to H3

**Limitations:**

- The capability matrix is a static contract for Ollama v0.34.3. It is
  not version-gated, and it is not probed per daemon.
- There is no minimum-version check. "Incompatible" means only that
  `/api/version` returned a non-2xx status.
- `/api/ps` (loaded models, VRAM) is not read, even though it is
  read-only. H2 was scoped to discovery, inventory and metadata.
- Only plain `http://` endpoints are supported: no TLS and no auth.
- The model name passed to `model inspect` is sent as given. There is no
  local canonicalization beyond whitespace/control rejection and the
  cloud-suffix refusal.

**Deferred to H3:** capability-aware planning. That means mapping or
comparing Ollama inventory against MEMBRANE's registry, building
`membrane_plan_t` for an Ollama model, running negotiation against the
Ollama matrix, and any recommendation. Any control (per-request
`num_ctx`/`num_gpu`, load/unload) or inference call remains out of scope
until explicitly requested.
