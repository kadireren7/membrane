# Active control (Milestone J)

```
plan  ->  observe  ->  advise  ->  apply
read      read         read        MUTATES (explicit, bounded, reversible
                                   where realistically possible)
```

`membrane apply` is the first MEMBRANE command that changes how a runtime
behaves because of what `observe`/`advise` found. Everything before it is
read-only and stays so. This page is the contract for the one step that is
not.

> **Nothing is applied unless you say so.** `membrane apply` is a dry run by
> default. `membrane advise` never applies anything, ever.

## 1. Read-only vs. mutating commands

| Command | Changes anything? |
|---|---|
| `membrane plan`, `observe`, `advise`, `runtime ...` | No (unchanged by this milestone) |
| `membrane apply` | **No** -- dry run (default) |
| `membrane apply --yes` / confirmed `y` on a TTY | **Yes** -- only the actions printed in the plan |

## 2. Dry run, consent, and `--yes`

```bash
membrane apply                          # dry run: recommendations -> proposed actions
membrane apply --runtime ollama         # same, for Ollama
membrane apply --context 4096           # still a dry run; shows the SET_CONTEXT action
membrane apply --context 4096 --yes     # actually applies it
membrane apply --json                   # machine-readable
```

* Default: show the recommendation, the proposed action(s), what cannot be
  applied and why, the risk of each, and **change nothing**.
* On a terminal, if there is something to apply, MEMBRANE prints the plan and
  asks `Apply these changes? [y/N]` (default no).
* `--yes` / `-y` consents without a prompt (required for automation).
* **Non-interactive without `--yes` fails closed:** a request that selects an
  action (`--context`, `--unload`, ...) with no TTY and no `--yes` exits `2`,
  reports `applied: false`, and changes nothing. `--json` never prompts.
* A plain `membrane apply` with no action-selecting flag is a pure
  preview and exits `0`.

JSON always distinguishes the three facts people confuse:

```json
{ "dry_run": true, "mutates_state": false, "applied": false, ... }
```

`mutates_state` is true only when a mutation was actually attempted;
`applied` is true only when every action succeeded (`status: "applied"`).
The JSON contains no timestamps, so identical inputs give byte-identical
output.

## 3. The action model

Every proposed change is a `membrane_action_t`
(`tools/membrane-run/action.h`), never free text:

| Field | Meaning |
|---|---|
| `code` | `SET_CONTEXT`, `SET_GPU_LAYERS`, `SET_KV_PRECISION`, `SET_KV_PLACEMENT`, `ACTIVATE_MODEL`, `UNLOAD_MODEL`, `PIN_MODEL`, `UNPIN_MODEL` |
| `runtime`, `model` | what it targets |
| `requested`, `current` | new value; present value (null if unknown) |
| `origin`, `source_recommendation` | `recommendation` or `user_request`, and the recommendation it answers |
| `reason` | why (Planner v2's verdict, or the unload-candidate rule) |
| `capability`, `executable` | always `supported` / `true` for a listed action |
| `risk` | `low` / `medium` / `high` |
| `reversible`, `rollback` | whether and how it can be undone |
| `requires_reload` | takes effect only when the model is next loaded |
| `mutates_state` | always `true` |
| `summary` | one human line |

**An action exists only for a real control primitive.** Anything else is
reported under `unapplied[]` with a stable reason code:

| Reason | Meaning |
|---|---|
| `NO_DETERMINISTIC_TARGET` | the recommendation names a direction ("reduce context"), not a value |
| `RUNTIME_CANNOT_CONTROL` | the runtime has no control primitive for it |
| `CAPABILITY_UNKNOWN` | unknown is treated as "no" (fail closed) |
| `NO_SAFE_CANDIDATE` | e.g. every resident model is pinned, generating, the default, or of unknown activity |
| `NO_CONTROL_PRIMITIVE` | e.g. `USE_LOWER_MEMORY_VARIANT`, `SWITCH_TO_RUNTIME_WITH_DEEPER_CONTROL` |
| `REQUEST_REJECTED` | bad value, model not resident, pinned/generating |
| `PLANNER_VALIDATION_FAILED` | Planner v2 said it does not fit (or could not be run) |
| `ALREADY_AS_REQUESTED` | nothing to change |

## 4. Recommendation -> action

| Recommendation | Becomes | When |
|---|---|---|
| `REDUCE_CONTEXT` | `SET_CONTEXT` | only if you pass `--context N` (validated by Planner v2) |
| `USE_LOWER_KV_PRECISION` | `SET_KV_PRECISION` | only with `--kv native\|q8\|q5` |
| `CHANGE_KV_PLACEMENT` | `SET_KV_PLACEMENT` | only with `--kv-placement` |
| `REDUCE_GPU_LAYERS` | `SET_GPU_LAYERS` | only with `--gpu-layers` |
| `UNLOAD_UNUSED_MODEL` | `UNLOAD_MODEL` | when a safe candidate exists (below) |
| `USE_LOWER_MEMORY_VARIANT`, `SWITCH_TO_RUNTIME_WITH_DEEPER_CONTROL` | -- | advisory only |
| `NO_ACTION_NEEDED` | -- | nothing |

### MEMBRANE never guesses a target value

`REDUCE_CONTEXT` does not say *to what*. Planner v2 plans the best fit for the
current host; it does not name a "reduced" figure, and no other documented
policy does either. So automatic advice **never** produces a `SET_*` action.
You supply the value; MEMBRANE re-checks it with the same Planner v2
resolver `membrane plan` uses -- with the *combined* result of what you
asked for and what is already set -- and refuses the action if Planner v2
says it is not feasible, or if the check cannot be run at all. The one
recommendation-derived target is the unload candidate:

> Among resident models that are **not** the configured default model, whose
> activity the runtime **reports**, that are **not pinned** and **not
> generating**: the one with the largest known size (frees the most);
> ties are broken by name.

## 5. Runtime differences

### `membrane-native`

The native service decides context, GPU layers and KV settings automatically
each time it loads a model or serves a request. Before this milestone there
was no way to change that without restarting with different flags, so
`apply` adds the smallest real mechanism: an optional `overrides` block in
`server.json` that the service re-reads (by mtime) at every model load and
request.

```json
{ "schema_version": 1, "...": "...",
  "overrides": { "context": 4096, "kv_precision": "q8",
                 "kv_placement": "cpu", "gpu_layers": 20 } }
```

| Action | How it is applied | Takes effect |
|---|---|---|
| `SET_CONTEXT` | `overrides.context`: a fixed context per request, and the load-time context | next request |
| `SET_KV_PRECISION` | `overrides.kv_precision` (`native`, `q8`, `q5`) | next model load |
| `SET_KV_PLACEMENT` | `overrides.kv_placement` | next model load |
| `SET_GPU_LAYERS` | `overrides.gpu_layers` (`all`, `auto`, `N`) | next model load |
| `UNLOAD_MODEL` | new loopback route `POST /membrane/v1/models/unload` (reuses the slot machinery) | immediately |
| `ACTIVATE_MODEL` | existing `POST /membrane/v1/models/activate` | immediately |
| `PIN_MODEL` / `UNPIN_MODEL` | existing `.../pin`, `.../unpin` | immediately |

`auto` clears an override (back to the pre-milestone automatic behavior).
Settings marked "next model load" do **not** change a model that is
already resident. Say so up front: the plan prints "takes effect when the
model is next loaded". `--reload [--model NAME]` adds an explicit reload
(unload, then activate) so they take effect now. A reload is only offered for
an unpinned, idle model, and if the reload fails the engine restores the
previous settings and reloads the model again.

A fixed `context` means a request whose prompt does not fit is rejected
rather than silently getting a bigger context -- that is what you asked for.

The unload route **refuses, never waits**: a pinned model (`409 MODEL_PINNED`)
or one with a generation in flight (`409 MODEL_BUSY`) is left alone.
`decode_inflight` is only incremented under the slot mutex the unload holds,
so no generation can start between the check and the close.

### `ollama`

Audited against upstream **v0.34.3** (`docs/faq.mdx`, `docs/api.md`,
`api/types.go`). The H/I security posture is unchanged: an allowlist, bounded
single requests, no redirects, loopback only for any mutation.

| Control | Result |
|---|---|
| unload a model | **Implemented**: `POST /api/generate {"model": NAME, "keep_alive": 0}` -- upstream's documented unload request. It carries no prompt, so it is not inference. The allowlist admits `/api/generate` only for that exact body. Never an automatic candidate (Ollama does not say whether a model is mid-request, and what it does to an in-flight request is undocumented); only `--unload NAME`. The effect is verified with `GET /api/ps`. Not reversible by MEMBRANE (reloading needs an inference-endpoint request). |
| context (`num_ctx`), GPU layers (`num_gpu`) | **Not implemented.** Request-level options only; there is no persistent setting to change. Applying them would mean sending an inference request, which `apply` never does. |
| KV precision | **Not implemented.** `OLLAMA_KV_CACHE_TYPE` is a server start-time environment variable. |
| KV placement | No control exists. |
| pin / unpin | No such concept (`keep_alive` is per request). |
| load / activate | Needs an inference-endpoint request -- refused as a design rule. |

Each shows up in the plan as `RUNTIME_CANNOT_CONTROL` with the reason, and no
action object. That is the intended outcome, not a gap.

## 6. Stale-plan protection

Advice is a snapshot. Before the first mutation `apply`:

1. re-collects the advice and checks every recommendation-derived action's
   recommendation is **still present** (pressure disappeared => refuse);
2. re-reads the runtime's live state and compares it with the state the plan
   was compiled against: runtime reachable, default model, current overrides,
   the set of resident models and each one's pinned / generating status.
   Any difference => refuse.

A refusal is `status: "stale_precondition"`, **nothing is changed**, and the
message says to rerun `membrane advise` / `membrane apply`. The same check
runs after you answer the confirmation prompt, so a generation that starts
while you are deciding stops the plan.

## 7. Transactions and rollback

Actions run in a fixed order (unload, then settings, then unpin, activate,
pin). If action N fails, N+1 does not run; already-applied **reversible**
actions are rolled back newest-first. No atomicity is claimed. The result
status is one of:

| Status | Meaning |
|---|---|
| `applied` | everything succeeded |
| `partially_applied` | something failed and at least one applied action could not (or cannot) be undone, e.g. `ACTIVATE_MODEL` (loading may have evicted another model) |
| `rolled_back` | something failed; everything applied was undone |
| `rollback_failed` | something failed and an undo itself failed -- listed per action |
| `failed` | the first action failed; nothing had been applied |
| `refused` | runtime unreachable / not loopback / state unreadable; nothing changed |
| `stale_precondition` | see section 6 |
| `unsupported`, `nothing_to_apply` | nothing executable |

Rollbacks: settings are restored to the captured previous value; pin <->
unpin; an unload is undone by reloading the model (which can fail under the
same memory pressure that motivated it -- hence `rollback_failed` exists);
`ACTIVATE_MODEL` is not reversible. Exit code is `0` only for `applied`
(and dry runs / nothing to apply).

## 8. Audit log

Every executed apply (not dry runs, not refusals before any change) appends
one JSON line to `$XDG_STATE_HOME/membrane/actions.jsonl`
(`~/.local/state/membrane/actions.jsonl`; `MEMBRANE_APPLY_AUDIT_PATH`
overrides): timestamp, runtime, per-action `before` / `requested` / `after`,
outcome, error and reason codes, and the overall status. It never contains
prompt text, generated content or secrets (none ever reach this command).
The file is `0600` and rotates once at 1 MiB to `actions.jsonl.1` (one
generation kept). A failure to write it is reported in `notes`, not fatal.

## 9. Security

* Mutation is local only: `apply` refuses a non-loopback `listen_address` in
  `server.json` and a non-loopback `MEMBRANE_OLLAMA_ENDPOINT`.
* The new unload route lives in the same loopback admin namespace as
  activate/pin. No CORS headers, no wildcard origin, no new listener, no
  credentials, no cloud.
* `server.json` overrides are validated on load; an invalid block fails the
  whole load rather than being half-honored.

## 10. Limitations

* Settings other than context apply at the next model load; resident models
  are not reconfigured live (the runtime cannot -- `dynamic_reconfiguration`
  is `unsupported`).
* `overrides` are service-wide, not per model.
* Pin state is per resident slot and is lost when a model is unloaded.
* The native "generating" signal is the server's own `evictable` flag; a
  request can still begin between your last look and the call, which is why
  the server re-checks under its own lock.
* Ollama: unload only; see section 5.
* Planner v2 is unchanged and is only consulted, never modified.
