# Adaptive runtime (Milestone K)

```
plan -> observe -> advise -> apply -> adapt
```

Adaptive runtime is **bounded, opt-in, explainable runtime adaptation**: a
controller that repeatedly observes the runtime, notices *sustained* memory
pressure, and -- only if you explicitly allowed it -- takes the least
disruptive validated step through the same pipeline `membrane apply` uses.

It is not an autonomous agent, not a hidden optimizer, and not a performance
claim. **It is off on a fresh install.** Nothing in this page happens until you
run `membrane adaptive enable`.

## 1. Quick start

```bash
membrane adaptive status                 # off by default
membrane adaptive enable                 # recommend-only: never changes anything
membrane adaptive status                 # see what it would suggest, and why
membrane adaptive enable --mode apply    # let it act (asks for consent)
membrane adaptive disable
```

## 2. Modes

| Mode | What happens |
|---|---|
| `off` (default) | Nothing. No sampling, no thread work. |
| `observe` | Samples and keeps a bounded history. Never suggests, never changes. |
| `recommend` | + produces the current adaptive suggestion (one concrete step and the reason chain). **Never mutates.** |
| `apply` | + may execute the suggested step, **only** within the action classes you allowed, through `membrane apply`'s pipeline. |

`enable` defaults to `recommend`. `apply` is never implied. Enabling `apply`
says plainly that runtime state may change, and needs an interactive `y`, or
`--yes` when not on a terminal (otherwise it exits `2` and saves nothing;
`--json` never prompts).

## 3. Policy and defaults

All conservative. Persisted by `membrane adaptive enable` in the `adaptive`
block of `server.json` (you never have to edit it by hand; the running service
re-reads it every tick, so no restart is needed):

| Setting | Default | Meaning |
|---|---|---|
| `enabled` | `false` | master switch |
| `mode` | `recommend` once enabled | see section 2 |
| `runtime` | `membrane-native` | `membrane-native` or `ollama` |
| `interval_s` | 30 (min 5) | sampling interval |
| `cooldown_s` | 600 | minimum gap between actions |
| `allow` | `context`, `kv_precision` | action classes it may use (`kv_placement`, `gpu_layers`, `unload` must be allowed explicitly) |
| `min_pressure` | `warning` | act only at/above this |
| `max_actions_per_window` | 2 per 3600 s | rate limit |
| `auto_restore` | `false` | step settings back up after recovery |

Internal, documented constants: pressure must be **sustained** for 3
consecutive samples (critical: 2); recovery needs 3 consecutive recovered
samples; history keeps at most 120 samples and one hour (hard cap 256); the
action history keeps the last 32 actions.

An invalid saved policy turns adaptive control **off** and `status` says why
(fail closed). A value out of range is rejected at `enable` time.

## 4. Pressure, hysteresis, and "measured dominates"

*When* there is pressure is decided by the Milestone I3 findings -- there is no
second threshold system. Only **measured** host-RAM / VRAM findings can
trigger an action (warning < 15 % headroom, critical < 5 %). A signal that is
only an estimate (e.g. Planner v2's KV estimate) may produce a recommendation
(`ESTIMATE_ONLY_SIGNAL`) but never acts, unless the policy explicitly allows
it.

Thresholds must not bounce, so every trigger has an enter and a stricter
exit:

| Transition | Enter | Exit (recovery) |
|---|---|---|
| healthy -> warning | I3 warning for 3 consecutive samples | -- |
| healthy -> critical | I3 critical for 2 consecutive samples | -- |
| warning -> healthy | -- | headroom >= **20 %** (and >= 1 GiB free RAM) for 3 consecutive samples |
| critical -> warning | -- | headroom >= **10 %** for 3 consecutive samples |

A host sitting at 17 % free is "healthy" to I3 yet stays in `warning` here: it
has not recovered meaningfully. Boundaries (199 vs 200 per mille, 99 vs 100)
are tested.

## 5. Cooldown, rate limit, anti-thrashing

* **Cooldown:** after any action (or a manual change it adopted) nothing else
  runs for `cooldown_s`. The status shows the seconds remaining.
* **Rate limit:** at most `max_actions_per_window` actions per window.
* **Anti-thrashing:** adaptive and manual changes go into a bounded action
  history. If the same setting (or model) would reverse direction
  `thrash_reversals` (3) times inside an hour -- context 8192 -> 4096 -> 8192
  -> 4096, GPU layers up/down, load/unload of one model -- adaptation is
  **suppressed** for 30 minutes with `ADAPTATION_SUPPRESSED_THRASHING`. It
  resumes after that window, or immediately after `membrane adaptive
  disable` + `enable`.

## 6. The action hierarchy (and why)

When pressure is sustained and nothing above blocks it, the controller picks
the **first viable** step; a class that is not allowed, not supported by the
runtime, or already at its floor is skipped and the skip is part of the reason
chain. Derived from what each action really does in the native runtime:

| # | Step | Target (deterministic) | Takes effect | Why here |
|---|---|---|---|---|
| 1 | context step-down | halve the current/planned context, rounded to 256, floor 1024 | next request | per-request context; no reload, nothing resident is disturbed |
| 2 | KV precision | native -> q8 -> q5 (one step) | next model load | no residency loss; lowers KV bytes |
| 3 | KV placement -> cpu | only under measured **VRAM** pressure | next model load | trades speed for VRAM |
| 4 | GPU layers -25 % | only under measured **VRAM** pressure, from the reported layers | next model load | trades speed for VRAM |
| 5 | unload an idle model | J's rule: not default, not pinned, not generating, activity known; largest first | immediately | loses a warm model: last |
| 6 | **ask the user** | -- | -- | `NO_SAFE_ADAPTATION`; never a destructive guess |

Targets are never improvised: step 1-4 values come from the rules above
(and are re-validated by Planner v2 before being written); step 5 uses J's
candidate rule. Steps that need a model load are marked **pending**; while one
is pending the controller **defers** (`PENDING_RELOAD`) instead of escalating
past a change that has not had a chance to work. It never reloads a model on
its own.

## 7. In-flight requests

The controller never changes anything beneath an active generation:

* the context override is read per request, so it applies to the *next*
  request;
* KV/placement/GPU-layer overrides are read at model load, so they apply to
  the next load (shown as `pending`);
* unload is refused by the server itself for a pinned or generating model, and
  the candidate rule excludes them anyway.

## 8. Architecture and lifecycle

The controller is **one thread inside `membrane serve`** -- not a separate
daemon. That process already owns the runtime state (resident models, pins,
in-flight generations) and the lifecycle (signals, graceful shutdown), so
there is one owner, no cross-process race on `server.json` or a model slot, no
extra unit to package, and nothing that can outlive the service. The thread
sleeps on a condition variable between ticks (never a busy loop), does nothing
while the mode is off, and is stopped and joined before the listener and the
models are closed.

```
tick: observe (membrane_advise_collect -- the I1/I3 providers)
   -> sample -> pressure state machine (hysteresis)
   -> decision (pure: tools/membrane-run/adaptive.[ch])
   -> [apply mode] membrane_apply_execute  == `membrane apply --yes`'s pipeline
        live re-read, Planner v2 check, stale-plan guards, rollback, audit
```

The adaptive layer has **no mutation path of its own**. Its audit records are
the ordinary J records with `"source": "adaptive"` and the reason chain.

`GET /membrane/v1/adaptive` (loopback admin namespace, read-only) exposes the
state for `membrane adaptive status`.

## 9. Restart, crash, and manual changes

* **Restart:** history and cooldowns live only in memory and are *not*
  restored. After a restart the controller re-observes, adopts the current
  state as its baseline, and must see pressure sustained again before it acts.
  Old actions are never replayed. The saved *policy* (mode) is restored, so a
  service you configured as `recommend` comes back as `recommend`.
* **Manual changes win.** If the context/KV/placement/GPU-layers override, the
  default model, or the set of resident models changes by something other than
  the controller, it records the change in its history, **adopts the new state
  as the baseline, starts a quiet period, and does not undo anything**
  (`MANUAL_OVERRIDE_DETECTED`). `auto_restore` never restores over a value the
  user has since changed.

## 10. Explainability

Every decision carries a chain `observed -> finding -> recommendation ->
policy -> action -> result`, in `membrane adaptive status` (compact) and in
`--json` (`reason_chain`), and in the audit record of every executed action.

## 11. Native vs. Ollama

* **membrane-native:** all classes available (subject to your `allow` list).
* **ollama:** adaptive control supports `observe` and `recommend`. `apply`
  can do nothing automatically: the only Ollama control is an explicit
  `keep_alive=0` unload, Ollama never says whether a model is mid-request, so
  there is no safe automatic candidate, and context/GPU/KV have no persistent
  control surface (see [active-control.md](active-control.md)). In `apply`
  mode on Ollama the outcome is `NO_SAFE_ADAPTATION` -- a recommendation, not
  an action. The capability surface decides this; nothing is pretended.

## 12. Data retention and privacy

History is in memory only: samples (timestamp, pressure level, headroom per
mille, a few booleans) and action records (code, model *name*, old/new value).
No prompts, no generated text, no request bodies, no tokens/credentials -- the
controller never sees them. Nothing is persisted except the policy block and
the J audit log (`actions.jsonl`, 0600, rotated at 1 MiB).

## 13. Did it help? (honest answer)

Deterministic **synthetic** timeline (`tools/membrane-run/test_adaptive.c`,
`test_timeline_simulation_benefit`): a modelled host where KV bytes scale with
context and another process spikes memory for two hours. With the real I3
thresholds the controller:

* cut ticks spent in a *critical* state from 120 to 1 (the single tick before
  the first sustained sample);
* raised mean headroom during the spike several-fold;
* took 2 actions (the rate limit), with zero direction reversals, then
  stopped.

Real-host check (this development machine, 5.6 GiB RAM, CPU backend,
SmolLM2-360M, `membrane serve` on an isolated config): with `adaptive enable
--mode recommend --interval 5` the controller sampled every 5 s inside the
service, reported `pressure: healthy`, `decision: none`, changed nothing, left
the service answering chat completions, and shut down cleanly with the
service. No memory pressure was induced on a real host (that would risk the
machine); the pressure behavior is therefore validated by the deterministic
timeline above, not on hardware.

That shows the **control logic** avoids an infeasible state and does not
oscillate. It is a simulation of memory arithmetic, **not** a measurement of
any real model's speed or quality, and no throughput claim is made. The
lightweight real-host check is in the milestone report. Techniques that showed
no useful effect are not enabled by default -- which is why `kv_placement`,
`gpu_layers` and `unload` are opt-in classes.

## 14. Limitations

* One runtime scope per controller; native `apply` is the deepest path.
* KV/placement/GPU-layer steps take effect only at the next model load; the
  controller never reloads on its own (`pending` is cleared when the resident
  set changes or empties).
* Tick cost is a full observation pass (host/GPU probes, planner metadata);
  hence the 30 s default.
* Cooldown and history reset on restart, by design.
* Thresholds are heuristics inherited from I3; they are tripwires, not
  guarantees.
