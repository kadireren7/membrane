#ifndef MEMBRANE_RUN_ADAPTIVE_H
# define MEMBRANE_RUN_ADAPTIVE_H

# include <stddef.h>
# include <stdint.h>

# include "action.h"
# include "memory_intelligence.h"
# include "observation.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Milestone K (adaptive runtime): the PURE policy / history / decision
 * layer.
 *
 *     observe (bounded, repeated)  ->  sample + history
 *       -> pressure state machine (hysteresis)   -> decision
 *       -> [mode = apply]  the SAME J action compiler + engine
 *
 * Nothing here touches the machine, a clock or a thread: every function
 * takes `now_ms` explicitly, so cooldowns, hysteresis and anti-thrashing
 * are tested with synthetic timelines and no sleeps. The threaded
 * controller (tools/membrane/adaptive_controller.cpp) is a thin driver
 * around these functions.
 *
 * What this layer is NOT:
 *   - not a second planner (it reads Planner v2's figures only through
 *     the observation snapshot, exactly like memory intelligence);
 *   - not a second threshold system. WHEN pressure exists is decided by
 *     the I3 findings (measured provenance only). The only new numbers
 *     are the hysteresis EXIT thresholds, which are higher than I3's
 *     enter thresholds on purpose (threshold bouncing);
 *   - not an executor. It produces a membrane_action_request_t; the
 *     caller runs it through membrane_apply (J), so stale-plan checks,
 *     rollback and the audit log are the ones `membrane apply` uses.
 *
 * Adaptive control is OPT-IN: membrane_adaptive_policy_defaults() is mode
 * OFF, and even when enabled the default mode is `recommend`, which never
 * mutates.
 */

# define MEMBRANE_ADAPTIVE_SCHEMA_VERSION	1

# define MEMBRANE_ADAPTIVE_MAX_SAMPLES		256
# define MEMBRANE_ADAPTIVE_MAX_ACTIONS		32
# define MEMBRANE_ADAPTIVE_MAX_CHAIN		8
# define MEMBRANE_ADAPTIVE_TEXT_MAX			200
# define MEMBRANE_ADAPTIVE_MIN_INTERVAL_S	5
# define MEMBRANE_ADAPTIVE_MIN_CONTEXT		1024

typedef enum e_membrane_adaptive_mode
{
	MEMBRANE_ADAPTIVE_OFF = 0,		/* nothing adaptive happens */
	MEMBRANE_ADAPTIVE_OBSERVE,		/* samples / history only */
	MEMBRANE_ADAPTIVE_RECOMMEND,	/* + current adaptive recommendation,
									 * never mutates */
	MEMBRANE_ADAPTIVE_APPLY			/* + may execute policy-allowed actions */
}	membrane_adaptive_mode_t;

const char	*membrane_adaptive_mode_name(membrane_adaptive_mode_t m);
/* 1 and *out set on a known name, else 0. */
int			membrane_adaptive_mode_parse(const char *s,
				membrane_adaptive_mode_t *out);

/* Action classes a policy may allow (bit mask), least disruptive first.
 * Defaults allow only the first two: neither unloads anything. */
# define MEMBRANE_ADAPTIVE_CLASS_CONTEXT		0x01
# define MEMBRANE_ADAPTIVE_CLASS_KV_PRECISION	0x02
# define MEMBRANE_ADAPTIVE_CLASS_KV_PLACEMENT	0x04
# define MEMBRANE_ADAPTIVE_CLASS_GPU_LAYERS		0x08
# define MEMBRANE_ADAPTIVE_CLASS_UNLOAD			0x10
# define MEMBRANE_ADAPTIVE_CLASS_ALL			0x1f

/* "context,kv_precision" -> mask; unknown token => returns -1. */
int			membrane_adaptive_classes_parse(const char *csv);
/* mask -> "context,kv_precision" (comma separated, stable order). */
void		membrane_adaptive_classes_format(int mask, char *out, size_t n);

typedef enum e_membrane_pressure
{
	MEMBRANE_PRESSURE_UNKNOWN = 0,
	MEMBRANE_PRESSURE_HEALTHY,
	MEMBRANE_PRESSURE_WARNING,
	MEMBRANE_PRESSURE_CRITICAL
}	membrane_pressure_t;

const char	*membrane_pressure_name(membrane_pressure_t p);

typedef struct s_membrane_adaptive_policy
{
	int		enabled;				/* 0 => behaves as mode OFF */
	membrane_adaptive_mode_t	mode;
	char	runtime_scope[MEMBRANE_RUNTIME_ID_MAX];
	int		interval_s;				/* sampling interval */
	int		cooldown_s;				/* min gap between actions */
	int		sustained_samples;		/* consecutive pressured samples
									 * before it is "sustained" */
	int		recovery_samples;		/* consecutive recovered samples */
	/* Hysteresis EXIT thresholds (per mille of total). I3's enter
	 * thresholds are 150 (warning) and 50 (critical) per mille; recovery
	 * needs strictly more headroom than that. */
	int		warning_exit_permille;
	int		critical_exit_permille;
	uint64_t	ram_exit_abs_bytes;	/* host RAM also needs this much free */
	int		max_actions_per_window;
	int		window_s;
	int		allowed_classes;
	membrane_pressure_t	min_pressure;	/* act only at/above this */
	int		allow_estimate_only;	/* let an estimate-only signal act */
	int		auto_restore;			/* step back up after recovery */
	int		restore_after_s;
	int		thrash_reversals;		/* direction reversals => suppress */
	int		thrash_window_s;
	int		suppress_s;				/* how long thrashing suppresses */
	int		history_max_samples;	/* <= MEMBRANE_ADAPTIVE_MAX_SAMPLES */
	int		history_max_age_s;
}	membrane_adaptive_policy_t;

/* Conservative defaults: OFF; if only `enabled` is flipped, mode stays
 * OFF; interval 30 s, cooldown 600 s, 3 sustained samples, at most 2
 * actions per hour, classes {context, kv_precision}, min pressure
 * WARNING, estimate-only signals advise but never act, no auto restore. */
void		membrane_adaptive_policy_defaults(membrane_adaptive_policy_t *p);
/* "" if valid, else why not. Never silently clamps. */
const char	*membrane_adaptive_policy_validate(
				const membrane_adaptive_policy_t *p);
/* The mode actually in force: OFF unless enabled && mode != OFF. */
membrane_adaptive_mode_t	membrane_adaptive_effective_mode(
				const membrane_adaptive_policy_t *p);

/* One bounded, timestamped observation. Never holds prompt/content. */
typedef struct s_membrane_adaptive_sample
{
	int64_t				t_ms;
	int					valid;			/* 0 = runtime unobservable */
	membrane_pressure_t	measured;		/* from MEASURED I3 findings only */
	int					estimate_only;	/* an ESTIMATED finding >= notice */
	int					ram_permille;	/* -1 unknown */
	int					vram_permille;	/* -1 unknown */
	uint64_t			ram_free_bytes;
	int					ram_known;
	int					vram_pressured;	/* measured VRAM finding present */
	int					ram_pressured;	/* measured host-RAM finding */
	int					generating;		/* some resident model busy */
}	membrane_adaptive_sample_t;

/* Pure: derives a sample from an already-built snapshot + intelligence. */
void		membrane_adaptive_make_sample(
				const membrane_observation_snapshot_t *snap,
				const membrane_memory_intelligence_t *intel,
				const membrane_apply_live_t *live, int64_t now_ms,
				membrane_adaptive_sample_t *out);

typedef struct s_membrane_adaptive_action_rec
{
	int64_t		t_ms;
	char		code[32];
	char		model[MEMBRANE_ACTION_NAME_MAX];
	int			direction;			/* -1 down, +1 up, 0 other */
	int			manual;				/* 1 = user's own change, observed */
	int			ok;
	int			needs_reload;
	char		from[MEMBRANE_ACTION_VALUE_MAX];
	char		to[MEMBRANE_ACTION_VALUE_MAX];
}	membrane_adaptive_action_rec_t;

typedef struct s_membrane_adaptive_state
{
	membrane_adaptive_sample_t		samples[MEMBRANE_ADAPTIVE_MAX_SAMPLES];
	size_t							sample_count;	/* oldest first */
	int64_t							started_ms;

	membrane_pressure_t				pressure;		/* state machine */
	int								above_count;	/* consecutive pressured */
	int								recovered_count;/* consecutive recovered */
	int64_t							healthy_since_ms;

	int64_t							last_action_ms;	/* -1 none */
	membrane_adaptive_action_rec_t	actions[MEMBRANE_ADAPTIVE_MAX_ACTIONS];
	size_t							action_count;	/* oldest first */

	int64_t							suppressed_until_ms;	/* 0 = no */
	char							suppress_reason[40];

	int								baseline_valid;
	membrane_apply_state_t			baseline;
	int								pending_reload;	/* applied settings
										 * not yet in effect (need a load) */
	char							resident_sig[160];
}	membrane_adaptive_state_t;

void		membrane_adaptive_state_init(membrane_adaptive_state_t *s,
				int64_t now_ms);

/* Appends a sample (bounded by count AND age), runs the hysteresis state
 * machine, returns the pressure now in force. Samples with !valid are
 * stored but change nothing. */
membrane_pressure_t	membrane_adaptive_record_sample(
				const membrane_adaptive_policy_t *p,
				membrane_adaptive_state_t *s,
				const membrane_adaptive_sample_t *sample);

typedef enum e_membrane_adaptive_decision_kind
{
	MEMBRANE_DECISION_NONE = 0,		/* nothing to do (healthy / off) */
	MEMBRANE_DECISION_OBSERVED,		/* observe mode: sampled only */
	MEMBRANE_DECISION_TRANSIENT,	/* pressure not (yet) sustained */
	MEMBRANE_DECISION_COOLDOWN,
	MEMBRANE_DECISION_RATE_LIMITED,
	MEMBRANE_DECISION_SUPPRESSED,	/* anti-thrashing */
	MEMBRANE_DECISION_MANUAL_OVERRIDE,	/* user changed state: baseline
										 * adopted, nothing undone */
	MEMBRANE_DECISION_DEFERRED,		/* wanted, but not safe right now */
	MEMBRANE_DECISION_RECOMMEND,	/* a step is recommended */
	MEMBRANE_DECISION_ACT,			/* a step may be applied now */
	MEMBRANE_DECISION_NO_SAFE_STEP	/* pressure, but nothing allowed/safe:
									 * ask the user */
}	membrane_adaptive_decision_kind_t;

const char	*membrane_adaptive_decision_name(
				membrane_adaptive_decision_kind_t k);

# define MEMBRANE_ADAPT_REASON_TRANSIENT		"PRESSURE_TRANSIENT"
# define MEMBRANE_ADAPT_REASON_COOLDOWN		"COOLDOWN_ACTIVE"
# define MEMBRANE_ADAPT_REASON_RATE			"ACTION_RATE_LIMIT"
# define MEMBRANE_ADAPT_REASON_THRASH		"ADAPTATION_SUPPRESSED_THRASHING"
# define MEMBRANE_ADAPT_REASON_MANUAL		"MANUAL_OVERRIDE_DETECTED"
# define MEMBRANE_ADAPT_REASON_PENDING		"PENDING_RELOAD"
# define MEMBRANE_ADAPT_REASON_ESTIMATE		"ESTIMATE_ONLY_SIGNAL"
# define MEMBRANE_ADAPT_REASON_NO_STEP		"NO_SAFE_ADAPTATION"
# define MEMBRANE_ADAPT_REASON_CLASS		"ACTION_CLASS_NOT_ALLOWED"
# define MEMBRANE_ADAPT_REASON_BELOW_MIN	"BELOW_MINIMUM_PRESSURE"
# define MEMBRANE_ADAPT_REASON_UNOBSERVABLE	"RUNTIME_UNOBSERVABLE"
# define MEMBRANE_ADAPT_REASON_WARMUP		"WARMUP_NOT_COMPLETE"

typedef struct s_membrane_adaptive_chain_step
{
	char	stage[20];		/* observed|finding|recommendation|policy|
							 * action|result */
	char	text[MEMBRANE_ADAPTIVE_TEXT_MAX];
}	membrane_adaptive_chain_step_t;

typedef struct s_membrane_adaptive_decision
{
	membrane_adaptive_decision_kind_t	kind;
	membrane_pressure_t					pressure;
	char		reason[40];			/* stable code, "" when none */
	int			action_class;		/* MEMBRANE_ADAPTIVE_CLASS_* or 0 */
	int			needs_reload;
	int64_t		cooldown_remaining_s;
	int			is_restore;
	/* The step, expressed as the J request it would run. Meaningful for
	 * RECOMMEND and ACT only. */
	membrane_action_request_t	request;
	char		step_text[MEMBRANE_ADAPTIVE_TEXT_MAX];
	membrane_adaptive_chain_step_t	chain[MEMBRANE_ADAPTIVE_MAX_CHAIN];
	size_t		chain_count;
}	membrane_adaptive_decision_t;

/*
 * The decision. Pure; deterministic for identical inputs. Order of
 * checks (first match wins): mode off/observe, unobservable, manual-
 * override drift, thrash suppression, restore path (healthy only),
 * transient, minimum pressure, estimate-only gate, cooldown, rate limit,
 * pending-reload, then the least-disruptive-first step hierarchy.
 *
 * Hierarchy (docs/adaptive-runtime.md section 6 gives the rationale):
 *   1 context step-down        applies to the next request, no reload
 *   2 KV precision one step    native->q8->q5, next model load
 *   3 KV placement -> cpu      only under measured VRAM pressure
 *   4 GPU layers -25%          only under measured VRAM pressure
 *   5 unload an idle model     J's deterministic candidate rule
 *   6 NO_SAFE_STEP             ask the user; never a destructive guess
 * A class that is not allowed, unsupported by the runtime (surface), or
 * already at its floor is skipped, not escalated past silently: skipping
 * is recorded in the chain.
 *
 * `surface` is the runtime's apply surface (what is really controllable).
 */
void		membrane_adaptive_decide(const membrane_adaptive_policy_t *p,
				membrane_adaptive_state_t *s, int64_t now_ms,
				const membrane_observation_snapshot_t *snap,
				const membrane_memory_intelligence_t *intel,
				const membrane_apply_live_t *live,
				const membrane_apply_surface_t *surface,
				membrane_adaptive_decision_t *out);

/* Records that the controller executed (or attempted) a step: history,
 * last_action, pending-reload bookkeeping, and the new manual-override
 * baseline taken from `live_after`. */
void		membrane_adaptive_note_action(const membrane_adaptive_policy_t *p,
				membrane_adaptive_state_t *s, int64_t now_ms,
				const membrane_adaptive_decision_t *d, int ok,
				const membrane_apply_live_t *live_after);

/* Adopt `live` as the baseline without recording an action (startup,
 * policy change, user override accepted). */
void		membrane_adaptive_adopt_baseline(membrane_adaptive_state_t *s,
				const membrane_apply_live_t *live);

# ifdef __cplusplus
}
# endif

#endif
