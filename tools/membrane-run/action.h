#ifndef MEMBRANE_RUN_ACTION_H
# define MEMBRANE_RUN_ACTION_H

# include <stddef.h>
# include <stdint.h>

# include "memory_intelligence.h"
# include "runtime_capabilities.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Milestone J (active control): the common action model behind
 * `membrane apply`.
 *
 *     plan -> observe -> advise -> apply
 *
 * Everything in this header is a PURE function of its inputs except
 * membrane_action_run(), which drives caller-supplied executor callbacks
 * (so the whole transaction engine is testable with a mock runtime, no
 * host, no model, no network).
 *
 * Rules this module enforces, each covered by test_action.c:
 *   - An action exists only for a real control primitive (the caller's
 *     membrane_apply_surface_t says which). Capability UNSUPPORTED /
 *     PARTIAL / UNKNOWN never yields an action -- the recommendation
 *     stays advisory with a stated reason. UNKNOWN fails closed.
 *   - A target value is NEVER guessed. A recommendation such as
 *     REDUCE_CONTEXT does not say "to what", so it yields an action only
 *     if an explicit user request (validated by Planner v2 by the
 *     caller) supplies the value. The one recommendation-derived target
 *     is UNLOAD_UNUSED_MODEL's candidate, chosen by a documented,
 *     deterministic rule (see membrane_action_pick_unload_candidate()).
 *   - Dry-run is the default; mutation needs the caller to say so.
 *   - Before mutating, live state is re-read and compared with the basis
 *     the plan was compiled against; any drift => STALE_PRECONDITION and
 *     nothing is touched.
 *   - Actions run in a fixed order. If one fails, later ones do not run,
 *     applied reversible ones are rolled back in reverse order, and the
 *     result says exactly what is and is not applied. No atomicity is
 *     claimed.
 */

# define MEMBRANE_ACTION_SCHEMA_VERSION	1

# define MEMBRANE_ACTION_MAX_ACTIONS	12
# define MEMBRANE_ACTION_MAX_UNAPPLIED	12
# define MEMBRANE_ACTION_MAX_RESIDENT	8
# define MEMBRANE_ACTION_NAME_MAX		128
# define MEMBRANE_ACTION_VALUE_MAX		64
# define MEMBRANE_ACTION_TEXT_MAX		240
# define MEMBRANE_ACTION_MAX_STATE		40
# define MEMBRANE_ACTION_STATE_KEY_MAX	48
# define MEMBRANE_ACTION_STATE_VAL_MAX	160

typedef enum e_membrane_action_code
{
	MEMBRANE_ACTION_UNLOAD_MODEL = 0,
	MEMBRANE_ACTION_SET_CONTEXT,
	MEMBRANE_ACTION_SET_KV_PRECISION,
	MEMBRANE_ACTION_SET_KV_PLACEMENT,
	MEMBRANE_ACTION_SET_GPU_LAYERS,
	MEMBRANE_ACTION_UNPIN_MODEL,
	MEMBRANE_ACTION_ACTIVATE_MODEL,
	MEMBRANE_ACTION_PIN_MODEL,
	MEMBRANE_ACTION_CODE_COUNT
}	membrane_action_code_t;
/* The enum order IS the execution order (deterministic): free memory
 * first, then persisted settings, then (un)pin/activate which act on the
 * resulting residency. */

const char	*membrane_action_code_name(membrane_action_code_t c);

typedef enum e_membrane_action_origin
{
	MEMBRANE_ACTION_ORIGIN_RECOMMENDATION = 0,
	MEMBRANE_ACTION_ORIGIN_USER_REQUEST
}	membrane_action_origin_t;

const char	*membrane_action_origin_name(membrane_action_origin_t o);

typedef enum e_membrane_action_risk
{
	MEMBRANE_ACTION_RISK_LOW = 0,
	MEMBRANE_ACTION_RISK_MEDIUM,
	MEMBRANE_ACTION_RISK_HIGH
}	membrane_action_risk_t;

const char	*membrane_action_risk_name(membrane_action_risk_t r);

/* What a runtime can ACTUALLY be controlled to do through MEMBRANE's
 * apply layer -- distinct from membrane_runtime_capabilities_t, which
 * describes the runtime's engine (native "context_control: SUPPORTED"
 * means membrane-run/plan flags exist, not that a running service is
 * controllable). One state per action code. */
typedef struct s_membrane_apply_surface
{
	membrane_capability_state_t	state[MEMBRANE_ACTION_CODE_COUNT];
	int							requires_reload[MEMBRANE_ACTION_CODE_COUNT];
	int							reversible[MEMBRANE_ACTION_CODE_COUNT];
	char						reason[MEMBRANE_ACTION_CODE_COUNT]
								[MEMBRANE_ACTION_TEXT_MAX];
}	membrane_apply_surface_t;

typedef struct s_membrane_action
{
	membrane_action_code_t		code;
	char						runtime_id[MEMBRANE_RUNTIME_ID_MAX];
	char						model[MEMBRANE_ACTION_NAME_MAX];	/* "" = n/a */
	char						requested[MEMBRANE_ACTION_VALUE_MAX];
	int							current_known;
	char						current[MEMBRANE_ACTION_VALUE_MAX];
	membrane_action_origin_t	origin;
	char						source_recommendation[MEMBRANE_INTEL_CODE_MAX];
	char						reason[MEMBRANE_ACTION_TEXT_MAX];
	membrane_capability_state_t	capability;
	int							executable;		/* always 1 for a listed action */
	membrane_action_risk_t		risk;
	int							reversible;
	int							rollback_known;
	char						rollback_value[MEMBRANE_ACTION_VALUE_MAX];
	int							requires_reload;
	int							restore_on_failure;	/* 1 = if THIS action
									 * fails, the engine also attempts its
									 * rollback after undoing earlier ones
									 * (a reload that unloaded the model
									 * first) */
	int							mutates_state;	/* always 1 */
	char						summary[MEMBRANE_ACTION_TEXT_MAX];
}	membrane_action_t;

/* A recommendation (or user request) that did NOT become an action, and
 * why -- advisory-only is a first-class, explained outcome. */
typedef struct s_membrane_action_unapplied
{
	char	source[MEMBRANE_INTEL_CODE_MAX];	/* recommendation code or
												 * the requested action */
	char	reason_code[40];
	char	text[MEMBRANE_ACTION_TEXT_MAX];
}	membrane_action_unapplied_t;

# define MEMBRANE_UNAPPLIED_NO_TARGET		"NO_DETERMINISTIC_TARGET"
# define MEMBRANE_UNAPPLIED_UNSUPPORTED	"RUNTIME_CANNOT_CONTROL"
# define MEMBRANE_UNAPPLIED_UNKNOWN_CAP	"CAPABILITY_UNKNOWN"
# define MEMBRANE_UNAPPLIED_NO_CANDIDATE	"NO_SAFE_CANDIDATE"
# define MEMBRANE_UNAPPLIED_NOT_ACTIONABLE	"NO_CONTROL_PRIMITIVE"
# define MEMBRANE_UNAPPLIED_REJECTED		"REQUEST_REJECTED"
# define MEMBRANE_UNAPPLIED_NOT_VALIDATED	"PLANNER_VALIDATION_FAILED"
# define MEMBRANE_UNAPPLIED_NO_CHANGE		"ALREADY_AS_REQUESTED"

/* A snapshot of live, material state: ordered key/value pairs. The only
 * thing the stale-plan check compares. Built from a membrane_apply_live_t
 * by membrane_apply_fingerprint() so executors never hand-roll keys. */
typedef struct s_membrane_apply_state
{
	char	key[MEMBRANE_ACTION_MAX_STATE][MEMBRANE_ACTION_STATE_KEY_MAX];
	char	value[MEMBRANE_ACTION_MAX_STATE][MEMBRANE_ACTION_STATE_VAL_MAX];
	size_t	count;
}	membrane_apply_state_t;

typedef struct s_membrane_apply_resident
{
	char		name[MEMBRANE_ACTION_NAME_MAX];
	int			activity_known;	/* 0 = the runtime does not say whether
								 * it is pinned / generating */
	int			pinned;
	int			generating;
	uint64_t	size_bytes;		/* estimated or reported; 0 = unknown */
	int			gpu_layers_known;	/* load-time GPU layers the runtime
								 * reports for this model */
	int			gpu_layers;
}	membrane_apply_resident_t;

/* Live control-relevant state of one runtime (what is true NOW). */
typedef struct s_membrane_apply_live
{
	int		runtime_reachable;
	char	default_model[MEMBRANE_ACTION_NAME_MAX];
	membrane_apply_resident_t	resident[MEMBRANE_ACTION_MAX_RESIDENT];
	size_t	resident_count;
	/* currently persisted overrides ("" + known=0 = automatic) */
	int		ctx_known;
	char	ctx[MEMBRANE_ACTION_VALUE_MAX];
	int		kv_precision_known;
	char	kv_precision[MEMBRANE_ACTION_VALUE_MAX];
	int		kv_placement_known;
	char	kv_placement[MEMBRANE_ACTION_VALUE_MAX];
	int		gpu_layers_known;
	char	gpu_layers[MEMBRANE_ACTION_VALUE_MAX];
}	membrane_apply_live_t;

void	membrane_apply_fingerprint(const membrane_apply_live_t *live,
			membrane_apply_state_t *out);

/* User-supplied explicit requests (J10). Empty string = not requested. */
typedef struct s_membrane_action_request
{
	char	runtime_id[MEMBRANE_RUNTIME_ID_MAX];
	char	context[MEMBRANE_ACTION_VALUE_MAX];
	char	kv_precision[MEMBRANE_ACTION_VALUE_MAX];
	char	kv_placement[MEMBRANE_ACTION_VALUE_MAX];
	char	gpu_layers[MEMBRANE_ACTION_VALUE_MAX];
	char	activate[MEMBRANE_ACTION_NAME_MAX];
	char	unload[MEMBRANE_ACTION_NAME_MAX];
	char	pin[MEMBRANE_ACTION_NAME_MAX];
	char	unpin[MEMBRANE_ACTION_NAME_MAX];
	char	model[MEMBRANE_ACTION_NAME_MAX];	/* target for --reload */
	int		reload;
}	membrane_action_request_t;

/* Outcome of the CALLER's Planner v2 check of the SET_* requests in a
 * request (the compiler never runs a planner itself). checked == 0 means
 * validation was not possible -> SET_* requests are refused. */
typedef struct s_membrane_action_validation
{
	int		checked;
	int		feasible;
	char	detail[MEMBRANE_ACTION_TEXT_MAX];
}	membrane_action_validation_t;

typedef struct s_membrane_action_plan
{
	int		schema_version;
	char	runtime_id[MEMBRANE_RUNTIME_ID_MAX];
	membrane_action_t			actions[MEMBRANE_ACTION_MAX_ACTIONS];
	size_t						action_count;
	membrane_action_unapplied_t	unapplied[MEMBRANE_ACTION_MAX_UNAPPLIED];
	size_t						unapplied_count;
	membrane_apply_state_t		basis;
	int		any_requires_reload;
}	membrane_action_plan_t;

/*
 * Deterministic UNLOAD_UNUSED_MODEL candidate: among resident models that
 * are NOT the configured default, whose activity is KNOWN, that are not
 * pinned and not generating -- the one with the largest known size
 * (frees the most), ties broken by name (strcmp). Returns the index or
 * -1; *why (optional, >= 240 bytes) says why none qualified.
 */
int	membrane_action_pick_unload_candidate(const membrane_apply_live_t *live,
		char *why, size_t why_size);

/*
 * The compiler. intel may be NULL when only explicit requests are being
 * compiled. Never throws away information: every recommendation that
 * does not become an action lands in unapplied[] with a reason.
 */
void	membrane_action_compile(const membrane_memory_intelligence_t *intel,
		const membrane_apply_surface_t *surface,
		const membrane_apply_live_t *live,
		const membrane_action_request_t *request,
		const membrane_action_validation_t *validation,
		membrane_action_plan_t *out);

/* Stale-plan protection (J7). Returns the number of drifted keys (0 =
 * fresh) and, when drift_out != NULL, a "; "-joined list of them. */
size_t	membrane_action_check_stale(const membrane_apply_state_t *basis,
		const membrane_apply_state_t *now, char *drift_out,
		size_t drift_size);

/* Every recommendation-origin action's source recommendation must still
 * be present in a freshly assembled intelligence result. Returns 1 iff
 * so; *missing (optional) gets the first absent code. */
int	membrane_action_recommendations_still_hold(
		const membrane_action_plan_t *plan,
		const membrane_memory_intelligence_t *fresh, char *missing,
		size_t missing_size);

/* ---- execution (J6) ------------------------------------------------ */

typedef enum e_membrane_apply_status
{
	MEMBRANE_APPLY_NOTHING_TO_APPLY = 0,
	MEMBRANE_APPLY_APPLIED,
	MEMBRANE_APPLY_PARTIALLY_APPLIED,
	MEMBRANE_APPLY_ROLLED_BACK,
	MEMBRANE_APPLY_ROLLBACK_FAILED,
	MEMBRANE_APPLY_REFUSED,
	MEMBRANE_APPLY_UNSUPPORTED,
	MEMBRANE_APPLY_STALE_PRECONDITION,
	MEMBRANE_APPLY_FAILED
}	membrane_apply_status_t;

const char	*membrane_apply_status_name(membrane_apply_status_t s);

typedef enum e_membrane_action_outcome
{
	MEMBRANE_OUTCOME_NOT_RUN = 0,
	MEMBRANE_OUTCOME_APPLIED,
	MEMBRANE_OUTCOME_FAILED,
	MEMBRANE_OUTCOME_ROLLED_BACK,
	MEMBRANE_OUTCOME_ROLLBACK_FAILED
}	membrane_action_outcome_t;

const char	*membrane_action_outcome_name(membrane_action_outcome_t o);

typedef struct s_membrane_action_executor
{
	void	*ctx;
	/* Fresh live state; return 0 on failure (-> REFUSED). */
	int		(*read_live)(void *ctx, membrane_apply_live_t *out);
	/* Perform one action; return 1 ok, 0 failed (err filled). */
	int		(*execute)(void *ctx, const membrane_action_t *a, char *err,
				size_t err_size);
	/* Undo one APPLIED, reversible action; 1 ok, 0 failed. */
	int		(*rollback)(void *ctx, const membrane_action_t *a, char *err,
				size_t err_size);
}	membrane_action_executor_t;

typedef struct s_membrane_action_result
{
	membrane_apply_status_t		status;
	membrane_action_outcome_t	outcome[MEMBRANE_ACTION_MAX_ACTIONS];
	char						error[MEMBRANE_ACTION_MAX_ACTIONS]
								[MEMBRANE_ACTION_TEXT_MAX];
	char						detail[MEMBRANE_ACTION_TEXT_MAX];
	size_t						applied_count;
}	membrane_action_result_t;

/*
 * The transaction engine. Re-reads live state through the executor and
 * refuses (STALE_PRECONDITION) on any drift from plan->basis BEFORE the
 * first mutation. Then executes in plan order; on the first failure it
 * stops and rolls back applied reversible actions in reverse order:
 *   all applied actions undone                -> ROLLED_BACK
 *   a rollback itself failed                  -> ROLLBACK_FAILED
 *   some applied actions are irreversible     -> PARTIALLY_APPLIED
 *   nothing had been applied                  -> FAILED
 */
void	membrane_action_run(const membrane_action_plan_t *plan,
		const membrane_action_executor_t *exec,
		membrane_action_result_t *out);

# ifdef __cplusplus
}
# endif

#endif
