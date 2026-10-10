#ifndef MEMBRANE_RUN_ORCHESTRATOR_H
# define MEMBRANE_RUN_ORCHESTRATOR_H

# include <stddef.h>
# include <stdint.h>

# include "runtime_capabilities.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Milestone L: runtime orchestration -- "which runtime should serve this
 * model on this machine, and why?" -- as a PURE, deterministic function.
 *
 *     workload intent + hardware topology
 *       + one membrane_orch_runtime_input_t per runtime
 *       -> membrane_orchestrate() -> selected candidate, alternatives,
 *          reason codes, confidence, control limitations
 *
 * Llama-free, no I/O, no clock. Every input is plain data a caller gathered
 * from the runtime adapters (runtime_adapter.h) -- the orchestrator has no
 * per-runtime branches: native, Ollama and vLLM differ only in the data and
 * capability matrix they bring.
 *
 * What it is NOT: no opaque scorer, no ML, no benchmark-backed ranking, no
 * "MEMBRANE owns it so prefer it". Ranking is a fixed lexicographic chain
 * of facts (docs/orchestration.md), ending in a stable tie-break. A
 * candidate is never marked feasible without evidence: native gets a
 * Planner v2 plan; an external runtime gets only what it itself reports
 * (resident now, or a reported size compared with measured memory), and
 * "unknown" stays "unknown".
 *
 * Intents implemented (each has a real, different policy):
 *   general        evidence of feasibility, then already-resident, then
 *                  observation quality (default; conservative)
 *   interactive    already-resident first (no load latency), then evidence
 *   memory_saving  smallest KNOWN memory footprint first
 *   long_context   largest KNOWN maximum context first
 * `throughput` is deliberately NOT implemented: there is no capability or
 * measurement MEMBRANE has that distinguishes runtimes by throughput
 * without inventing a benchmark claim.
 */

# define MEMBRANE_ORCH_SCHEMA_VERSION	1
# define MEMBRANE_ORCH_MAX_RUNTIMES		8
# define MEMBRANE_ORCH_MAX_DEVICES		16
# define MEMBRANE_ORCH_MAX_REASONS		10
# define MEMBRANE_ORCH_TEXT_MAX			200

typedef enum e_membrane_orch_intent
{
	MEMBRANE_INTENT_GENERAL = 0,
	MEMBRANE_INTENT_INTERACTIVE,
	MEMBRANE_INTENT_MEMORY_SAVING,
	MEMBRANE_INTENT_LONG_CONTEXT
}	membrane_orch_intent_t;

const char	*membrane_orch_intent_name(membrane_orch_intent_t i);
int			membrane_orch_intent_parse(const char *s,
				membrane_orch_intent_t *out);

/* ---- hardware topology (L8/L11) ------------------------------------- */

typedef enum e_membrane_orch_device_kind
{
	MEMBRANE_DEV_KIND_CPU = 0,		/* the host CPU "device" */
	MEMBRANE_DEV_KIND_DISCRETE,		/* own dedicated memory (VRAM) */
	MEMBRANE_DEV_KIND_UMA,			/* integrated: shares host RAM */
	MEMBRANE_DEV_KIND_UNKNOWN
}	membrane_orch_device_kind_t;

const char	*membrane_orch_device_kind_name(membrane_orch_device_kind_t k);

typedef struct s_membrane_orch_device
{
	char		id[16];				/* stable: "gpu0", "gpu1", ... in
									 * enumeration order */
	char		backend[32];
	char		name[128];
	membrane_orch_device_kind_t	kind;
	uint64_t	total_bytes;		/* 0 = unknown */
	int			free_known;			/* measured free memory known */
	uint64_t	free_bytes;			/* meaningful iff free_known */
}	membrane_orch_device_t;

typedef struct s_membrane_topology
{
	int			ram_known;
	uint64_t	ram_total_bytes;
	uint64_t	ram_available_bytes;
	membrane_orch_device_t	devices[MEMBRANE_ORCH_MAX_DEVICES];
	size_t		device_count;		/* ALL enumerated GPU devices -- never
									 * collapsed to "the first one" */
}	membrane_topology_t;

size_t	membrane_topology_gpu_count(const membrane_topology_t *t);
/* Deterministic device pick for a footprint: among discrete GPUs whose
 * measured free memory covers `need_bytes`, the one with the most free
 * memory (ties: lowest index). Returns the index or -1 (no discrete GPU
 * fits, or free memory unknown: never guessed). */
int		membrane_topology_pick_device(const membrane_topology_t *t,
			uint64_t need_bytes);

/* ---- per-runtime input ------------------------------------------------ */

typedef struct s_membrane_orch_runtime_input
{
	char	runtime_id[MEMBRANE_RUNTIME_ID_MAX];
	char	display_name[MEMBRANE_RUNTIME_NAME_MAX];
	membrane_runtime_availability_t	availability;
	membrane_runtime_health_t		health;
	char	unavailable_reason[MEMBRANE_RUNTIME_REASON_MAX];
	membrane_runtime_capabilities_t	caps;

	/* the requested model, AS THIS RUNTIME KNOWS IT */
	int		model_present;
	char	model_id[128];
	int		resident_known;			/* does the runtime say what is loaded? */
	int		resident;				/* the model is loaded right now */
	int		size_known;
	uint64_t	size_bytes;			/* weights (+ whatever the runtime
									 * reports); NOT a KV estimate */
	int		context_max_known;
	uint64_t	context_max;

	/* Planner v2 evidence -- only a runtime MEMBRANE can plan for (native)
	 * may set plan_valid */
	int		plan_valid;
	int		plan_feasible;
	uint64_t	plan_context;
	int		plan_gpu_layers;		/* -1 unknown/auto */
	char	plan_kv_precision[16];
	char	plan_kv_placement[16];
	int		plan_mem_known;
	uint64_t	plan_host_required_bytes;
}	membrane_orch_runtime_input_t;

/* ---- result ------------------------------------------------------------ */

typedef enum e_membrane_orch_evidence
{
	MEMBRANE_EVIDENCE_INFEASIBLE = 0,
	MEMBRANE_EVIDENCE_UNKNOWN,		/* nothing supports "it fits" */
	MEMBRANE_EVIDENCE_ESTIMATE,		/* reported size fits measured memory;
									 * weights only, no KV accounting */
	MEMBRANE_EVIDENCE_RESIDENT,		/* the runtime reports it loaded */
	MEMBRANE_EVIDENCE_PLANNED		/* Planner v2 found a feasible plan */
}	membrane_orch_evidence_t;

const char	*membrane_orch_evidence_name(membrane_orch_evidence_t e);

# define MEMBRANE_ORCH_REASON_ONLY_CANDIDATE		"ONLY_CANDIDATE"
# define MEMBRANE_ORCH_REASON_PLANNED				"PLANNER_V2_FEASIBLE"
# define MEMBRANE_ORCH_REASON_RESIDENT				"MODEL_RESIDENT_NOW"
# define MEMBRANE_ORCH_REASON_STRONGER_EVIDENCE		"STRONGER_FEASIBILITY_EVIDENCE"
# define MEMBRANE_ORCH_REASON_OBSERVATION			"BETTER_OBSERVATION_QUALITY"
# define MEMBRANE_ORCH_REASON_SMALLEST_MEMORY		"SMALLEST_KNOWN_MEMORY"
# define MEMBRANE_ORCH_REASON_LARGEST_CONTEXT		"LARGEST_KNOWN_CONTEXT"
# define MEMBRANE_ORCH_REASON_MORE_CONTROL			"MORE_MEMORY_REDUCING_CONTROLS"
# define MEMBRANE_ORCH_REASON_TIEBREAK				"TIE_BROKEN_BY_RUNTIME_ID"
# define MEMBRANE_ORCH_REASON_UNAVAILABLE			"RUNTIME_UNAVAILABLE"
# define MEMBRANE_ORCH_REASON_NO_MODEL				"MODEL_NOT_ON_RUNTIME"
# define MEMBRANE_ORCH_REASON_INFEASIBLE			"INFEASIBLE_ON_THIS_HOST"
# define MEMBRANE_ORCH_REASON_UNKNOWN_FIT			"FEASIBILITY_UNKNOWN"
# define MEMBRANE_ORCH_REASON_NO_CANDIDATE			"NO_VIABLE_RUNTIME"

typedef struct s_membrane_orch_candidate
{
	char	runtime_id[MEMBRANE_RUNTIME_ID_MAX];
	char	display_name[MEMBRANE_RUNTIME_NAME_MAX];
	char	model_id[128];
	int		viable;						/* 0 = excluded (see exclusion) */
	char	exclusion_code[40];
	char	exclusion_text[MEMBRANE_ORCH_TEXT_MAX];

	membrane_orch_evidence_t	evidence;
	char	evidence_text[MEMBRANE_ORCH_TEXT_MAX];
	int		resident;
	/* configuration this candidate would run with -- only the dimensions
	 * the runtime actually exposes are set (others stay "unknown"/-1) */
	int		context_known;
	uint64_t	context;
	int		gpu_layers;					/* -1 = not applicable/unknown */
	char	kv_precision[16];			/* "" = not applicable/unknown */
	char	kv_placement[16];
	int		mem_known;
	uint64_t	mem_bytes;
	char	mem_provenance[24];			/* planner_v2|runtime_reported|"" */
	int		context_max_known;
	uint64_t	context_max;
	int		control_count;				/* SUPPORTED control capabilities
										 * (context, gpu layers, kv
										 * precision, kv placement) */
	int		observation_quality;		/* 0 none, 1 partial, 2 good */
	char	limitations[4][MEMBRANE_ORCH_TEXT_MAX];
	size_t	limitation_count;
}	membrane_orch_candidate_t;

typedef struct s_membrane_orch_decision
{
	int		schema_version;
	membrane_orch_intent_t	intent;
	int		has_selection;
	int		selected;					/* index into candidates[] */
	membrane_orch_candidate_t	candidates[MEMBRANE_ORCH_MAX_RUNTIMES];
	size_t	candidate_count;			/* input order */
	int		order[MEMBRANE_ORCH_MAX_RUNTIMES];	/* viable candidates, best
										 * first (order[0] == selected) */
	size_t	order_count;
	char	reasons[MEMBRANE_ORCH_MAX_REASONS][48];
	size_t	reason_count;
	char	confidence[8];				/* high|medium|low */
	char	confidence_text[MEMBRANE_ORCH_TEXT_MAX];
	char	summary[MEMBRANE_ORCH_TEXT_MAX * 2];
}	membrane_orch_decision_t;

/* Pure and deterministic: identical inputs => byte-identical output. */
void	membrane_orchestrate(membrane_orch_intent_t intent,
			const membrane_topology_t *topology,
			const membrane_orch_runtime_input_t *inputs, size_t n,
			membrane_orch_decision_t *out);

# ifdef __cplusplus
}
# endif

#endif
