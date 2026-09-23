#ifndef MEMBRANE_RUN_RUNTIME_CAPABILITIES_H
# define MEMBRANE_RUN_RUNTIME_CAPABILITIES_H

# include <stddef.h>

# include "membrane_plan.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Milestone H1 (runtime abstraction foundation): the ONE place that
 * says, explicitly and testably, what an inference RUNTIME (the thing
 * that actually executes a model -- MEMBRANE's own embedded llama.cpp
 * today, an external engine like Ollama/vLLM later) can and cannot do.
 *
 * This is deliberately NOT a "backend" -- that word is already double-
 * booked in this codebase: include/membrane/backend.h is the cold-tier
 * KV-block STORAGE backend (file/RAM/CXL/FPGA), and gpu_device.h's own
 * `backend` field is the ggml COMPUTE backend name (CPU/CUDA/Vulkan/
 * Metal) a single native runtime may use internally. "Runtime" here
 * means a THIRD, higher-level thing: which inference ENGINE is running
 * the model at all (native llama.cpp vs. a future external adapter),
 * completely orthogonal to both of the above.
 *
 * Llama-free, no I/O, no device/model access -- same testable-without-
 * a-model/GPU pattern as membrane_plan.h/plan_v2_resolver.h. The whole
 * point of H1 is that this module can be reasoned about and tested as
 * plain data + pure functions, with zero coupling to whether a GPU is
 * present or `membrane serve` is running.
 *
 * This module describes only the BUILT-IN, statically-known runtime:
 * MEMBRANE_RUNTIME_ID_NATIVE. membrane_runtime_discover()/membrane_
 * runtime_describe() below never do I/O and therefore never describe an
 * external runtime, whose availability can only be known by probing it.
 *
 * Milestone H2: MEMBRANE_RUNTIME_ID_OLLAMA now has a real, read-only
 * adapter -- tools/membrane/runtime_ollama.h (all HTTP lives there, never
 * here), composed with this module's native descriptor by the static
 * table in tools/membrane/runtime_adapter.h (runtime_registry.cpp).
 * MEMBRANE_RUNTIME_ID_VLLM is still a RESERVED string
 * constant only; no vLLM adapter exists in this build.
 */

# define MEMBRANE_RUNTIME_SCHEMA_VERSION	1

# define MEMBRANE_RUNTIME_ID_MAX		32
# define MEMBRANE_RUNTIME_NAME_MAX		64
# define MEMBRANE_RUNTIME_VERSION_MAX	32
# define MEMBRANE_RUNTIME_ENDPOINT_MAX	128
# define MEMBRANE_RUNTIME_REASON_MAX	160

# define MEMBRANE_RUNTIME_ID_NATIVE	"membrane-native"
/* H2: read-only adapter in tools/membrane/runtime_ollama.h. */
# define MEMBRANE_RUNTIME_ID_OLLAMA	"ollama"
/* Reserved only -- see this header's own top comment. Not implemented. */
# define MEMBRANE_RUNTIME_ID_VLLM		"vllm"

/*
 * Part 3: one explicit capability state per dimension, never a bare
 * bool -- an external runtime later may genuinely not know (UNKNOWN),
 * genuinely half-support something (PARTIAL, e.g. this project's own
 * native runtime's memory telemetry: a static byte estimate exists,
 * nothing live), fully support it, or not support it at all.
 *
 * UNKNOWN is deliberately the zero value: a zero-initialized (or not-
 * yet-populated) membrane_runtime_capabilities_t claims nothing, which
 * is the only fail-closed default -- an accidental zero must never read
 * as "supported" NOR as a confident "unsupported" it has no basis for.
 * membrane_runtime_negotiate_plan() below treats UNKNOWN the same as
 * UNSUPPORTED for "can this be relied on" purposes (Part 13, test D),
 * while still keeping it a distinct, honestly-reported state.
 */
typedef enum e_membrane_capability_state
{
	MEMBRANE_CAPABILITY_UNKNOWN = 0,
	MEMBRANE_CAPABILITY_UNSUPPORTED,
	MEMBRANE_CAPABILITY_PARTIAL,
	MEMBRANE_CAPABILITY_SUPPORTED
}	membrane_capability_state_t;

const char	*membrane_capability_state_name(membrane_capability_state_t s);

/*
 * Part 11: where a capability's value came from -- a small, flat enum,
 * never a per-field provenance graph. H1's entire native matrix is
 * STATIC_CONTRACT (Part 4: every value below was established by reading
 * this exact codebase, not by probing the running binary), not
 * COMPILED_IN (which would mean a real preprocessor/build-flag check,
 * e.g. "was this built with CUDA" -- H1 deliberately does not do that,
 * see runtime_capabilities.c's own top comment) and not RUNTIME_PROBE
 * (no live call into gpu_device.h/service_state.h happens here). The
 * external-runtime values exist now so a future adapter's own
 * provenance has an agreed vocabulary to report into, not because H1
 * produces them.
 */
typedef enum e_membrane_capability_provenance
{
	MEMBRANE_CAPABILITY_PROVENANCE_UNKNOWN = 0,
	MEMBRANE_CAPABILITY_PROVENANCE_COMPILED_IN,
	MEMBRANE_CAPABILITY_PROVENANCE_STATIC_CONTRACT,
	MEMBRANE_CAPABILITY_PROVENANCE_RUNTIME_PROBE,
	MEMBRANE_CAPABILITY_PROVENANCE_API_PROBE,
	MEMBRANE_CAPABILITY_PROVENANCE_VERSION_PROBE
}	membrane_capability_provenance_t;

const char	*membrane_capability_provenance_name(
				membrane_capability_provenance_t p);

typedef enum e_membrane_runtime_type
{
	MEMBRANE_RUNTIME_TYPE_NATIVE = 0,
	MEMBRANE_RUNTIME_TYPE_EXTERNAL
}	membrane_runtime_type_t;

const char	*membrane_runtime_type_name(membrane_runtime_type_t t);

/*
 * Part 2/12: describes the ENGINE's relationship to the MEMBRANE
 * process, never how a CLI subcommand happens to reach it. Native
 * llama.cpp is linked directly into `membrane`/`membrane serve`'s own
 * binary (runtime_session.cpp calls llama_model_load_from_file() in-
 * process) -- EMBEDDED_NATIVE -- regardless of the fact that `membrane
 * chat` itself is a separate loopback-HTTP client of `membrane serve`
 * (chat_cmd.h's own "no second inference path" contract): that is a
 * MEMBRANE-internal client/server split, not evidence the engine is
 * external. A future Ollama/vLLM adapter run as its own local daemon
 * MEMBRANE talks to over a local socket/port is LOCAL_EXTERNAL; one
 * reached over the network is REMOTE_EXTERNAL.
 */
typedef enum e_membrane_runtime_execution_mode
{
	MEMBRANE_RUNTIME_EXEC_EMBEDDED_NATIVE = 0,
	MEMBRANE_RUNTIME_EXEC_LOCAL_EXTERNAL,
	MEMBRANE_RUNTIME_EXEC_REMOTE_EXTERNAL
}	membrane_runtime_execution_mode_t;

const char	*membrane_runtime_execution_mode_name(
				membrane_runtime_execution_mode_t m);

/*
 * Part 12: "runtime exists / available / healthy / currently running"
 * are NOT the same question, and H1 deliberately only answers the
 * first two here. AVAILABLE means "this build can genuinely describe
 * and, in principle, use this runtime" -- for membrane-native that is
 * a static fact (the native engine is compiled into this very binary,
 * unconditionally), TRUE even when no `membrane serve`/service process
 * is running anywhere on the host right now. Whether the background
 * HTTP service is actually up is a SEPARATE, already-existing question
 * (`membrane status` / `membrane service status`) this module never
 * reads or mutates -- see runtime_capabilities.c's native descriptor
 * comment and docs/runtime-abstraction.md for the full rationale.
 */
typedef enum e_membrane_runtime_availability
{
	MEMBRANE_RUNTIME_AVAILABILITY_UNKNOWN = 0,
	MEMBRANE_RUNTIME_AVAILABILITY_UNAVAILABLE,
	MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE
}	membrane_runtime_availability_t;

const char	*membrane_runtime_availability_name(
				membrane_runtime_availability_t a);

/*
 * Milestone H2: the result of the most recent live probe of a runtime,
 * kept SEPARATE from availability (H1's "exists / available / healthy"
 * distinction). NOT_PROBED is the zero value and is what membrane-native
 * always reports: it is embedded in this binary, there is no separate
 * process to probe. An external adapter reports exactly one of the other
 * four after one bounded probe:
 *   UNREACHABLE  -- no HTTP response at all (refused, timeout, DNS)
 *   HEALTHY      -- the runtime's own version API answered as documented
 *   INCOMPATIBLE -- something answered, but not with the documented API
 *                   (e.g. a non-2xx status on the version endpoint)
 *   UNKNOWN      -- a 2xx answer whose body was malformed/unparseable
 */
typedef enum e_membrane_runtime_health
{
	MEMBRANE_RUNTIME_HEALTH_NOT_PROBED = 0,
	MEMBRANE_RUNTIME_HEALTH_UNREACHABLE,
	MEMBRANE_RUNTIME_HEALTH_HEALTHY,
	MEMBRANE_RUNTIME_HEALTH_INCOMPATIBLE,
	MEMBRANE_RUNTIME_HEALTH_UNKNOWN
}	membrane_runtime_health_t;

const char	*membrane_runtime_health_name(membrane_runtime_health_t h);

/*
 * Part 3: the full capability surface, grouped exactly as the H1 task
 * groups it. Every field is a membrane_capability_state_t, never a
 * bare bool -- see that type's own top comment. Grouped by comment,
 * not by nested struct, to keep this one flat, inspectable shape (no
 * generic capability-map/plugin framework, Part 6).
 */
typedef struct s_membrane_runtime_capabilities
{
	/* MODEL / LIFECYCLE */
	membrane_capability_state_t	model_enumeration;
	membrane_capability_state_t	model_load_unload;
	membrane_capability_state_t	model_switch;
	membrane_capability_state_t	model_metadata;

	/* PLANNING / CONTROL */
	membrane_capability_state_t	context_control;
	membrane_capability_state_t	gpu_layer_control;
	membrane_capability_state_t	quant_variant_control;
	membrane_capability_state_t	kv_precision_control;
	membrane_capability_state_t	kv_placement_control;
	membrane_capability_state_t	concurrency_control;
	membrane_capability_state_t	device_selection;
	membrane_capability_state_t	memory_headroom_telemetry;

	/* INFERENCE */
	membrane_capability_state_t	chat_completions;
	membrane_capability_state_t	streaming;
	membrane_capability_state_t	cancellation;

	/* OBSERVABILITY */
	membrane_capability_state_t	current_model;
	membrane_capability_state_t	active_context;
	membrane_capability_state_t	ram_usage;
	membrane_capability_state_t	vram_usage;
	membrane_capability_state_t	kv_cache_usage;
	membrane_capability_state_t	loaded_model_memory;

	/* ADVANCED -- deliberately included so the matrix also says what
	 * membrane-native does NOT do (Part 4: "do not overclaim"), not
	 * just what it does. Neither exists anywhere in this codebase
	 * today: a session's context/GPU-layers/KV-precision are fixed at
	 * llama_init_from_model() construction time (decode_loop.cpp) with
	 * no live reconfiguration path, and there is no cross-session or
	 * cross-runtime KV transfer of any kind. */
	membrane_capability_state_t	live_kv_migration;
	membrane_capability_state_t	dynamic_reconfiguration;
}	membrane_runtime_capabilities_t;

typedef struct s_membrane_runtime_descriptor
{
	char	id[MEMBRANE_RUNTIME_ID_MAX];
	char	display_name[MEMBRANE_RUNTIME_NAME_MAX];
	membrane_runtime_type_t				type;
	membrane_runtime_execution_mode_t		execution_mode;
	membrane_runtime_availability_t		availability;
	char	unavailable_reason[MEMBRANE_RUNTIME_REASON_MAX];	/* empty
								 * when AVAILABLE; H2: also set for an
								 * UNKNOWN (malformed-probe) result */

	membrane_runtime_health_t				health;		/* H2 */

	int		version_known;
	char	version[MEMBRANE_RUNTIME_VERSION_MAX];
	membrane_capability_provenance_t		version_provenance;	/* H2:
								 * UNKNOWN unless version_known */
	int		endpoint_known;			/* 0 for an in-process/embedded
								 * runtime -- there is no separate
								 * address to reach it at */
	char	endpoint[MEMBRANE_RUNTIME_ENDPOINT_MAX];

	membrane_capability_provenance_t		capability_provenance;
	membrane_runtime_capabilities_t		capabilities;
}	membrane_runtime_descriptor_t;

/*
 * Part 6: the smallest useful discovery interface -- no dynamic
 * plugin loading, no registration macros, no factory. A fixed, static
 * table of exactly one entry: the BUILT-IN runtime(s) only, no I/O.
 * External runtimes (H2: Ollama) are added on top of this by tools/
 * membrane/runtime_adapter.h's table, which is what `membrane runtime`
 * uses.
 *
 * Writes up to max_out descriptors into out, returns the number
 * written (always 0 or 1). Deterministic: repeated calls in the same
 * process always produce the same result.
 */
size_t	membrane_runtime_discover(membrane_runtime_descriptor_t *out,
			size_t max_out);

/* Looks up one runtime by id among exactly what membrane_runtime_
 * discover() would return. Returns 1 and fills *out iff found, 0
 * otherwise (out is left untouched on a miss -- callers must not read
 * it after a 0 return). */
int	membrane_runtime_describe(const char *runtime_id,
			membrane_runtime_descriptor_t *out);

/*
 * Part 5: read-only capability negotiation -- "can this runtime
 * satisfy this plan", never an application of anything. Consumes an
 * already-assembled membrane_plan_t (membrane_plan.h) purely by
 * reading it; never calls membrane_plan_assemble() or any planner
 * function itself (Part 10: plan representation/math is untouched by
 * H1).
 */
typedef enum e_membrane_negotiation_result
{
	MEMBRANE_NEGOTIATION_UNSUPPORTED = 0,
	MEMBRANE_NEGOTIATION_PARTIAL,
	MEMBRANE_NEGOTIATION_FULLY_SUPPORTED
}	membrane_negotiation_result_t;

const char	*membrane_negotiation_result_name(
				membrane_negotiation_result_t r);

# define MEMBRANE_NEGOTIATION_DIM_CONTEXT_CONTROL			"CONTEXT_CONTROL"
# define MEMBRANE_NEGOTIATION_DIM_GPU_LAYERS_CONTROL		"GPU_LAYERS_CONTROL"
# define MEMBRANE_NEGOTIATION_DIM_QUANT_VARIANT_CONTROL	"QUANT_VARIANT_CONTROL"
# define MEMBRANE_NEGOTIATION_DIM_KV_PRECISION_CONTROL		"KV_PRECISION_CONTROL"
# define MEMBRANE_NEGOTIATION_DIM_KV_PLACEMENT_CONTROL		"KV_PLACEMENT_CONTROL"

# define MEMBRANE_NEGOTIATION_MAX_DIMS	5

typedef struct s_membrane_negotiation_outcome
{
	membrane_negotiation_result_t	result;
	char	unsupported[MEMBRANE_NEGOTIATION_MAX_DIMS][40];	/* one of the
								 * MEMBRANE_NEGOTIATION_DIM_* strings
								 * above */
	size_t	unsupported_count;
}	membrane_negotiation_outcome_t;

/*
 * Which of the plan's own dimensions this exact plan actually relies
 * on is derived ONLY from fields membrane_plan_t already has -- never
 * a second guess: decisions.has_decisions (context/GPU-layers/KV-
 * precision/KV-placement are all populated together, or none are --
 * see membrane_plan.h's own struct comment) and identity.variant_known
 * (a specific quant was pinned). A dimension the plan never relied on
 * is never counted against a runtime's capabilities -- e.g. a catalog-
 * only plan with has_decisions == 0 needs nothing from any runtime and
 * always negotiates FULLY_SUPPORTED.
 *
 * A dimension counts as satisfied only when the matching capability is
 * exactly MEMBRANE_CAPABILITY_SUPPORTED; PARTIAL and UNKNOWN are both
 * treated as NOT satisfied (Part 13, test D: unknown is never silently
 * read as supported) and both appear in out->unsupported[].
 */
membrane_negotiation_outcome_t	membrane_runtime_negotiate_plan(
			const membrane_runtime_capabilities_t *caps,
			const membrane_plan_t *plan);

# ifdef __cplusplus
}
# endif

#endif
