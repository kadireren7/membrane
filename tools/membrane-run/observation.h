#ifndef MEMBRANE_RUN_OBSERVATION_H
# define MEMBRANE_RUN_OBSERVATION_H

# include <stddef.h>
# include <stdint.h>

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Milestone I1 (observation foundation): the backend-neutral, READ-ONLY,
 * point-in-time observation snapshot -- "what is happening on this
 * machine right now", as facts only. No pressure classification, no
 * recommendation, no history (all later milestones -- see
 * docs/observability.md).
 *
 * The one rule this module exists to enforce: a number never loses where
 * it came from. Every telemetry field is a small wrapper (value + known +
 * provenance + source) instead of a naked integer, so
 *
 *     VRAM free           [measured]   (read from the device right now)
 *     estimated KV bytes  [estimated]  (Planner v2 arithmetic)
 *
 * can sit side by side without either being mistaken for the other.
 *
 * Field invariants (enforced by the setters below, asserted by
 * test_observation.c):
 *   known == 0  =>  provenance == UNKNOWN, value zeroed. An unknown
 *                   field is NEVER a zero-valued fact; renderers print
 *                   "unknown" / JSON null, never 0.
 *   known == 1  =>  provenance != UNKNOWN.
 *   source      =>  a short, stable machine token naming the real probe
 *                   or input the value came from ("proc_meminfo",
 *                   "planner_v2", ...), or -- for an unknown field -- why
 *                   it is unknown ("no_gpu_device", "not_instrumented").
 *
 * Pure: llama-free, no I/O, no allocation. The native provider that
 * actually probes the host lives in tools/membrane/observe_cmd.cpp.
 */

# define MEMBRANE_OBSERVATION_SCHEMA_VERSION	1

typedef enum e_membrane_obs_provenance
{
	MEMBRANE_OBS_PROV_UNKNOWN = 0,		/* not known -- value meaningless */
	MEMBRANE_OBS_PROV_MEASURED,			/* read from the OS / device /
										 * service manager during THIS
										 * snapshot (e.g. /proc/meminfo,
										 * ggml device memory query) --
										 * or derived only from such reads */
	MEMBRANE_OBS_PROV_RUNTIME_REPORTED,	/* stated by the running runtime
										 * itself (e.g. the native server's
										 * own GET /v1/status) */
	MEMBRANE_OBS_PROV_ESTIMATED,		/* computed by MEMBRANE arithmetic
										 * (Planner v2, the server's own
										 * load-time estimates) -- never a
										 * measurement */
	MEMBRANE_OBS_PROV_STATIC_METADATA,	/* recorded facts about an
										 * artifact (GGUF header, registry
										 * entry, built-in catalog) */
	MEMBRANE_OBS_PROV_CONFIGURED		/* the user's persisted config
										 * (server.json) -- intent, not
										 * live state */
}	membrane_obs_provenance_t;

const char	*membrane_obs_provenance_name(membrane_obs_provenance_t p);

# define MEMBRANE_OBS_SOURCE_MAX	64
# define MEMBRANE_OBS_STR_MAX		128

typedef struct s_membrane_obs_u64
{
	int							known;
	uint64_t					value;
	membrane_obs_provenance_t	provenance;
	char						source[MEMBRANE_OBS_SOURCE_MAX];
}	membrane_obs_u64_t;

typedef struct s_membrane_obs_bool
{
	int							known;
	int							value;
	membrane_obs_provenance_t	provenance;
	char						source[MEMBRANE_OBS_SOURCE_MAX];
}	membrane_obs_bool_t;

typedef struct s_membrane_obs_str
{
	int							known;
	char						value[MEMBRANE_OBS_STR_MAX];
	membrane_obs_provenance_t	provenance;
	char						source[MEMBRANE_OBS_SOURCE_MAX];
}	membrane_obs_str_t;

/* Setters. A known setter called with MEMBRANE_OBS_PROV_UNKNOWN (or a NULL
 * string value) degrades to the unknown setter -- a caller can never
 * produce a "known but provenance unknown" field. `source`/`reason` may be
 * NULL ("" is stored). */
void	membrane_obs_u64_set(membrane_obs_u64_t *f, uint64_t value,
			membrane_obs_provenance_t prov, const char *source);
void	membrane_obs_u64_unknown(membrane_obs_u64_t *f, const char *reason);
void	membrane_obs_bool_set(membrane_obs_bool_t *f, int value,
			membrane_obs_provenance_t prov, const char *source);
void	membrane_obs_bool_unknown(membrane_obs_bool_t *f, const char *reason);
void	membrane_obs_str_set(membrane_obs_str_t *f, const char *value,
			membrane_obs_provenance_t prov, const char *source);
void	membrane_obs_str_unknown(membrane_obs_str_t *f, const char *reason);

/*
 * Overall status (Part 14 of the I1 task) -- deterministic:
 *   UNAVAILABLE  the runtime itself could not be observed at all
 *                (runtime_observable == 0). The only non-success status.
 *   COMPLETE     runtime observable AND every field in the snapshot's
 *                field table is known.
 *   PARTIAL      runtime observable, at least one field unknown.
 * For membrane-native in I1, COMPLETE is not reachable in practice: live
 * KV bytes, active context and server process RSS are not instrumented,
 * so they are always unknown (docs/observability.md). That is the honest
 * answer, not a bug.
 */
typedef enum e_membrane_obs_status
{
	MEMBRANE_OBS_STATUS_UNAVAILABLE = 0,
	MEMBRANE_OBS_STATUS_PARTIAL,
	MEMBRANE_OBS_STATUS_COMPLETE
}	membrane_obs_status_t;

const char	*membrane_obs_status_name(membrane_obs_status_t s);

/* One model the running native server reports as resident. Every field is
 * as reported by GET /v1/status: identity/backend/precision are
 * RUNTIME_REPORTED, the byte figures are the server's own load-time
 * ESTIMATES (runtime_session.cpp gs.estimated_*), never measured usage. */
# define MEMBRANE_OBS_MAX_RESIDENT	8

typedef struct s_membrane_obs_resident_model
{
	membrane_obs_str_t	name;
	membrane_obs_str_t	state;
	membrane_obs_str_t	backend;
	membrane_obs_u64_t	gpu_layers;
	membrane_obs_str_t	kv_precision;
	membrane_obs_u64_t	estimated_model_bytes;
	membrane_obs_u64_t	estimated_kv_bytes;
}	membrane_obs_resident_model_t;

typedef struct s_membrane_observation_snapshot
{
	int					schema_version;

	/* IDENTITY -- timestamps are set by the collector: wall-clock UTC at
	 * collection start, plus how long collection took on a monotonic
	 * clock. Every probe below ran inside that window. */
	int64_t				timestamp_unix_ms;
	char				timestamp_utc[32];		/* RFC 3339, "Z" */
	uint64_t			collection_duration_ms;
	char				runtime_id[32];
	int					runtime_observable;
	membrane_obs_str_t	runtime_availability;
	membrane_obs_status_t	status;				/* set by _finalize() */

	/* HOST MEMORY */
	membrane_obs_u64_t	ram_total_bytes;
	membrane_obs_u64_t	ram_available_bytes;
	membrane_obs_u64_t	ram_used_bytes;			/* derived: total - available */
	membrane_obs_u64_t	swap_total_bytes;
	membrane_obs_u64_t	swap_free_bytes;
	membrane_obs_u64_t	process_rss_bytes;		/* the serving process --
												 * not instrumented in I1 */

	/* GPU / DEVICE -- the first GPU/iGPU ggml enumerates, the same device
	 * choice `membrane plan` makes. */
	membrane_obs_u64_t	gpu_device_count;
	membrane_obs_str_t	gpu_backend;
	membrane_obs_str_t	gpu_device_name;
	membrane_obs_str_t	gpu_device_description;
	membrane_obs_u64_t	vram_total_bytes;
	membrane_obs_u64_t	vram_free_bytes;
	membrane_obs_u64_t	vram_used_bytes;		/* derived: total - free,
												 * device-wide (all
												 * processes) */

	/* MODEL -- configured intent vs. what the runtime reports loaded are
	 * separate fields on purpose (Part 11): a configured default model is
	 * NOT evidence that anything is resident. */
	membrane_obs_str_t	configured_model;
	membrane_obs_bool_t	configured_model_registered;
	membrane_obs_u64_t	model_file_size_bytes;
	membrane_obs_str_t	model_quant;
	membrane_obs_str_t	model_arch;
	membrane_obs_u64_t	resident_model_count;
	size_t				resident_model_len;		/* entries in resident_models */
	membrane_obs_resident_model_t	resident_models[MEMBRANE_OBS_MAX_RESIDENT];

	/* CONTEXT */
	membrane_obs_u64_t	context_active;			/* not reported by the
												 * native server in I1 */
	membrane_obs_u64_t	context_planned;		/* Planner v2 */
	membrane_obs_u64_t	context_model_max;		/* GGUF metadata */

	/* KV */
	membrane_obs_str_t	kv_planned_precision;	/* Planner v2 */
	membrane_obs_str_t	kv_planned_placement;	/* Planner v2 */
	membrane_obs_u64_t	kv_estimated_bytes;		/* Planner v2 -- NEVER a
												 * claim of live usage */
	membrane_obs_u64_t	kv_measured_bytes;		/* live KV instrumentation --
												 * not available in I1 */

	/* SERVICE / RUNTIME STATE */
	membrane_obs_str_t	service_manager;
	membrane_obs_bool_t	service_installed;
	membrane_obs_bool_t	service_active;
	membrane_obs_str_t	server_endpoint;
	membrane_obs_bool_t	server_reachable;
	membrane_obs_str_t	server_version;

	/* HEADROOM RAW FACTS -- what is free right now, with NO planner
	 * reserve and NO planned footprint subtracted (policy-adjusted
	 * headroom and pressure classification belong to I3). */
	membrane_obs_u64_t	ram_headroom_bytes;
	membrane_obs_u64_t	vram_headroom_bytes;
}	membrane_observation_snapshot_t;

/* Every field unknown (source "not_collected"), status UNAVAILABLE,
 * runtime_id copied (NULL -> ""). */
void	membrane_obs_snapshot_init(membrane_observation_snapshot_t *s,
			const char *runtime_id);

/* Derives ram_used/vram_used/headroom from their inputs -- only when every
 * input is known and consistent (e.g. available <= total); otherwise the
 * derived field stays unknown with source "inputs_unknown"/
 * "inputs_inconsistent". Derived provenance is the inputs' provenance
 * (always MEASURED today). Then computes status. Idempotent. */
void	membrane_obs_snapshot_finalize(membrane_observation_snapshot_t *s);

/*
 * The single, ordered field table: every JSON field, every "unknown
 * fields" list entry and the status computation walk exactly this list,
 * so they cannot disagree. `path` is "<section>.<field>" (the JSON
 * layout). Resident models are a variable-length array and are not part
 * of the table (their container, resident_model_count, is).
 */
typedef enum e_membrane_obs_field_kind
{
	MEMBRANE_OBS_KIND_U64 = 0,
	MEMBRANE_OBS_KIND_BOOL,
	MEMBRANE_OBS_KIND_STR
}	membrane_obs_field_kind_t;

typedef struct s_membrane_obs_field_ref
{
	const char					*path;
	membrane_obs_field_kind_t	kind;
	const void					*field;		/* membrane_obs_{u64,bool,str}_t */
}	membrane_obs_field_ref_t;

# define MEMBRANE_OBS_MAX_FIELDS	64

size_t	membrane_obs_snapshot_fields(const membrane_observation_snapshot_t *s,
			membrane_obs_field_ref_t *out, size_t max_out);

/* Uniform accessors over a field_ref, so callers need not switch on kind
 * just to read known/provenance/source. */
int							membrane_obs_field_known(
								const membrane_obs_field_ref_t *r);
membrane_obs_provenance_t	membrane_obs_field_provenance(
								const membrane_obs_field_ref_t *r);
const char					*membrane_obs_field_source(
								const membrane_obs_field_ref_t *r);

/* Counts over the field table. */
void	membrane_obs_snapshot_count(const membrane_observation_snapshot_t *s,
			size_t *known, size_t *total);

/* Formats unix milliseconds as RFC 3339 UTC ("2026-09-24T10:11:12.345Z")
 * without gmtime_r/gmtime_s (portable, pure). Returns 0 if out_size < 25
 * or ms is negative. */
int		membrane_obs_format_utc(int64_t unix_ms, char *out, size_t out_size);

# ifdef __cplusplus
}
# endif

#endif
