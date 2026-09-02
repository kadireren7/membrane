#ifndef MEMBRANE_RUN_RUNTIME_SESSION_H
# define MEMBRANE_RUN_RUNTIME_SESSION_H

# include <cstdint>
# include <string>
# include <vector>

# include "llama.h"
# include "product_cli.h"
# include "decode_loop.h"
# include "kv_store_telemetry.h"
# include "joint_planner.h"
# include "auto_fallback.h"
# include "context_recommender.h"
# include "kv_residency_policy.h"
# include "gpu_device.h"
# include "gpu_policy.h"

/*
 * Mega Phase A, PR A1: the reusable MEMBRANE runtime-session core.
 *
 * Extracted (behavior-preserving, not a redesign -- see runtime_session.cpp's
 * own top comment) from tools/membrane-run/main.cpp so both the existing CLI
 * (main.cpp) and a future long-lived server (Mega Phase A3, not this PR) call
 * into the SAME model-lifecycle/planner/generation code -- there must never be
 * a separate "CLI runtime" and "server runtime" implementation.
 *
 * Model/context lifecycle policy (Section 5 of the Mega Phase A task):
 * persistent model, a NEW llama_context per generate() call -- this is
 * already how the underlying decode loop works (run_kv_store_pass() has
 * always created and destroyed one context per call), so this module does
 * not change that shape, only gives it a reusable, named entry point.
 *
 * This is an INTERNAL product API (Section 3 of the task), not a public SDK
 * -- membrane_run_opts_t (product_cli.h) is reused directly as the resolved-
 * request shape rather than inventing a parallel type, since it is already
 * plain data (ints/strings/enums), never raw argv text; a caller that is not
 * membrane-run's own CLI (e.g. a future server) constructs one itself from
 * whatever its own request shape is, rather than parsing anything.
 *
 * Every function in this header is llama_model_t/llama_context-aware C++ --
 * unlike joint_planner.h/host_memory_guard.h/etc. (plain-data, llama-free, C
 * with extern "C" linkage), this module is deliberately NOT llama-free: it IS
 * the one place model/context lifetime is owned, matching gpu_device.cpp's/
 * decode_loop.cpp's own existing llama-aware-C++-module precedent. No
 * extern "C" wrapping -- every caller (main.cpp today, a future server) is
 * C++ too.
 *
 * NO PRINTING, NO stdout/stderr, NO exit()/abort() anywhere in this module --
 * every failure path returns a structured membrane_run_error_t the caller
 * decides how to present (CLI: fprintf + print_error_json, unchanged text;
 * server: JSON HTTP error, Mega Phase A3). This is the one real fix PR A1
 * makes relative to main.cpp's pre-extraction code, where resolve_gpu_config()/
 * resolve_cpu_adaptive_kv()/resolve_kv_placement() printed directly --
 * confirmed by direct audit as the one genuine "CLI leaking into runtime
 * core" issue in the pre-existing code (never a hidden global/CLI-state
 * dependency otherwise: every one of these functions already took plain data
 * in via its parameters).
 */

/* Structured error info, populated by any resolve_*()/session function
 * instead of printing -- the caller (main.cpp today) reconstructs the EXACT
 * same fprintf(stderr, ...)/print_error_json(...) text this project's
 * existing tests already check, from these fields, at its own single call
 * site. `human` is the complete pre-formatted message (no "membrane-run: "
 * prefix, no trailing newline -- the CLI caller adds both, matching every
 * existing call site's own convention) for a human-readable stderr line;
 * `message`/`suggestion`/`available_devices` mirror print_error_json()'s own
 * parameter list exactly, so a caller can forward them verbatim. */
typedef struct s_membrane_run_error
{
	bool						set;
	int							exit_code;
	char						reason_code[64];
	std::string					human;
	std::string					message;
	std::string					suggestion;			/* empty = none */
	std::vector<std::string>	available_devices;	/* empty = none */
}	membrane_run_error_t;

/* Linux-only host-memory OBSERVABILITY snapshot (never a safety mechanism on
 * its own -- see host_memory_guard.h for the real enforcement path). */
typedef struct s_membrane_host_meminfo
{
	bool		ok;
	uint64_t	total_bytes;
	uint64_t	available_bytes;
	uint64_t	swap_total_bytes;
	uint64_t	swap_free_bytes;
}	membrane_host_meminfo_t;

void	membrane_read_host_meminfo(membrane_host_meminfo_t *out);

typedef struct s_model_shape
{
	std::string	arch_name;
	int32_t		n_layer;
	int32_t		n_embd;
	int32_t		n_head;
	int32_t		n_head_kv;
	int64_t		n_embd_gqa;
}	model_shape_t;

void		membrane_read_model_shape(llama_model *model, model_shape_t *s);
uint64_t	membrane_native_kv_bytes(const model_shape_t &s,
				uint32_t ctx_size);
uint64_t	membrane_q8_kv_bytes(const model_shape_t &s, uint32_t ctx_size);
uint64_t	membrane_q5_kv_bytes(const model_shape_t &s, uint32_t ctx_size);
uint64_t	membrane_kv_bytes_for_mode(const model_shape_t &s,
				uint32_t ctx_size, int kv_mode);
const char	*membrane_kv_mode_name(int kv_mode);
const char	*membrane_kv_type_label(int kv_mode);
std::string	membrane_gpu_layers_label(int32_t gpu_layers);

/* Phase 9B/9B.1/20/21/25 (see main.cpp's original field-by-field comments,
 * preserved verbatim in runtime_session.cpp): the FULL resolved-plan state --
 * what was requested vs. what was actually resolved by resolve_gpu_config()/
 * resolve_cpu_adaptive_kv()/resolve_kv_placement(), used both by this
 * module's own planner orchestration AND by the CLI's print_*() functions
 * (main.cpp), which is why the type itself, not just the functions that
 * populate it, is shared. */
typedef struct s_membrane_gpu_state
{
	bool		requested;
	bool		backend_gpu_capable;
	int32_t		gpu_layers_requested;
	int32_t		gpu_layers_selected;
	std::string	device_requested;
	std::string	device_selected;
	std::string	backend_selected;
	bool		policy_used;
	uint64_t	device_total_bytes;
	uint64_t	device_free_bytes;
	uint64_t	safety_reserve_bytes;
	uint64_t	estimated_model_bytes;
	uint64_t	estimated_kv_bytes;

	bool		adaptive_used;
	int			adaptive_selected_mode;
	std::string	adaptive_reason;
	uint64_t	adaptive_q8_kv_bytes;
	uint64_t	adaptive_q5_kv_bytes;
	int32_t		adaptive_q8_layers;
	int32_t		adaptive_q5_layers;
	int			adaptive_q8_valid;
	int			adaptive_q5_valid;

	bool		kv_placement_resolved;
	membrane_kv_residency_result_t	kv_placement;

	std::string	plan_reason_code;
	bool		auto_cpu_fallback;

	bool		joint_planner_used;
	int			joint_candidate_count;
	int			joint_selected_index;
	membrane_joint_plan_result_t	joint_result;

	membrane_fallback_trace_t	fallback_trace;
	bool		fallback_engaged;

	std::vector<std::string>	reason_trace;
	std::vector<std::string>	warnings;

	double		device_enumeration_ms;
	double		gguf_prescan_ms;
	double		joint_planner_core_ms;
}	membrane_gpu_state_t;

typedef struct s_membrane_timings
{
	double	planner_ms;
	double	model_load_ms;
	double	tokenization_ms;
	double	total_ms;
}	membrane_timings_t;

typedef struct s_membrane_ctxauto_outcome
{
	bool						ok;
	membrane_ctxrec_result_t	rec;
	size_t						prompt_token_count;
	bool						used_cpu_only_adaptive_path;
}	membrane_ctxauto_outcome_t;

/* Section 35's own doc comment (main.cpp, unchanged): pure device-selection
 * logic, reused by both resolve_gpu_config() and --ctx auto's own
 * pre-recommendation step. */
bool	membrane_select_gpu_device(const membrane_run_opts_t &o,
			const membrane_gpu_device_info_t *devices, size_t n_devices,
			size_t *out_chosen_idx, size_t *out_match_count);

/* Resolves GPU layer count / KV precision / KV placement's pre-load half
 * (device residency of weights) together, via the existing joint planner
 * (Phase 20) -- the SAME function main.cpp always called, moved here
 * unchanged except that every failure path now populates *err instead of
 * printing (see this header's own top comment). */
bool	membrane_resolve_gpu_config(const membrane_run_opts_t &o,
			uint32_t ctx_size, std::vector<ggml_backend_dev_t> *device_storage,
			llama_model_params *mp, membrane_gpu_state_t *gs,
			membrane_run_error_t *err);

/* --kv adaptive on CPU (post-model-load, real shape known) -- see Phase 11A
 * Section 5 in runtime_session.cpp for the full contract. */
bool	membrane_resolve_cpu_adaptive_kv(const membrane_run_opts_t &o,
			const model_shape_t &shape, uint32_t ctx_size,
			membrane_gpu_state_t *gs, membrane_run_error_t *err);

/* --kv-placement's static residency resolution (post-model-load). */
bool	membrane_resolve_kv_placement(const membrane_run_opts_t &o,
			const model_shape_t &shape, uint32_t ctx_size,
			int effective_kv_mode, membrane_gpu_state_t *gs,
			membrane_run_error_t *err);

/* --ctx auto (Phase 35): resolves o.ctx to a concrete, hardware-aware value
 * BEFORE resolve_gpu_config() runs -- see context_recommender.h/context_auto_
 * cli.h for the underlying pure recommendation engine this only orchestrates.
 * Already llama-free-error-clean in main.cpp (never printed directly); moved
 * verbatim. */
bool	membrane_resolve_ctx_auto(const membrane_run_opts_t &o,
			const std::string &prompt_text,
			const membrane_host_meminfo_t &host,
			membrane_ctxauto_outcome_t *out);

/* ------------------------------------------------------------------------
 * Session API (Section 3 of the Mega Phase A task).
 * ------------------------------------------------------------------------
 */

/* Wraps llama_backend_init()/llama_backend_free() -- process-global in
 * llama.cpp itself (not really per-instance state), given a thin named
 * handle here only to match the task's own suggested API shape (Section 3);
 * do not over-design beyond that. Call membrane_runtime_init() exactly once
 * before any membrane_model_open(), membrane_runtime_shutdown() exactly once
 * after every session has been closed. */
typedef struct s_membrane_runtime
{
	bool	initialized;
}	membrane_runtime_t;

void	membrane_runtime_init(membrane_runtime_t *rt);
void	membrane_runtime_shutdown(membrane_runtime_t *rt);

/* One loaded model plus the resolved plan it was loaded under. Model
 * lifetime is fully decoupled from any one generate() call (Section 4):
 * membrane_model_open() loads once; membrane_session_generate() may be
 * called any number of times against the same session; membrane_model_
 * close() frees it. A generate() call's own bounded apply-time fallback
 * (Section 6/Phase 21, reused unchanged) may reload the model in place if
 * the primary plan can't actually be instantiated -- callers must treat
 * `model` as owned by the session, never cache it themselves across calls. */
typedef struct s_membrane_model_session
{
	llama_model					*model;
	std::string					model_path;
	llama_model_params			mp_template;
	std::vector<ggml_backend_dev_t>	device_storage;
	int32_t						loaded_gpu_layers;
	membrane_gpu_state_t		gs;
	uint32_t					plan_ctx_size;	/* the ctx_size resolve_gpu_
												 * config() was planned
												 * against at open time */
}	membrane_model_session_t;

/* Loads the model exactly once (Section 4: "load model once"), after first
 * resolving GPU layers/KV precision/KV placement's device-residency half via
 * membrane_resolve_gpu_config() -- the same two-step main() always
 * performed. `ctx_size` is the ctx this session's plan is resolved against
 * (an explicit --ctx N or an already-resolved --ctx auto value; never 0 --
 * callers auto-sizing from a not-yet-tokenized prompt must resolve ctx_size
 * themselves first, exactly as main() already does). Returns false (session
 * left unopened, *err populated, nothing to close) on any planner or model-
 * load failure. */
bool	membrane_model_open(membrane_runtime_t *rt, const char *model_path,
			const membrane_run_opts_t &o, uint32_t ctx_size,
			membrane_model_session_t *session, membrane_run_error_t *err,
			double *out_planner_ms = NULL, double *out_model_load_ms = NULL);

void	membrane_model_close(membrane_model_session_t *session);

typedef struct s_membrane_generation_request
{
	const membrane_run_opts_t	*o;			/* resolved options: gen_tokens,
											 * kv_mode, kv_placement, verbose,
											 * kv budget, etc. -- NOT
											 * mutated */
	std::string					prompt_text;	/* tokenized against the
											 * session's own model */
	uint32_t					ctx_size;	/* must equal (or be <=) the ctx
											 * this session was opened with;
											 * 0 means "auto-size from this
											 * prompt + o->gen_tokens + 8",
											 * matching main()'s own pre-
											 * existing ctx==0 sentinel
											 * exactly */
	membrane_token_cb_t			token_cb;	/* NULL = no streaming */
}	membrane_generation_request_t;

typedef struct s_membrane_generation_result
{
	bool						ok;
	int							exit_code;
	membrane_run_error_t		err;			/* meaningful iff !ok */

	std::string					text;
	gen_run_result_t			gen_result;
	membrane_kv_store_telemetry_t	tel;
	std::vector<llama_token>	prompt_tokens;
	uint32_t					ctx_size;		/* the ctx actually used */
	int							effective_kv_mode;

	bool						fallback_engaged;
	membrane_fallback_trace_t	fallback_trace;	/* n_entries==0 iff
											 * fallback was never engaged */
}	membrane_generation_result_t;

/* Tokenizes the request's prompt against the session's already-loaded model,
 * resolves ctx_size/CPU-adaptive-KV/compatibility/KV-placement exactly as
 * main() always did post-load, then generates -- via the existing bounded
 * apply-time fallback controller (Phase 21, auto_fallback.h) when the
 * session's plan came from the joint planner, or a direct single-attempt
 * run_kv_store_pass() otherwise (identical branch main.cpp's run_normal_mode()
 * always took). May be called repeatedly against the same session (Section
 * 4/7: "3 sequential requests from one loaded model"). Never prints; the
 * caller (main.cpp today) is responsible for all presentation. */
bool	membrane_session_generate(membrane_model_session_t *session,
			const membrane_generation_request_t &req,
			membrane_generation_result_t *out);

#endif
