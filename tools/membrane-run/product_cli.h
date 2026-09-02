#ifndef MEMBRANE_RUN_PRODUCT_CLI_H
# define MEMBRANE_RUN_PRODUCT_CLI_H

#include <cstdint>
#include <cstdio>
#include <string>

/*
 * membrane-run's product-facing CLI, llama-free and testable without
 * a model (test_product_cli.cpp) -- deliberately a SEPARATE parser
 * from tools/membrane-llama-runtime/cli_parse.h, not an extension of
 * it: that one's flag surface (shadow/inject modes, debug-perturbation
 * flags) is diagnostic/research, this one is the clean product surface
 * Section 3 asks for. Both link the same underlying decode loop
 * (tools/membrane-llama-runtime/decode_loop.h).
 */

/* Single authoritative version string -- referenced by product_cli.cpp
 * (--version/--help) and main.cpp (startup summary, JSON output) so
 * a release version bump only needs to change this one line. */
# define MEMBRANE_VERSION "0.3.0"

/* Phase 11A: --kv adaptive's CLI-level request value -- NOT a real
 * storage type (kv_store_telemetry.h's MEMBRANE_KV_STORE_NATIVE/Q8/Q5
 * are the only real ggml types; decode_loop.cpp/run_kv_store_pass()
 * never receives this value). main.cpp resolves an adaptive request
 * into a concrete MEMBRANE_KV_STORE_Q8/Q5 exactly once (CPU: after
 * model load, from real hparams; GPU: inside resolve_gpu_config(),
 * from the same pre-load GGUF estimate the existing --gpu-layers auto
 * guard already uses) before it ever reaches run_kv_store_pass/
 * kv_bytes_for_mode/membrane_check_kv_compat -- see main.cpp's
 * resolve_effective_kv_mode(). Deliberately defined here (product_cli
 * is the only CLI surface that exposes "adaptive") rather than in
 * kv_store_telemetry.h, which stays real-storage-only. */
# define MEMBRANE_KV_STORE_ADAPTIVE	3

/* Stable exit codes (Section 8) -- never change the meaning of an
 * already-shipped code, only add new ones. */
# define MEMBRANE_EXIT_SUCCESS			0
# define MEMBRANE_EXIT_CLI_ERROR		2
# define MEMBRANE_EXIT_MODEL_ERROR		3
# define MEMBRANE_EXIT_RUNTIME_ERROR	4
# define MEMBRANE_EXIT_UNSUPPORTED_KV	5	/* also used for an
											 * unsupported/unavailable
											 * GPU or device request
											 * (Phase 9B) -- same
											 * "runtime configuration
											 * this build/model can't
											 * satisfy, fail closed"
											 * semantic as Q8 KV. */

/* Phase 13.1, Section 4/5: stable, machine-readable top-level reason
 * codes for runtime failures that don't belong to any one pure policy
 * module (gpu_policy.h/kv_residency_policy.h/adaptive_kv_policy.h each
 * own their own set already) -- never change the MEANING of an
 * already-shipped code, only add new ones. */
# define MEMBRANE_REASON_MODEL_LOAD_FAILED		"MODEL_LOAD_FAILED"
# define MEMBRANE_REASON_TOKENIZATION_FAILED	"TOKENIZATION_FAILED"
# define MEMBRANE_REASON_CTX_TOO_SMALL			"CTX_TOO_SMALL_FOR_PROMPT"
# define MEMBRANE_REASON_GENERATION_FAILED		"GENERATION_FAILED"
# define MEMBRANE_REASON_KV_COMPAT_UNSUPPORTED	"KV_COMPAT_UNSUPPORTED"
/* Review fix (CodeRabbit, PR #22): was an inline literal in main.cpp's
 * plan_primary_reason() -- every other stable code in this cohort
 * lives behind a macro; an inline literal can't be referenced by tests
 * and is easy to change by accident. */
# define MEMBRANE_REASON_DEFAULT_BEHAVIOR_PRESERVED	"DEFAULT_BEHAVIOR_PRESERVED"

/* Phase 28, Section 6: the one reason_code used for a CLI-parse-time
 * failure's JSON error object (--json given anywhere in argv, but the
 * parse itself failed before o.want_json could even be trusted -- see
 * main.cpp's argv_has_json_flag()/print_cli_parse_error_json()). Every
 * parse-time failure shares this one code (unlike the runtime reason
 * codes above, parse failures have no finer-grained taxonomy -- the
 * `message` field, not `reason_code`, carries the specific cause). */
# define MEMBRANE_REASON_CLI_PARSE_ERROR	"CLI_PARSE_ERROR"

/* Phase 35: --ctx auto's own reason codes -- distinct from context_
 * recommender.h's own MEMBRANE_CTXREC_STATUS_* (those are the pure
 * core's internal taxonomy; these are what main.cpp's CLI-facing error
 * path actually emits, echoing the core's status via `message`/
 * `reason_code` verbatim where one exists, and using these only for
 * CLI-layer-specific conditions the core has no concept of). Never
 * change the meaning of an already-shipped code, only add new ones. */
# define MEMBRANE_REASON_CTX_AUTO_NEEDS_PROMPT		"CTX_AUTO_NEEDS_PROMPT"
# define MEMBRANE_REASON_CTX_AUTO_TOKENIZE_FAILED	"CTX_AUTO_TOKENIZE_FAILED"
# define MEMBRANE_REASON_CTX_AUTO_HOST_MEMORY_STALE	"CTX_AUTO_HOST_MEMORY_STALE_AT_APPLY"

/* Phase 13.2, Section 16: stable, machine-readable plan-warning codes --
 * informational, never errors (a warning never changes exit_code or
 * ok). Same never-change-the-meaning convention as every other reason-
 * code set in this project. Kept small and deliberately non-exhaustive:
 * only semantically meaningful, actionable conditions get a code, not
 * every branch that COULD be worth mentioning (Section 16: "Do NOT
 * spam warnings"). */
# define MEMBRANE_WARNING_CPU_FALLBACK			"CPU_FALLBACK"
# define MEMBRANE_WARNING_LOW_GPU_HEADROOM		"LOW_GPU_HEADROOM"
# define MEMBRANE_WARNING_METADATA_ESTIMATE_ONLY	"METADATA_ESTIMATE_ONLY"
# define MEMBRANE_WARNING_HOST_MEMORY_PRESSURE	"HOST_MEMORY_PRESSURE"
# define MEMBRANE_WARNING_EXPLICIT_OVERRIDE		"EXPLICIT_OVERRIDE"
/* Review fix (CodeRabbit, PR #23): EXPLICIT_OVERRIDE is the only code
 * above emitted with a runtime suffix -- build_plan_warnings() (main.cpp)
 * appends ":gpu_layers", ":kv", or ":kv_placement" to name which
 * --auto-managed field the user explicitly overrode, one warning entry
 * per overridden field. Every other code above is always emitted
 * verbatim (never suffixed) -- a consumer comparing warning strings for
 * exact equality must account for this one exception. */

typedef enum e_membrane_run_prompt_mode
{
	MEMBRANE_RUN_PROMPT_NONE = 0,
	MEMBRANE_RUN_PROMPT_TEXT,		/* --prompt "..." */
	MEMBRANE_RUN_PROMPT_FILE,		/* --prompt-file FILE */
	MEMBRANE_RUN_PROMPT_STDIN		/* --prompt - */
}	membrane_run_prompt_mode_t;

/* Phase 35, Section 3: an explicit representation for --ctx's three
 * real states -- deliberately NOT overloading ctx==0 (which already
 * has a real, pre-existing, unrelated meaning: "not given, auto-size
 * tightly to prompt length + --gen-tokens + 8", used throughout
 * main.cpp/product_cli.cpp before this phase). MEMBRANE_RUN_CTX_AUTO
 * is the NEW `--ctx auto` hardware-aware recommendation request;
 * `ctx` itself stays 0 (unused placeholder) for AUTO until main.cpp's
 * own recommendation step resolves it to a concrete value BEFORE any
 * ctx==0-branching code runs -- every pre-existing `o.ctx == 0`/
 * `o.ctx > 0` check in this codebase is left completely untouched by
 * this phase (Section 4: byte-for-byte backward compatible), since by
 * the time those checks run, ctx has already been resolved to a real
 * positive number for both EXPLICIT and (post-resolution) AUTO. */
typedef enum e_membrane_run_ctx_mode
{
	MEMBRANE_RUN_CTX_UNSPECIFIED = 0,	/* --ctx not given at all --
										 * pre-existing ctx==0 auto-size-
										 * from-prompt default, unchanged */
	MEMBRANE_RUN_CTX_EXPLICIT,			/* --ctx N */
	MEMBRANE_RUN_CTX_AUTO				/* --ctx auto (Phase 35) */
}	membrane_run_ctx_mode_t;

typedef struct s_membrane_run_opts
{
	const char					*model_path;
	membrane_run_prompt_mode_t	prompt_mode;
	std::string					prompt_text;	/* PROMPT_TEXT only */
	const char					*prompt_file;	/* PROMPT_FILE only */

	uint32_t	ctx;			/* meaningful only when ctx_mode ==
								 * EXPLICIT (the --ctx N value) OR after
								 * main.cpp's own --ctx auto resolution
								 * step has run (then holds the resolved
								 * value) -- see ctx_mode's own doc
								 * comment. 0 with ctx_mode UNSPECIFIED:
								 * pre-existing, unchanged "auto-size to
								 * prompt + gen_tokens + 8" default. */
	membrane_run_ctx_mode_t	ctx_mode;	/* Phase 35 -- see its own
								 * enum doc comment above */
	int			kv_mode;		/* MEMBRANE_KV_STORE_NATIVE/Q8/Q5/ADAPTIVE (default
								 * NATIVE -- Section 4: v0.2 never
								 * unexpectedly changes model behavior).
								 * Also selects which compressed mode
								 * --compare-kv/--gpu-bench compare
								 * native against (Q8 unless Q5 was
								 * explicitly requested -- see
								 * run_compare_mode()/
								 * run_gpu_bench_mode() in main.cpp). May
								 * also resolve to a real mode chosen by
								 * --kv adaptive's policy -- see
								 * main.cpp's resolve_effective_kv_mode(). */
	int			want_kv_budget;	/* whether --kv-budget-mib was given */
	uint64_t	kv_budget_bytes;	/* --kv-budget-mib N * 1024 * 1024,
									 * Section 5: a hard input to --kv
									 * adaptive's policy only -- rejected
									 * at parse time if given without
									 * --kv adaptive, so it can never
									 * silently alter an explicit q8/q5
									 * choice. Only meaningful when
									 * want_kv_budget. */
	int			gen_tokens;
	int			threads;		/* 0 = let llama.cpp pick its own default */

	int			want_json;
	int			quiet;
	int			verbose;
	int			include_text;	/* JSON only: include generated text */
	int			compare_kv;		/* Section 6: explicit benchmark/compare
								 * mode, reuses the Phase 7 3-pass
								 * machinery -- never the default */
	int			plan_only;		/* Phase 13.2, Section 11: --plan-only --
								 * resolve the exact same policy
								 * pipeline as a normal run (model load,
								 * shape read, GPU/adaptive/placement
								 * resolution) and print the resulting
								 * plan, but never call run_kv_store_
								 * pass() -- no generation, no decode.
								 * Mutually exclusive with --compare-kv/
								 * --gpu-bench (neither has a "plan"
								 * concept of its own). */

	int32_t		gpu_layers;		/* Phase 9B/9B.1: --gpu-layers. 0 =
								 * CPU-only (default, also settable
								 * explicitly as the CPU-forcing
								 * option) -- a build with a GPU
								 * backend compiled in still defaults
								 * to CPU-only unless requested. -1 =
								 * "all" (MEMBRANE_GPU_LAYERS_ALL).
								 * -2 = "auto" (MEMBRANE_GPU_LAYERS_
								 * AUTO, gpu_policy.h -- MEMBRANE
								 * picks a safe count from real
								 * device/model memory information).
								 * N>0 = exactly N layers. Never
								 * implicit. */
	std::string	device;			/* --device NAME, empty if not given.
								 * Only meaningful with gpu_layers != 0
								 * (validated at parse time). */
	int			want_device;	/* whether --device was explicitly
								 * given (device.empty() alone can't
								 * distinguish "not given" from a
								 * name that happens to be empty) */
	int			gpu_bench;		/* Phase 9B.1: --gpu-bench -- explicit
								 * native-vs-q8 comparison under a GPU
								 * configuration, requires gpu_layers
								 * != 0, mutually exclusive with
								 * compare_kv (that one is CPU/generic;
								 * this one is GPU-specific and adds
								 * memory-policy telemetry). */

	int			kv_placement;	/* Phase 12H: MEMBRANE_KV_PLACEMENT_
								 * DEFAULT/GPU/CPU/AUTO (default DEFAULT
								 * -- zero behavior change unless
								 * explicitly requested, Section 4). A
								 * SEPARATE dimension from kv_mode
								 * (precision): --kv-placement never
								 * changes KV precision, --kv never
								 * changes KV device residency. gpu/auto
								 * require gpu_layers != 0 (no GPU
								 * device to place KV on otherwise);
								 * cpu/default are always valid. */

	/* Phase 13.1: --auto is a PRESET applied once, after the parse
	 * loop, to whichever of gpu_layers/kv_mode/kv_placement the user
	 * did NOT also explicitly pass -- it resolves into the exact same
	 * MEMBRANE_GPU_LAYERS_AUTO/MEMBRANE_KV_STORE_ADAPTIVE/MEMBRANE_KV_
	 * PLACEMENT_AUTO values --gpu-layers auto/--kv adaptive/--kv-
	 * placement auto already produce, never a second/parallel decision
	 * path. want_gpu_layers/want_kv_mode/want_kv_placement (same
	 * "was this explicitly given" convention as want_device/
	 * want_kv_budget above) exist ONLY so --auto knows which fields it
	 * may fill in -- explicit user flags always win (Section 8). Not
	 * reset once auto_mode fills a field in: e.g. after `--auto --kv
	 * q8`, want_kv_mode is still 1 (as parsed) and kv_mode is q8 (the
	 * explicit value); auto_mode being 1 only means gpu_layers/
	 * kv_placement (not explicitly given here) were auto-filled. */
	int			auto_mode;			/* --auto was given */
	int			want_gpu_layers;	/* --gpu-layers was explicitly given */
	int			want_kv_mode;		/* --kv was explicitly given */
	int			want_kv_placement;	/* --kv-placement was explicitly
									 * given */

	int			want_version;
	int			want_help;

	/* Phase 28, Section 10/15: both short-circuit BEFORE --model/
	 * --prompt are required, exactly like want_help/want_version above
	 * -- neither needs a model file. Handled in main.cpp (both need
	 * live device enumeration via gpu_device.h, which is llama-linked,
	 * so they can't be resolved inside this llama-free parser). */
	int			want_list_devices;	/* --list-devices */
	int			want_doctor;		/* --doctor */

	/* Phase 30, Section 5-8: --inspect-model -- unlike want_list_
	 * devices/want_doctor above, this one DOES need --model (checked
	 * normally, same required-ness as an ordinary run) but does NOT
	 * need --prompt and triggers no generation -- see main.cpp's
	 * run_inspect_model_mode(), which reads only GGUF metadata
	 * (membrane_gpu_estimate_model(), gpu_device.h) and never calls
	 * llama_model_load_from_file() at all. o->ctx is honored if given
	 * (enables the KV-byte-estimate section of the output) but never
	 * defaulted/invented if omitted (Section 6: "Do not invent 2048
	 * silently"). */
	int			want_inspect_model;	/* --inspect-model */
}	membrane_run_opts_t;

void	membrane_run_usage(FILE *out);
void	membrane_run_print_version(FILE *out);

/* Returns MEMBRANE_EXIT_SUCCESS on success, MEMBRANE_EXIT_CLI_ERROR on
 * any parse/validation failure (message already printed to stderr).
 * want_help/want_version are checked by the caller BEFORE treating a
 * nonzero return as fatal -- --help/--version short-circuit before
 * --model is required, matching ordinary CLI convention. */
int		membrane_run_parse_opts(int argc, char **argv,
			membrane_run_opts_t *o);

#endif
