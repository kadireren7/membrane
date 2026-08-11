#ifndef MEMBRANE_RUNTIME_CORE_H
# define MEMBRANE_RUNTIME_CORE_H

# include <stddef.h>
# include <stdint.h>
# include <stdio.h>

# include "membrane/codec.h"
# include "membrane/quant_simd.h"
# include "precision_policy.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * MEMBRANE Product Phase 5: llama-free core for the live shadow runtime.
 * Everything here operates on plain uint16_t (F16 bit pattern) arrays
 * and integers only -- no ggml_tensor, no llama_context, no llama.cpp
 * header anywhere in this file or runtime_core.c. This is deliberate:
 * it is what tests/unit/test_runtime_core.c exercises unconditionally
 * (no MEMBRANE_ENABLE_LLAMA needed), and it is the ONLY place block
 * packing, telemetry aggregation, and the baseline/shadow token
 * comparison live. tools/membrane-llama-runtime/llama_hook.cpp is the
 * sole file that touches ggml/llama types; it extracts live tensor data
 * into a uint16_t buffer and calls straight into
 * membrane_runtime_observe_tensor() below -- MEMBRANE's own policy/
 * quantization logic never sees an llama.cpp type.
 *
 * SHADOW MODE ONLY (Product Phase 5 scope): native llama KV remains the
 * authoritative store for attention. This module only ever reads
 * already-computed K/V projection values and quantizes/dequantizes them
 * for measurement -- it never writes back into anything llama.cpp uses.
 * "encoded payload bytes" below describes what MEMBRANE's codec would
 * produce for the observed values, not an actual reduction in llama's
 * process memory.
 */

# define MEMBRANE_RUNTIME_ELEMS_PER_BLOCK	128u	/* matches Phase 1-4 */
# define MEMBRANE_RUNTIME_MAX_LAYERS		512u

typedef enum e_membrane_runtime_mode
{
	MEMBRANE_RUNTIME_MODE_BASELINE = 0,
	MEMBRANE_RUNTIME_MODE_SHADOW_Q8,
	MEMBRANE_RUNTIME_MODE_SHADOW_ADAPTIVE,
	MEMBRANE_RUNTIME_MODE_COUNT
}	membrane_runtime_mode_t;

const char	*membrane_runtime_mode_name(membrane_runtime_mode_t m);
int			membrane_runtime_mode_from_name(const char *name,
				membrane_runtime_mode_t *out);

typedef struct s_membrane_runtime_telemetry
{
	membrane_runtime_mode_t	mode;
	uint64_t					generated_tokens;
	uint64_t					inference_steps;

	uint32_t					layers_seen;
	uint64_t					k_blocks;
	uint64_t					v_blocks;
	uint64_t					total_blocks;
	uint64_t					total_values_observed;
	uint64_t					tail_values_excluded;

	membrane_workload_accum_t	accum;	/* q4/q8 blocks, encoded_bytes,
										 * decode_failures, q4_err/q8_err */

	double						membrane_seconds;	/* MEMBRANE-side work:
													 * tensor extraction +
													 * select+encode+decode
													 * +validate, timed by
													 * llama_hook.cpp around
													 * its whole callback
													 * body */
	double						inference_seconds;	/* wall time of every
													 * llama_decode() call.
													 * NOTE: this is the
													 * OUTER measurement --
													 * the eval callback
													 * (and therefore
													 * membrane_seconds)
													 * runs synchronously
													 * inside it, so
													 * membrane_seconds is a
													 * SUBSET of
													 * inference_seconds,
													 * never additive to it.
													 * "overhead ratio" =
													 * membrane_seconds /
													 * inference_seconds is
													 * the fraction of wall
													 * time spent in shadow
													 * processing, not extra
													 * time on top. */
}	membrane_runtime_telemetry_t;

typedef struct s_membrane_runtime_collector	membrane_runtime_collector_t;

/* max_layers bounds the internal layers-seen bitset (rejects a layer
 * index >= max_layers rather than overflowing); pass
 * MEMBRANE_RUNTIME_MAX_LAYERS unless a caller has a specific reason not
 * to. Returns NULL on allocation failure. */
membrane_runtime_collector_t	*membrane_runtime_collector_create(
									membrane_runtime_mode_t mode,
									uint32_t max_layers);
void	membrane_runtime_collector_destroy(membrane_runtime_collector_t *c);

/*
 * Packs `n_elems` F16 values (block-major, already extracted from live
 * llama KV projection tensor data by the caller) into
 * MEMBRANE_RUNTIME_ELEMS_PER_BLOCK-sized blocks and, for every FULL
 * block, runs membrane_bench_process_block() under the policy implied
 * by c's mode (BASELINE: this function returns MEMBRANE_OK immediately
 * without touching ANY counter -- not even total_values_observed/
 * tail_values_excluded -- and no block is processed, matching "no
 * MEMBRANE quantization" for the native path exactly; in practice this
 * branch is never reached in BASELINE mode at all, since the caller
 * never installs the eval callback that would call this function;
 * SHADOW_Q8: MEMBRANE_BENCH_POLICY_
 * Q8_ONLY; SHADOW_ADAPTIVE: MEMBRANE_BENCH_POLICY_ADAPTIVE with the
 * unmodified default threshold, MEMBRANE_QUANT_SELECT_DEFAULT_MAX_
 * Q4_REL_L2_ERROR). A trailing remainder (n_elems %
 * MEMBRANE_RUNTIME_ELEMS_PER_BLOCK) is never processed or padded --
 * counted in the returned telemetry's tail_values_excluded instead (see
 * docs/live-runtime.md for why no padding is used). is_v selects the K
 * vs V block/byte counters; layer must be < the max_layers this
 * collector was created with.
 *
 * Returns MEMBRANE_OK, MEMBRANE_ERR_INVALID_ARG (NULL c/values_f16, or
 * layer out of range), or whatever fatal infrastructure status
 * membrane_bench_process_block propagates (allocation failure, or the
 * quant_simd engine itself reporting non-OK -- a per-block DECODE
 * failure is soft-reported via the accumulator, not a fatal return,
 * matching precision_policy.h's existing contract).
 */
membrane_status_t	membrane_runtime_observe_tensor(
						membrane_runtime_collector_t *c,
						membrane_simd_backend_t backend, int32_t layer,
						int is_v, const uint16_t *values_f16,
						uint64_t n_elems);

/* One llama_decode() call boundary -- increments inference_steps and
 * resets the "blocks processed this step" counter read back by
 * membrane_runtime_step_block_count() (used for --debug-runtime's live-
 * interleaving proof: call begin_step before llama_decode, observe
 * tensors during it via the eval callback, then read
 * step_block_count() and call end_step after it returns). */
void	membrane_runtime_begin_step(membrane_runtime_collector_t *c);
void	membrane_runtime_end_step(membrane_runtime_collector_t *c);
uint64_t	membrane_runtime_step_block_count(
				const membrane_runtime_collector_t *c);

/* Adds to two independently-tracked wall-clock totals -- the caller
 * measures both with its own CLOCK_MONOTONIC readings around the
 * relevant region; this module never calls a clock itself, keeping it
 * platform/clock-source agnostic and trivially testable with fabricated
 * durations. */
void	membrane_runtime_add_inference_seconds(
			membrane_runtime_collector_t *c, double seconds);
void	membrane_runtime_add_membrane_seconds(
			membrane_runtime_collector_t *c, double seconds);
void	membrane_runtime_set_generated_tokens(membrane_runtime_collector_t *c,
			uint64_t n);

/* Copies the collector's final state into *out. Does not reset/destroy
 * the collector. */
void	membrane_runtime_finalize(const membrane_runtime_collector_t *c,
			membrane_runtime_telemetry_t *out);

/*
 * Block-count-weighted mean rel-L2 error across whichever mix of Q4/Q8
 * blocks *t actually used (same formula as
 * tools/membrane-quant-policy-bench's membrane_bench_weighted_mean_
 * rel_l2_error, reimplemented here against membrane_workload_accum_t
 * directly rather than depending on membrane_bench_core). 0.0 if no
 * blocks were processed (e.g. baseline mode).
 */
double	membrane_runtime_weighted_mean_rel_l2_error(
			const membrane_runtime_telemetry_t *t);
double	membrane_runtime_max_rel_l2_error(
			const membrane_runtime_telemetry_t *t);

/* FP16 bytes for every observed value, including the excluded tail
 * (2 bytes/element) -- "FP16 bytes observed", never called a process-
 * memory or RAM figure (see runtime_core.c's module comment and
 * docs/live-runtime.md). */
uint64_t	membrane_runtime_fp16_bytes_observed(
				const membrane_runtime_telemetry_t *t);

/* Reduction of t->accum.encoded_bytes against the FP16 size of only the
 * values that were actually encoded (full blocks, excluding the tail
 * that was never processed) -- an apples-to-apples "encoded payload
 * reduction for observed KV blocks" ratio, 0.0 if no blocks were
 * processed. Never claims actual RAM/process-memory reduction. */
double	membrane_runtime_theoretical_payload_reduction(
			const membrane_runtime_telemetry_t *t);

/* True iff both token ID sequences have the same length and every
 * element matches, in order. Pure/allocation-free, used both by the
 * runtime's own validation step and directly unit-tested. */
int	membrane_runtime_tokens_equal(const int32_t *a, size_t na,
		const int32_t *b, size_t nb);

/* Returns a pointer into `path` itself (never allocates, never
 * copies): the substring after the last '/', or `path` unchanged if it
 * has none. This is the ONLY path-sanitization boundary in this tool
 * -- main.cpp calls it before passing model_label/prompt_fixture to
 * membrane_runtime_print_json()/print_human(), which then emit
 * whatever string they are given verbatim (by design: see those
 * functions' own doc comments). Exposed here, not as a main.cpp-local
 * static, specifically so it has direct unit coverage with absolute-
 * path inputs (test_runtime_core.c), not just the vacuous "the printer
 * doesn't corrupt an already-safe basename" check. */
const char	*membrane_runtime_safe_basename(const char *path);

void	membrane_runtime_print_human(const membrane_runtime_telemetry_t *t,
			FILE *f);
/* model_label/prompt_fixture: safe basenames only, never absolute
 * paths (the caller is responsible for stripping paths -- see
 * llama_hook.cpp/main.cpp's safe_basename); token_ids may be NULL if
 * n_token_ids == 0. */
void	membrane_runtime_print_json(const membrane_runtime_telemetry_t *t,
			const char *model_label, const char *prompt_fixture,
			const int32_t *token_ids, size_t n_token_ids, FILE *f);

# ifdef __cplusplus
}
# endif

#endif
