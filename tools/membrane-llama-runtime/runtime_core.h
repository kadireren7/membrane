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
 * MEMBRANE Product Phase 5/6: llama-free core for the live runtime.
 * Everything here operates on plain uint16_t (F16 bit pattern) arrays
 * and integers only -- no ggml_tensor, no llama_context, no llama.cpp
 * header anywhere in this file or runtime_core.c. This is deliberate:
 * it is what tests/unit/test_runtime_core.c exercises unconditionally
 * (no MEMBRANE_ENABLE_LLAMA needed), and it is the ONLY place block
 * packing, injection-scope filtering, telemetry aggregation, and
 * behavior-comparison math live. tools/membrane-llama-runtime/
 * llama_hook.cpp is the sole file that touches ggml/llama types.
 *
 * SHADOW modes (Phase 5): native llama KV remains authoritative. This
 * module only reads already-computed K/V projection values and
 * quantizes/dequantizes them for measurement -- never written back.
 *
 * INJECT modes (Phase 6, new): selected reconstructed K/V values DO
 * get written back into the tensor that feeds the actual KV-cache
 * write (llama_hook.cpp does the write; this module only computes the
 * reconstruction and the in-scope/full-block/tail split). Native
 * cache tensor allocation is unchanged either way -- "encoded payload
 * bytes" / "theoretical payload reduction" below describe what
 * MEMBRANE's codec would produce for the observed/injected values,
 * NEVER an actual reduction in llama's process memory (see
 * docs/live-runtime.md).
 */

# define MEMBRANE_RUNTIME_ELEMS_PER_BLOCK	128u	/* matches Phase 1-4 */
# define MEMBRANE_RUNTIME_MAX_LAYERS		512u

typedef enum e_membrane_runtime_mode
{
	MEMBRANE_RUNTIME_MODE_BASELINE = 0,
	MEMBRANE_RUNTIME_MODE_SHADOW_Q8,
	MEMBRANE_RUNTIME_MODE_SHADOW_ADAPTIVE,
	MEMBRANE_RUNTIME_MODE_INJECT_Q8,
	MEMBRANE_RUNTIME_MODE_INJECT_ADAPTIVE,
	MEMBRANE_RUNTIME_MODE_COUNT
}	membrane_runtime_mode_t;

const char	*membrane_runtime_mode_name(membrane_runtime_mode_t m);
int			membrane_runtime_mode_from_name(const char *name,
				membrane_runtime_mode_t *out);
/* True for INJECT_Q8/INJECT_ADAPTIVE only. */
int			membrane_runtime_mode_is_inject(membrane_runtime_mode_t m);
/* True for SHADOW_Q8/SHADOW_ADAPTIVE only. */
int			membrane_runtime_mode_is_shadow(membrane_runtime_mode_t m);

/* ------------------------------------------------------------------ */
/* Injection scope filter (Phase 6)                                    */
/* ------------------------------------------------------------------ */

# define MEMBRANE_RUNTIME_TENSOR_K		0
# define MEMBRANE_RUNTIME_TENSOR_V		1
# define MEMBRANE_RUNTIME_TENSOR_BOTH	2

/*
 * Controlled injection scope: which (layer, tensor, absolute token
 * position) triples are eligible for write-back. Default (after
 * _init) is "everything eligible" -- callers narrow it down with the
 * _add_layer/_set_tensor/_set_token_range calls, matching the CLI's
 * --inject-layer/--inject-tensor/--inject-token-start/--inject-token-
 * end flags (see main.cpp). A layer never added via _add_layer is
 * ineligible once ANY _add_layer call has been made (i.e. the filter
 * becomes an allow-list on first use, matching "only inject where the
 * user explicitly asked").
 */
typedef struct s_membrane_runtime_scope
{
	int			has_layer_filter;
	uint8_t		layer_eligible[MEMBRANE_RUNTIME_MAX_LAYERS];
	int			tensor_filter;		/* MEMBRANE_RUNTIME_TENSOR_* */
	int			has_token_range;
	uint64_t	token_start;		/* absolute position, inclusive */
	uint64_t	token_end;			/* absolute position, inclusive */
}	membrane_runtime_scope_t;

void	membrane_runtime_scope_init(membrane_runtime_scope_t *s);
void	membrane_runtime_scope_add_layer(membrane_runtime_scope_t *s,
			uint32_t layer);
void	membrane_runtime_scope_set_tensor(membrane_runtime_scope_t *s,
			int tensor_filter);
void	membrane_runtime_scope_set_token_range(membrane_runtime_scope_t *s,
			uint64_t start, uint64_t end);
/* Layer/tensor check only (token range ignored). */
int		membrane_runtime_scope_matches_layer_tensor(
			const membrane_runtime_scope_t *s, uint32_t layer, int is_v);
/* Token-range check only. */
int		membrane_runtime_scope_matches_token(
			const membrane_runtime_scope_t *s, uint64_t abs_token_pos);

/* ------------------------------------------------------------------ */
/* Injection reconstruction (Phase 6)                                  */
/* ------------------------------------------------------------------ */

typedef struct s_membrane_runtime_inject_result
{
	uint64_t					tokens_in_scope;
	uint64_t					eligible_blocks;	/* full blocks attempted
													 * (in-scope tokens) */
	uint64_t					injected_blocks;	/* successfully
													 * reconstructed */
	uint64_t					failed_blocks;		/* 0 or 1: reconstruction
													 * stops at the first
													 * failure */
	uint64_t					native_tail_values;	/* per-token tail,
													 * in-scope tokens only,
													 * summed; left
													 * unmodified in
													 * values_f16 */
	membrane_workload_accum_t	accum;
	int							ok;	/* 1 iff every attempted block
									 * succeeded; 0 means values_f16 may
									 * be a PARTIAL reconstruction (safe
									 * to discard: every unreconstructed
									 * position still holds its
									 * original, genuine value) -- the
									 * caller MUST NOT write back on
									 * ok==0 */
}	membrane_runtime_inject_result_t;

/*
 * In-place: for every token in [0, n_tokens) whose absolute position
 * (token_start_abs + i) is in `scope`'s token range AND whose (layer,
 * is_v) matches `scope`'s layer/tensor filter, reconstructs
 * (quantize-then-dequantize under `policy`, adaptive threshold
 * hardcoded to MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR --
 * never a parameter here, so it can never be tuned by a caller) every
 * FULL MEMBRANE_RUNTIME_ELEMS_PER_BLOCK-sized block within that
 * token's own `elements_per_token`-sized slice of `values_f16`
 * (row-major: elements_per_token fastest-varying, then token -- i.e.
 * token i occupies values_f16[i*elements_per_token ..
 * (i+1)*elements_per_token)). A block never straddles a token
 * boundary. Any value in an out-of-scope token, or in an in-scope
 * token's trailing remainder (elements_per_token %
 * MEMBRANE_RUNTIME_ELEMS_PER_BLOCK), is left completely unmodified.
 *
 * Stops at the first block that fails to reconstruct -- either a hard
 * infrastructure failure from membrane_bench_process_block_reconstruct
 * (non-OK return), or a soft decode failure (OK return but
 * acc->decode_failures increments) -- since precision_policy.h treats
 * the latter as non-fatal for MEASUREMENT, but an unreconstructed
 * block can never be safely written back for INJECTION: out->ok is 0
 * in both cases, and every value from that point onward in
 * values_f16 is guaranteed untouched (still the genuine original).
 *
 * Returns MEMBRANE_OK (check out->ok for the write-back decision) or
 * MEMBRANE_ERR_INVALID_ARG (NULL scope/values_f16/out, or
 * elements_per_token == 0).
 */
membrane_status_t	membrane_runtime_inject_reconstruct(
						membrane_simd_backend_t backend,
						membrane_bench_policy_t policy,
						const membrane_runtime_scope_t *scope,
						uint32_t layer, int is_v, uint64_t token_start_abs,
						uint64_t n_tokens, uint64_t elements_per_token,
						uint16_t *values_f16,
						membrane_runtime_inject_result_t *out);

/* ------------------------------------------------------------------ */
/* Behavior comparison (Phase 6)                                       */
/* ------------------------------------------------------------------ */

typedef struct s_membrane_runtime_divergence
{
	int			identical;				/* 1 iff same length and every
										 * position matches */
	int64_t		first_divergence_step;	/* -1 if identical; otherwise the
										 * index of the first mismatched
										 * (or missing, if lengths
										 * differ) position */
	uint64_t	divergent_positions;	/* count of mismatched positions
										 * within the overlapping
										 * length */
}	membrane_runtime_divergence_t;

/* Pure position-by-position comparison of two token ID sequences (not
 * an equality check -- membrane_runtime_tokens_equal below is that).
 * Safe with na != nb (compares up to min(na, nb); a length mismatch
 * alone is enough for identical == 0, even if every overlapping
 * position matches). */
void	membrane_runtime_detect_divergence(const int32_t *a, size_t na,
			const int32_t *b, size_t nb,
			membrane_runtime_divergence_t *out);

typedef struct s_membrane_runtime_step_logit_diff
{
	double	abs_diff_max;
	double	abs_diff_mean;
	double	rel_l2;			/* ||b - a||_2 / ||a||_2; 0.0 if a is all-
							 * zero and b matches, 1.0 if a is all-zero
							 * and b does not (same convention as
							 * membrane_quant_rel_l2_error) */
	int		top1_preserved;	/* 1 iff argmax(a) == argmax(b) */
	int		topk_overlap;	/* |top-k(a) ∩ top-k(b)|, order-independent
							 * set overlap; -1 if k <= 0 (not computed) */
}	membrane_runtime_step_logit_diff_t;

/*
 * Compares two same-length logit vectors (n_vocab elements each, `a`
 * the reference/baseline, `b` the comparison/injected). `k` bounds the
 * top-k overlap computation (pass <= 0 to skip it -- out->topk_overlap
 * is set to -1). O(n_vocab) time, O(k) auxiliary space on the stack (k
 * is expected small, e.g. 5 or 10; capped internally at 64 -- a larger
 * k is clamped, never a buffer overrun). Returns MEMBRANE_ERR_INVALID_
 * ARG for NULL a/b/out or n_vocab <= 0.
 */
membrane_status_t	membrane_runtime_compare_logits(const float *a,
						const float *b, int32_t n_vocab, int k,
						membrane_runtime_step_logit_diff_t *out);

/* -log(softmax(logits)[target_token]), computed with the standard
 * max-subtraction for numerical stability. Pure. Returns a negative
 * sentinel-free NaN-free value; caller passes a valid target_token in
 * [0, n_vocab). */
double	membrane_runtime_nll(const float *logits, int32_t n_vocab,
			int32_t target_token);

typedef struct s_membrane_runtime_behavior_accum
	membrane_runtime_behavior_accum_t;

membrane_runtime_behavior_accum_t	*membrane_runtime_behavior_create(void);
void	membrane_runtime_behavior_destroy(
			membrane_runtime_behavior_accum_t *b);

/* Records one aligned (teacher-forced) step's comparison. `diff` is
 * typically the output of membrane_runtime_compare_logits() for that
 * step's (baseline logits, injected logits) pair; baseline_nll/
 * injected_nll are each run's own NLL of the SAME actual next
 * reference token at this step. */
void	membrane_runtime_behavior_add_step(
			membrane_runtime_behavior_accum_t *b,
			const membrane_runtime_step_logit_diff_t *diff,
			double baseline_nll, double injected_nll);

typedef struct s_membrane_runtime_behavior_summary
{
	uint64_t	steps;
	double		logit_abs_diff_mean;	/* mean of per-step means */
	double		logit_abs_diff_max;		/* max of per-step maxes */
	double		logit_rel_l2_mean;		/* mean of per-step rel_l2 */
	double		top1_preservation_rate;	/* fraction of steps with
										 * top1_preserved == 1 */
	double		topk_overlap_mean;		/* mean topk_overlap across
										 * steps where it was computed
										 * (k > 0); 0.0 if none were */
	double		mean_nll_baseline;
	double		mean_nll_injected;
	double		delta_nll;				/* mean_nll_injected -
										 * mean_nll_baseline */
}	membrane_runtime_behavior_summary_t;

void	membrane_runtime_behavior_finalize(
			const membrane_runtime_behavior_accum_t *b,
			membrane_runtime_behavior_summary_t *out);

/* ------------------------------------------------------------------ */
/* Collector / telemetry (Phase 5, extended for Phase 6)               */
/* ------------------------------------------------------------------ */

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
										 * decode_failures, q4_err/q8_err
										 * -- for INJECT modes, this IS
										 * the injection's own
										 * reconstruction accounting
										 * (the reconstruct pass doubles
										 * as the measurement) */

	/* Phase 6: injection-only fields (all 0 for BASELINE/SHADOW_*). */
	int							injection_requested;
	int							injection_succeeded;	/* only meaningful
														 * once the run has
														 * finished; 0 if
														 * injection_
														 * requested and
														 * ANY block ever
														 * failed */
	uint64_t					injection_eligible_blocks;
	uint64_t					injected_blocks;
	uint64_t					failed_blocks;
	uint64_t					injection_native_tail_values;
	uint32_t					layers_targeted;
	uint32_t					layers_injected;

	/* Phase 6: behavior comparison (only meaningful for INJECT modes
	 * when an aligned evaluation was run; free-running divergence is
	 * always computed for INJECT modes). */
	membrane_runtime_divergence_t			divergence;
	membrane_runtime_behavior_summary_t	behavior;
	int										behavior_available;

	double						membrane_seconds;	/* MEMBRANE-side work:
													 * tensor extraction +
													 * select+encode+decode
													 * +validate (+write-
													 * back for INJECT
													 * modes), timed by
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
													 * time spent in
													 * MEMBRANE processing,
													 * not extra time on
													 * top. */
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
 * SHADOW_Q8: MEMBRANE_BENCH_POLICY_Q8_ONLY; SHADOW_ADAPTIVE:
 * MEMBRANE_BENCH_POLICY_ADAPTIVE with the unmodified default
 * threshold, MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR; for an
 * INJECT_* mode collector this function still only OBSERVES/measures
 * -- use membrane_runtime_inject_reconstruct + membrane_runtime_
 * record_injection for the actual injection path). A trailing
 * remainder (n_elems % MEMBRANE_RUNTIME_ELEMS_PER_BLOCK) is never
 * processed or padded -- counted in the returned telemetry's
 * tail_values_excluded instead. is_v selects the K vs V block/byte
 * counters; layer must be < the max_layers this collector was created
 * with.
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

/*
 * Folds one tensor's membrane_runtime_inject_reconstruct() result into
 * the collector's running injection totals (injected_blocks,
 * failed_blocks, injection_eligible_blocks, injection_native_tail_
 * values, layers_targeted/layers_injected bitsets, and r->accum merged
 * into the collector's own accum) and marks the collector's injection-
 * failed flag if r->ok == 0. Call this for every layer/tensor
 * observed under an INJECT_* mode, regardless of whether anything was
 * actually in scope (r->eligible_blocks == 0 is a normal, harmless
 * no-op fold). layer must be < the collector's max_layers.
 */
void	membrane_runtime_record_injection(membrane_runtime_collector_t *c,
			uint32_t layer, int is_v,
			const membrane_runtime_inject_result_t *r);

/* True iff membrane_runtime_record_injection has ever been called with
 * r->ok == 0 for this collector. Sticky: once true, stays true. Used
 * by main.cpp to decide the process exit code -- an INJECT_* run with
 * this true must exit nonzero, never report success. */
int		membrane_runtime_injection_has_failed(
			const membrane_runtime_collector_t *c);

/* Records this run's free-running divergence (INJECT modes only;
 * harmless no-op recorded for other modes) and, if available, the
 * aligned-evaluation behavior summary. Either may be called at most
 * once per collector; calling finalize() before either is fine (the
 * telemetry fields stay zeroed / behavior_available stays 0). */
void	membrane_runtime_set_divergence(membrane_runtime_collector_t *c,
			const membrane_runtime_divergence_t *d);
void	membrane_runtime_set_behavior(membrane_runtime_collector_t *c,
			const membrane_runtime_behavior_summary_t *b);

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

/* Injection coverage: injected_blocks * MEMBRANE_RUNTIME_ELEMS_PER_
 * BLOCK values, divided by (that same numerator + injection_native_
 * tail_values) -- i.e. what fraction of every value in scope actually
 * got reconstructed vs. left as a native tail remainder. 0.0 if
 * nothing was in scope. */
double	membrane_runtime_injection_coverage_ratio(
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
