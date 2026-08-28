#ifndef MEMBRANE_LLAMA_RUNTIME_KV_STORE_TELEMETRY_H
# define MEMBRANE_LLAMA_RUNTIME_KV_STORE_TELEMETRY_H

# include <stddef.h>
# include <stdint.h>
# include <stdio.h>

# ifdef __cplusplus
extern "C" {
# endif

/* KV storage mode constants -- shared by cli_parse.h (--kv-store) and
 * decode_loop.h (run_kv_store_pass), so this file (already the
 * llama-free home of everything else KV-storage-related) is the one
 * place that owns them, not either CLI-specific header. */
# define MEMBRANE_KV_STORE_NATIVE	0
# define MEMBRANE_KV_STORE_Q8		1
/* Phase 10C: KV cache tensors genuinely GGML_TYPE_Q5_1 -- real
 * quantization, not a shadow copy, same authoritative-storage shape
 * as Q8 above. "q5" on the CLI always means Q5_1 specifically (never
 * Q5_0) -- see tools/membrane-run/product_cli.cpp's --kv help text
 * and results/v0.4/q5-validation.json (this branch) for the
 * productization derisk data; the prior Q4_0/Q5_0/Q5_1 comparison
 * this choice is based on lives on the experiment/q5-kv-evaluation
 * branch (results/phase10/q5-kv-evaluation.json there), not committed
 * on this branch. */
# define MEMBRANE_KV_STORE_Q5		2

/*
 * Product Phase 7: llama-free core for compressed-KV-storage
 * measurement. Like runtime_core.h, this file and its .c never include
 * a ggml/llama header -- llama_hook.cpp is the only place that reads
 * real ggml_type row sizes (via ggml_row_size()) and real llama_context
 * memory (via /proc + getrusage, which are OS-level, not llama-
 * specific) and hands plain integers in here. This is what
 * test_kv_store_telemetry.c exercises unconditionally, no
 * MEMBRANE_ENABLE_LLAMA needed.
 *
 * This module is deliberately NOT about quantize/dequantize math --
 * Phase 7's storage compression is llama.cpp's own ggml_cpy()/
 * ggml_mul_mat() type-conversion machinery (see docs/live-runtime.md),
 * not a MEMBRANE-authored codec. This module only accounts for and
 * reports the real byte/memory consequences of that storage choice.
 */

/* ------------------------------------------------------------------ */
/* Process memory snapshot (Section 7 checkpoints)                     */
/* ------------------------------------------------------------------ */

typedef struct s_membrane_kv_store_rss
{
	uint64_t	vm_rss_kb;		/* /proc/self/status VmRSS, 0 if unreadable */
	uint64_t	vm_hwm_kb;		/* /proc/self/status VmHWM (peak RSS so far),
								 * 0 if unreadable */
	uint64_t	ru_maxrss_kb;	/* getrusage(RUSAGE_SELF).ru_maxrss -- Linux
								 * reports this in KB already; always
								 * populated (POSIX), independent cross-
								 * check against VmHWM */
	int			proc_status_ok;	/* 1 iff /proc/self/status was readable
									 * and both VmRSS/VmHWM were parsed */
}	membrane_kv_store_rss_t;

/* ------------------------------------------------------------------ */
/* KV cache byte accounting (Section 6)                                */
/* ------------------------------------------------------------------ */

/*
 * bytes_per_token_k/v are the REAL ggml row size in bytes for one
 * token's worth of K (resp. V) data at a given layer's cache dtype --
 * the caller computes this via ggml_row_size(type, n_embd_k_gqa) (a
 * real ggml function, not a MEMBRANE-invented formula) since that is
 * the exact arithmetic llama.cpp's own tensor allocation uses. This
 * function just multiplies out across layers/kv_size -- pure
 * accounting, still cross-checked at the call site against real RSS
 * deltas and (where available) real ggml_nbytes() of the live cache
 * tensors themselves.
 */
typedef struct s_membrane_kv_store_bytes
{
	uint64_t	n_layer;
	uint64_t	kv_size;			/* context size the cache was sized for */
	uint64_t	bytes_per_token_k;	/* real ggml_row_size() for one token, K */
	uint64_t	bytes_per_token_v;	/* real ggml_row_size() for one token, V */
}	membrane_kv_store_bytes_t;

/* n_layer * kv_size * (bytes_per_token_k + bytes_per_token_v). Returns
 * 0 if b is NULL. */
uint64_t	membrane_kv_store_total_bytes(
				const membrane_kv_store_bytes_t *b);

/* ------------------------------------------------------------------ */
/* Telemetry (Section 18)                                              */
/* ------------------------------------------------------------------ */

typedef struct s_membrane_kv_store_telemetry
{
	int			kv_store_requested;	/* --kv-store was given */
	const char	*kv_store_mode_name;	/* "native" | "q8" */
	uint32_t	ctx_size;
	uint64_t	generated_tokens;	/* actual tokens produced by the
									 * canonical/reported pass -- may be
									 * less than --gen-tokens if the
									 * model hit an end-of-generation
									 * token first */

	/* storage */
	uint64_t	native_kv_allocated_bytes;		/* what an F16 cache at this
												 * ctx_size WOULD allocate --
												 * always computed, for the
												 * ratio, even in q8 mode */
	uint64_t	compressed_kv_allocated_bytes;	/* the REAL allocated bytes
												 * for the run's actual
												 * cache dtype (== native_*
												 * bytes when mode is
												 * "native") */
	uint64_t	metadata_bytes;		/* KV cache bookkeeping the accounting
									 * here cannot see (ggml_context
									 * overhead, tensor structs) --
									 * reported as 0 unless a real source
									 * is available (see llama_hook.cpp) */
	uint64_t	scratch_peak_bytes;	/* max transient decode/compare buffer
									 * bytes MEMBRANE itself allocated this
									 * run -- must stay << full-context
									 * size (Section 13) */

	/* memory (RSS checkpoints, Section 7/18) */
	membrane_kv_store_rss_t	rss_after_model_load;
	membrane_kv_store_rss_t	rss_after_context;
	membrane_kv_store_rss_t	rss_after_prompt;
	membrane_kv_store_rss_t	rss_final;
	membrane_kv_store_rss_t	rss_peak;	/* max VmHWM/ru_maxrss seen across
										 * every checkpoint above */

	/* performance */
	double		prompt_tok_per_s;
	double		generation_tok_per_s;
	double		encode_seconds;	/* MEMBRANE-side accounting/instrumentation
								 * time, NOT llama's own ggml_cpy quantize
								 * (that is inside inference_seconds --
								 * there is no separate MEMBRANE encode
								 * step in this architecture, see the
								 * file-level comment above) */
	double		decode_seconds;	/* likewise for read-side instrumentation */

	/* Phase 24: raw stage durations in milliseconds, additive alongside
	 * the throughput fields above (which are derived from the SAME
	 * underlying prompt_seconds/gen_seconds this file's own decode_loop.cpp
	 * already measures via clock_gettime(CLOCK_MONOTONIC) -- these just
	 * expose the raw duration too, not a new measurement). context_create_ms
	 * times ONLY the llama_init_from_model() call itself (context/KV
	 * tensor allocation), never conflated with prompt_ms (prefill) or
	 * generation_ms (decode) -- see docs/performance-profiling.md.
	 * first_token_ms is the single decode step that produces the FIRST
	 * generated token (a stage-boundary measurement, not per-token
	 * logging -- Section 25 of the Phase 24 task), -1.0 if no token was
	 * generated (e.g. immediate EOG or a failed pass) rather than a
	 * fabricated 0. */
	double		context_create_ms;
	double		prompt_ms;
	double		generation_ms;
	double		first_token_ms;

	/* quality (only meaningful when kv_store_mode == q8; zeroed/self-
	 * identical for native, matching how Phase 6 zeroes injection
	 * fields for BASELINE) */
	int			quality_available;
	int			token_identity;		/* 1 iff free-running q8 tokens ==
									 * free-running native tokens */
	int32_t		first_divergence;	/* -1 if identical */
	double		logit_rel_l2;
	double		top1_preservation;
	double		delta_nll;

	/* compression */
	uint64_t	q4_blocks;
	uint64_t	q8_blocks;
	uint64_t	encoded_blocks;

	int			no_fallback_occurred;	/* always 1 in this implementation --
										 * see Section 17 */
}	membrane_kv_store_telemetry_t;

/* Reads the current process's RSS via /proc/self/status (VmRSS/VmHWM)
 * and getrusage(RUSAGE_SELF) (ru_maxrss). Never fails outright -- if
 * /proc is unavailable, proc_status_ok is 0 and vm_rss_kb/vm_hwm_kb
 * stay 0, but ru_maxrss_kb (POSIX, portable) is still populated. */
void	membrane_kv_store_read_rss(membrane_kv_store_rss_t *out);

/* out->rss_peak = componentwise max(a, b) of vm_hwm_kb/ru_maxrss_kb
 * (vm_rss_kb is a snapshot, not meaningful to "max"). */
void	membrane_kv_store_rss_max(const membrane_kv_store_rss_t *a,
			const membrane_kv_store_rss_t *b,
			membrane_kv_store_rss_t *out);

void	membrane_kv_store_print_human(const membrane_kv_store_telemetry_t *t,
			FILE *out);
void	membrane_kv_store_print_json(const membrane_kv_store_telemetry_t *t,
			FILE *out);

# ifdef __cplusplus
}
# endif

#endif
