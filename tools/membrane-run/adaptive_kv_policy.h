#ifndef MEMBRANE_RUN_ADAPTIVE_KV_POLICY_H
# define MEMBRANE_RUN_ADAPTIVE_KV_POLICY_H

# include <stddef.h>
# include <stdint.h>

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Phase 11A: llama-free (no ggml/llama header) pure arithmetic for
 * choosing ONE authoritative whole-cache KV storage mode -- Q8_0 or
 * Q5_1 -- for --kv adaptive. This is NOT mixed precision: the output
 * is a single mode for the entire context, never a per-layer/per-
 * block choice.
 *
 * This module does not compute KV byte estimates or GPU layer counts
 * itself -- those already exist (main.cpp's kv_bytes_for_mode(),
 * gpu_policy.h's membrane_gpu_policy_resolve()) and are reused
 * as-is, called once per candidate mode by the caller. This module's
 * only job is the DECISION between two already-evaluated candidates,
 * so the storage/layer arithmetic can never drift out of sync between
 * modes (Section 9/6 of the Phase 11A spec) and the policy stays pure
 * and unit-testable with synthetic inputs (test_adaptive_kv_policy.c).
 *
 * Policy principle: prefer Q8 when it safely fits with the same
 * practical benefit as Q5; use Q5 only when Q8 would lose full GPU
 * residency, fail the memory guard, or exceed a configured budget,
 * while Q5 still fits safely. Never choose Q5 merely because it uses
 * less memory -- see membrane_adaptive_kv_resolve()'s doc comment for
 * the exact decision order.
 */

typedef struct s_membrane_adaptive_kv_candidate
{
	int			valid;			/* 1 if this mode fits within every
								 * constraint the caller checked
								 * (memory guard, optional --kv-
								 * budget-mib) at all -- 0 if it does
								 * not fit under any layer count/
								 * budget. */
	int			full_residency;	/* GPU: 1 iff valid AND every
									 * target layer was placed on the
									 * device (selected_layers ==
									 * the model's real total layer
									 * count). CPU: always equal to
									 * valid (there is no partial-
									 * residency concept off-GPU). */
	int32_t		selected_layers;	/* GPU: the real layer count this
									 * mode's own membrane_gpu_policy_
									 * resolve() call selected
									 * (meaningful only if valid).
									 * CPU: always 0, unused. */
	uint64_t	kv_bytes;			/* real KV byte estimate for this
									 * mode/ctx (kv_bytes_for_mode()),
									 * echoed through into the result
									 * for telemetry -- never
									 * recomputed here. */
}	membrane_adaptive_kv_candidate_t;

/* Fixed, stable reason-code set (Section 4) -- machine-readable,
 * small, and never renamed once shipped (JSON's adaptive_reason field
 * is a public contract). */
# define MEMBRANE_ADAPTIVE_REASON_Q8_FITS \
	"Q8_FITS"
# define MEMBRANE_ADAPTIVE_REASON_Q8_FULL_RESIDENCY \
	"Q8_FULL_RESIDENCY"
# define MEMBRANE_ADAPTIVE_REASON_Q5_REQUIRED_FOR_FULL_RESIDENCY \
	"Q5_REQUIRED_FOR_FULL_RESIDENCY"
# define MEMBRANE_ADAPTIVE_REASON_Q5_REQUIRED_FOR_MEMORY_GUARD \
	"Q5_REQUIRED_FOR_MEMORY_GUARD"
# define MEMBRANE_ADAPTIVE_REASON_Q5_ONLY_COMPRESSED_MODE_THAT_FITS \
	"Q5_ONLY_COMPRESSED_MODE_THAT_FITS"
# define MEMBRANE_ADAPTIVE_REASON_NO_COMPRESSED_MODE_FITS \
	"NO_COMPRESSED_MODE_FITS"
# define MEMBRANE_ADAPTIVE_REASON_CPU_ADAPTIVE_Q8_DEFAULT \
	"CPU_ADAPTIVE_Q8_DEFAULT"
# define MEMBRANE_ADAPTIVE_REASON_CPU_MEMORY_PRESSURE_Q5 \
	"CPU_MEMORY_PRESSURE_Q5"

/* selected_mode uses the same numbering as kv_store_telemetry.h's
 * MEMBRANE_KV_STORE_Q8 (1) / MEMBRANE_KV_STORE_Q5 (2) -- not included
 * here to keep this header dependency-free, same convention
 * compat_check.h already documents for its own kv_mode parameter. */
# define MEMBRANE_ADAPTIVE_KV_MODE_Q8	1
# define MEMBRANE_ADAPTIVE_KV_MODE_Q5	2

typedef struct s_membrane_adaptive_kv_result
{
	int			ok;					/* 1 if a mode was selected, 0 =
									 * fail closed (never native) */
	int			selected_mode;		/* MEMBRANE_ADAPTIVE_KV_MODE_Q8/Q5,
									 * meaningful only if ok */
	int32_t		selected_layers;	/* echoed from the winning
									 * candidate (GPU only, 0 for
									 * CPU), meaningful only if ok */
	uint64_t	selected_kv_bytes;	/* echoed from the winning
									 * candidate, meaningful only if
									 * ok */
	char		reason[40];			/* one of the MEMBRANE_ADAPTIVE_
									 * REASON_* strings above, always
									 * set (both on success and
									 * failure) */
}	membrane_adaptive_kv_result_t;

/*
 * q8/q5: the two already-evaluated candidates (see membrane_adaptive_
 * kv_candidate_t above) -- both computed by the caller via the exact
 * same real arithmetic every explicit --kv q8/q5 request already uses.
 * is_gpu_backend: 0 for CPU-only (full_residency/selected_layers on
 * both candidates are ignored; only valid is consulted), nonzero for
 * a GPU backend (full-residency/partial-offload comparison applies).
 *
 * Decision order (Section 7, GPU):
 *   1. Q8 fits AND reaches full residency -> Q8 (Q8_FULL_RESIDENCY).
 *   2. Else Q5 fits AND reaches full residency -> Q5 (Q5_REQUIRED_
 *      FOR_FULL_RESIDENCY).
 *   3. Else compare safe partial offload: if both fit and select the
 *      SAME layer count, prefer Q8 (Q8_FITS) -- never sacrifice
 *      quality for zero practical benefit. If Q5 fits with MORE
 *      layers than Q8, choose Q5 (Q5_REQUIRED_FOR_MEMORY_GUARD). If
 *      only Q5 fits at all, choose Q5 (Q5_ONLY_COMPRESSED_MODE_THAT_
 *      FITS). If only Q8 fits, choose Q8 (Q8_FITS). If neither fits,
 *      fail closed (NO_COMPRESSED_MODE_FITS).
 *
 * CPU (Section 5): Q8 unless it is invalid (e.g. exceeds an explicit
 * --kv-budget-mib) and Q5 is valid -> CPU_MEMORY_PRESSURE_Q5. Neither
 * valid -> fail closed.
 *
 * Deterministic: identical inputs always produce identical output, no
 * randomness, no I/O, no clock. Returns out->ok (also the function's
 * return value); never silently falls back to native -- callers must
 * treat ok==0 as a hard failure (Section 11).
 */
int	membrane_adaptive_kv_resolve(
		const membrane_adaptive_kv_candidate_t *q8,
		const membrane_adaptive_kv_candidate_t *q5, int is_gpu_backend,
		membrane_adaptive_kv_result_t *out);

# ifdef __cplusplus
}
# endif

#endif
