#ifndef MEMBRANE_RUN_HOST_MEMORY_GUARD_H
# define MEMBRANE_RUN_HOST_MEMORY_GUARD_H

# include <stddef.h>
# include <stdint.h>

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Phase 34: llama-free (no ggml/llama header), no `/proc` read, no
 * device access -- pure arithmetic for a real host-RAM capacity check,
 * same testable-without-a-model/GPU pattern as gpu_policy.h. Closes
 * (for the SCOPE named below) the gap Phase 33 disclosed and audited
 * directly: joint_planner.h's own candidate field comment already
 * said "fits_host... always 1 today (no host-RAM capacity guard
 * exists in the product)", and main.cpp's only host-memory signal
 * (MEMBRANE_WARNING_HOST_MEMORY_PRESSURE) is explicitly observability-
 * only, "never influenced any pass/fail decision."
 *
 * SCOPE (read carefully -- this is not a universal host-memory
 * guarantee): this module validates HOST-RESIDENT MODEL WEIGHT AND KV
 * BYTES ONLY -- the Primary Objective's own wording ("plans that leave
 * model-weight or KV bytes resident on CPU/host memory"). It does
 * NOT, and cannot yet, validate the process/backend BASELINE host
 * overhead that exists even for a fully GPU-resident plan (real
 * evidence, results/planner-accuracy/measurements.json: a 100%-GPU-
 * offloaded Vulkan run, zero host-resident weight/KV bytes, still
 * showed ~248 MiB of real host RSS after model load -- Vulkan driver/
 * command-buffer overhead, unrelated to weight/KV residency, and NOT
 * modeled here). A candidate with host_weight_bytes==0 and
 * host_kv_bytes==0 passes this guard trivially (HOST_MEMORY_ZERO_
 * REQUIREMENT) -- that means "this guard found nothing of ITS OWN
 * scope to check," never "this plan needs zero real host RAM." See
 * docs/host-memory-guard.md's "Residual uncertainty" section.
 *
 * Reserve policy derivation (Section 5 of the Phase 34 task: no
 * arbitrary 10/20/25% default) -- real evidence, THIS repository's own
 * models directory, CPU-only build, SmolLM2-135M-Instruct-f16.gguf:
 *
 *   - This phase's own 4 fresh samples (native/q8 KV modes, ctx
 *     2048/4096, real membrane-run --json invocations): combined
 *     "loader+runtime baseline" + "context-creation transient"
 *     residual (real RSS minus the real GGUF weight-byte total minus
 *     the real ggml_row_size()-based KV-byte prediction) clustered at
 *     22.7-26.6 MiB -- a ~8-10% ratio of the model's own real weight
 *     bytes (~258 MiB).
 *   - Phase 19's own PRE-EXISTING evidence
 *     (results/planner-accuracy/measurements.json), the SAME exact
 *     model+config, a DIFFERENT session/host-state: combined residual
 *     was ~151.7-153.7 MiB instead -- a ~58% ratio.
 *
 * That is a real, disclosed ~6x discrepancy for the IDENTICAL
 * model+config across two sessions on the same 5.6 GiB dev host (this
 * phase's own measurement session ran with 4.3-4.6 GiB of swap already
 * in use throughout -- see results/host-memory-guard/validation.json's
 * host notes) -- consistent with genuine RSS-measurement noise under
 * real memory/swap pressure, not a fixed per-model constant. Rather
 * than silently keep only the smaller, more flattering sample set,
 * MEMBRANE_HOST_RESERVE_FIXED_BYTES (256 MiB) is set to comfortably
 * exceed the WORSE of the two real, measured worst cases (~153.7 MiB)
 * with real margin -- not a tight point estimate, and not an invented
 * round number either. MEMBRANE_HOST_RESERVE_PCT (10%) is kept as a
 * minimal, deliberately-conservative placeholder for how this might
 * scale to a LARGER model this phase has no direct evidence for (it
 * does not bind for anything actually measured -- the fixed floor
 * dominates in every real sample above); applied to this candidate's
 * own (host_weight_bytes + host_kv_bytes), NOT to total host RAM
 * (unlike gpu_policy.h's device-total-relative percentage -- see this
 * header's own top comment for why: loader/runtime overhead is
 * causally connected to how much of THIS model is host-resident, not
 * to how much total system RAM happens to be installed).
 * EXPLICITLY NOT VALIDATED AT LARGER MODEL SCALE this phase (memory-
 * constrained dev host, ~250-900 MiB available during this phase's own
 * measurement session) -- disclosed, not silently assumed to
 * generalize. See docs/host-memory-guard.md's "Residual uncertainty".
 */

# define MEMBRANE_HOST_RESERVE_FIXED_BYTES	((uint64_t)256 * 1024 * 1024)
# define MEMBRANE_HOST_RESERVE_PCT			10

# define MEMBRANE_HOST_GUARD_REASON_FIT					"HOST_MEMORY_FIT"
# define MEMBRANE_HOST_GUARD_REASON_ZERO_REQUIREMENT		"HOST_MEMORY_ZERO_REQUIREMENT"
# define MEMBRANE_HOST_GUARD_REASON_INSUFFICIENT			"HOST_MEMORY_INSUFFICIENT"
# define MEMBRANE_HOST_GUARD_REASON_UNKNOWN				"HOST_MEMORY_UNKNOWN"
# define MEMBRANE_HOST_GUARD_REASON_INVALID_CONFIG			"HOST_MEMORY_GUARD_INVALID_CONFIG"

typedef struct s_membrane_host_guard_request
{
	/* From read_host_meminfo()-equivalent (main.cpp's own real
	 * /proc/meminfo reader -- this module never reads /proc itself,
	 * Section 9 of the Phase 34 task: reuse the single existing
	 * source). host_available_known distinguishes "read and it was
	 * genuinely 0" (never realistic, but handled) from "could not be
	 * read at all" (non-Linux host, unreadable /proc) -- the latter
	 * fails closed (HOST_MEMORY_UNKNOWN), never silently treated as
	 * infinite availability (Section 10). */
	uint64_t	host_total_bytes;
	uint64_t	host_available_bytes;
	int			host_available_known;

	/* This candidate's own real, already-computed host-resident byte
	 * estimates -- see this header's own top comment for exactly what
	 * is and is not covered. Both 0 is the trivial "nothing of this
	 * guard's scope to check" case. */
	uint64_t	host_weight_bytes;
	uint64_t	host_kv_bytes;
}	membrane_host_guard_request_t;

typedef struct s_membrane_host_guard_result
{
	int			ok;
	char		reason_code[40];	/* MEMBRANE_HOST_GUARD_REASON_* */
	char		reason[256];

	uint64_t	required_bytes;			/* host_weight_bytes +
										 * host_kv_bytes, checked add */
	uint64_t	reserve_bytes;			/* 0 when required_bytes == 0 --
										 * see this header's own top
										 * comment: no reserve is applied
										 * when nothing is in this
										 * guard's scope */
	uint64_t	available_after_reserve;	/* host_available_bytes -
										 * reserve_bytes, clamped to 0,
										 * meaningless (0) if
										 * !host_available_known */
}	membrane_host_guard_result_t;

/* The reserve formula alone, exposed for reuse (same pattern as
 * gpu_policy.h's own membrane_gpu_reserve_bytes()) -- 0 when
 * host_resident_bytes is 0 (Section 5: no arbitrary floor applied to
 * a candidate this guard has nothing to check). */
uint64_t	membrane_host_memory_reserve_bytes(uint64_t host_resident_bytes);

/*
 * Fails closed (out->ok = 0) when:
 *   - out is NULL (returns 0, out left untouched)
 *   - req is NULL (INVALID_CONFIG)
 *   - host_weight_bytes/host_kv_bytes are both 0: succeeds trivially
 *     (ZERO_REQUIREMENT) -- see this header's own top-comment scope
 *     note, this is NOT a host-memory guarantee for that candidate
 *   - !host_available_known: UNKNOWN (never assumes infinite RAM)
 *   - required_bytes + reserve_bytes > host_available_bytes:
 *     INSUFFICIENT
 * Checked, saturating uint64_t arithmetic throughout (Section 18) --
 * required_bytes/reserve_bytes never overflow into a small value that
 * would let an oversized candidate pass; available_after_reserve never
 * underflows.
 *
 * Deterministic: identical inputs always produce an identical result.
 * Returns out->ok (also the function's return value).
 */
int	membrane_host_memory_guard_resolve(
		const membrane_host_guard_request_t *req,
		membrane_host_guard_result_t *out);

# ifdef __cplusplus
}
# endif

#endif
