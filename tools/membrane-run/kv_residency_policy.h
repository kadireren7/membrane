#ifndef MEMBRANE_RUN_KV_RESIDENCY_POLICY_H
# define MEMBRANE_RUN_KV_RESIDENCY_POLICY_H

# include <stddef.h>
# include <stdint.h>

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Phase 12H: llama-free (no ggml/llama header) pure arithmetic for a
 * STATIC KV residency plan -- given model weight placement is already
 * decided (gpu_policy.h's job, upstream of this module and NEVER fed
 * back into by it -- Section 21: weight placement must stay identical
 * regardless of --kv-placement), decide how many of a model's KV
 * layers can safely stay GPU-resident, with the remainder CPU-
 * resident, before context construction. No runtime movement: the map
 * this produces is handed once to llama_context_params.kv_dev_override
 * at construction time (see docs/kv-residency.md) and never revisited
 * during decode -- deliberately simpler and safer than the Phase 12F
 * research dynamic scheduler, which this module does NOT productize.
 *
 * Same testable-without-a-GPU-or-model pattern as gpu_policy.h/
 * compat_check.h/adaptive_kv_policy.h: every input is a plain integer
 * the caller already has (device_free/total from
 * ggml_backend_dev_memory(), weight_bytes_selected from gpu_policy's
 * own already-computed estimated_model_bytes, kv_bytes_per_layer from
 * the same ggml_row_size()-based formula membrane-run's existing
 * native_kv_bytes()/q8_kv_bytes()/q5_kv_bytes() already use, divided
 * by n_layer).
 */

# define MEMBRANE_KV_PLACEMENT_DEFAULT	0	/* no override: kv_dev_override
											 * left NULL, KV follows
											 * upstream llama.cpp's own
											 * per-layer device exactly as
											 * before this feature existed
											 * -- Section 4/19: the only
											 * mode with zero behavioral
											 * change from pre-Phase-12H. */
# define MEMBRANE_KV_PLACEMENT_GPU		1	/* all KV GPU-resident, or fail
											 * closed */
# define MEMBRANE_KV_PLACEMENT_CPU		2	/* all KV CPU-resident (always
											 * satisfiable: CPU is always
											 * available on every build) */
# define MEMBRANE_KV_PLACEMENT_AUTO		3	/* planner splits per-layer to
											 * maximize GPU residency
											 * subject to safe budget --
											 * see membrane_kv_residency_
											 * resolve()'s doc comment for
											 * why "maximize GPU" was
											 * chosen for Phase 12H
											 * specifically (conservative
											 * compatibility, NOT a
											 * performance claim -- Phase
											 * 12G found no GPU-KV
											 * throughput advantage on
											 * tested hardware). */

/* Matches tools/membrane-llama-runtime/runtime_core.h's
 * MEMBRANE_RUNTIME_MAX_LAYERS -- the largest real model layer count
 * this project's tooling budgets for anywhere. Kept as this module's
 * OWN constant (not an #include of runtime_core.h, which pulls in
 * llama-enabled collector state this module has no business knowing
 * about) -- a `_Static_assert`-style comment, not an automatic check,
 * ties the two: if runtime_core.h's constant ever changes, this one
 * should be reviewed too. */
# define MEMBRANE_KV_RESIDENCY_MAX_LAYERS	512u

/* Stable fail-closed/outcome reason codes (Section 18) -- never change
 * the MEANING of an already-shipped string, only add new ones. Also
 * used verbatim as the JSON `placement_reason` field. */
# define MEMBRANE_KV_PLACEMENT_REASON_GPU_FULL				"PLACEMENT_GPU_FULL"
# define MEMBRANE_KV_PLACEMENT_REASON_CPU_FULL				"PLACEMENT_CPU_FULL"
# define MEMBRANE_KV_PLACEMENT_REASON_AUTO_SPLIT			"PLACEMENT_AUTO_SPLIT"
# define MEMBRANE_KV_PLACEMENT_REASON_DEFAULT_UNCHANGED	"PLACEMENT_DEFAULT_UNCHANGED"
# define MEMBRANE_KV_PLACEMENT_REASON_BUDGET_INSUFFICIENT	"PLACEMENT_BUDGET_INSUFFICIENT"
# define MEMBRANE_KV_PLACEMENT_REASON_BACKEND_UNAVAILABLE	"PLACEMENT_BACKEND_UNAVAILABLE"
# define MEMBRANE_KV_PLACEMENT_REASON_INVALID_CONFIG		"PLACEMENT_INVALID_CONFIG"

/* Runtime margin (Section 12): an ADDITIONAL conservative allowance
 * beyond gpu_policy.h's own reserve_for()-equivalent
 * (MEMBRANE_GPU_RESERVE_FIXED_BYTES/_PCT, reused verbatim by this
 * module -- see kv_residency_policy.c's reserve_for()), specific to
 * per-layer device-split KV construction: Phase 12 research
 * (results/phase12/kv-device-override/) measured real allocator/
 * compute-buffer overhead beyond nominal KV bytes when a context is
 * built with a non-uniform per-layer device map (more graph splits at
 * lower GPU-KV ratios -- Phase 12G's graph_splits.json). This phase's
 * own real-hardware validation (results/v0.3/kv-residency-
 * productization/capacity_uplift.json) did NOT instrument an exact
 * predicted-vs-real-free-VRAM byte gap (no such probe was built this
 * phase) -- what IS directly measured is the practical consequence:
 * at ctx=28500 on a real GTX 1650, the DEFAULT path (no margin, no
 * placement flexibility -- attempts full GPU KV directly) fails with
 * a real Vulkan out-of-device-memory error that gpu_policy's own
 * pre-load ESTIMATE did not predict, while this module's margin-aware
 * AUTO/CPU plans, computed against the SAME estimate minus this
 * margin, succeed at the exact same configuration. That is the
 * evidence these constants are doing real, needed work here -- not a
 * direct numeric confirmation that they are exactly right, which
 * remains open (see capacity_uplift.json's limitations). */
# define MEMBRANE_KV_RESIDENCY_MARGIN_FIXED_BYTES	((uint64_t)64 * 1024 * 1024)
# define MEMBRANE_KV_RESIDENCY_MARGIN_PCT			5

typedef struct s_membrane_kv_residency_result
{
	int			ok;
	int32_t		n_layer;
	int32_t		gpu_kv_layers;
	int32_t		cpu_kv_layers;
	/* Deterministic per-layer map: 1 = GPU-resident KV, 0 =
	 * CPU-resident KV. Only indices [0, n_layer) are meaningful.
	 * Ascending layer index is filled GPU-first (Section 9: simplest
	 * stable order -- earliest layers preferred for GPU residency,
	 * NOT derived from Phase 11's precision-sensitivity ranking, which
	 * answered a different question). */
	uint8_t		layer_on_gpu[MEMBRANE_KV_RESIDENCY_MAX_LAYERS];
	uint64_t	kv_bytes_per_layer;
	uint64_t	total_kv_bytes;
	uint64_t	gpu_kv_bytes;
	uint64_t	cpu_kv_bytes;
	uint64_t	safety_reserve_bytes;
	uint64_t	runtime_margin_bytes;
	uint64_t	budget_for_kv_bytes;	/* device_free - reserve - weights
										 * - compute_buffer - margin;
										 * the actual remainder this
										 * module had to place GPU KV
										 * into */
	char		reason[256];
}	membrane_kv_residency_result_t;

/*
 * placement_mode: one of MEMBRANE_KV_PLACEMENT_{DEFAULT,GPU,CPU,AUTO}.
 * n_layer: model's real total layer count (llama_model_n_layer()),
 *   must be in [1, MEMBRANE_KV_RESIDENCY_MAX_LAYERS].
 * kv_bytes_per_layer: real per-layer (K+V combined) KV byte cost at
 *   the requested precision -- same ggml_row_size()-based arithmetic
 *   as native_kv_bytes()/q8_kv_bytes()/q5_kv_bytes(), for ONE layer
 *   (total_kv_bytes / n_layer, since KV byte cost is uniform per layer
 *   for a non-hybrid, non-SWA model -- see kv-device-override.patch's
 *   own scope note).
 * device_free_bytes/device_total_bytes: from ggml_backend_dev_memory(),
 *   the SAME real values gpu_policy_resolve() was already called with.
 * weight_bytes_selected: gpu_policy_resolve()'s own
 *   out->estimated_model_bytes for the ALREADY-RESOLVED --gpu-layers
 *   value -- this module NEVER recomputes or second-guesses weight
 *   placement, only consumes its result (Section 21).
 * compute_buffer_estimate_bytes: reserved for a future, separately-
 *   measured ggml compute-buffer estimate; membrane-run's call site
 *   currently passes 0 (compute-buffer risk is absorbed into the
 *   existing reserve_for()+runtime-margin floor, matching gpu_policy.h's
 *   own established convention of a blanket conservative reserve
 *   rather than a modeled compute-buffer number -- documented
 *   simplification, not a silent gap).
 *
 * Returns 1 (out->ok=1) if a valid plan was produced (which, for AUTO,
 * includes an all-CPU-KV result when GPU budget is 0 -- CPU is always
 * safe/available on a GPU-capable build, so AUTO never fails purely
 * for lack of GPU budget). Returns 0 (out->ok=0, out->reason filled)
 * only for GPU-mode-doesn't-fit or invalid-input cases -- callers must
 * fail closed, never silently retry with a different mode.
 */
int	membrane_kv_residency_resolve(int placement_mode, int32_t n_layer,
		uint64_t kv_bytes_per_layer, uint64_t device_free_bytes,
		uint64_t device_total_bytes, uint64_t weight_bytes_selected,
		uint64_t compute_buffer_estimate_bytes,
		membrane_kv_residency_result_t *out);

const char	*membrane_kv_placement_mode_name(int placement_mode);

/*
 * Phase 13.1: the fix for a real gpu_policy/kv_residency integration
 * gap -- gpu_policy_resolve()'s weight-layer preflight (main.cpp's
 * resolve_gpu_config(), called BEFORE this module, BEFORE model load)
 * previously reserved GPU budget for the model's FULL KV size
 * unconditionally, regardless of --kv-placement, because gpu_policy.h
 * has (and keeps) zero awareness of KV placement modes by design (weight
 * placement must never depend on --kv-placement -- see this header's
 * own top comment and Section 21 of docs/kv-residency.md). That made
 * `--kv-placement cpu` (KV never GPU-resident at all) and
 * `--kv-placement auto` (this module's own AUTO branch below already
 * degrades to CPU-resident KV under pressure and NEVER fails for lack
 * of GPU KV budget) reserve/require GPU memory for KV that either
 * never lands on the GPU (cpu) or is never guaranteed to (auto) --
 * needlessly shrinking the weight-layer budget, or rejecting an
 * explicit --gpu-layers N that would truly fit once KV correctly
 * placed itself off-GPU.
 *
 * This function is the single place that answers "how much of the
 * model's full KV byte estimate should gpu_policy_resolve()'s weight
 * preflight assume needs GPU budget, given the requested placement
 * mode" -- called by main.cpp ONCE per candidate precision, BEFORE
 * each membrane_gpu_policy_resolve() call, in place of the raw
 * full_kv_bytes_estimate. It does not touch, call, or duplicate
 * anything in gpu_policy.h -- gpu_policy_resolve()'s own signature and
 * logic are unchanged; only the number a caller passes in for
 * kv_bytes_estimate changes, and only for cpu/auto placement:
 *
 *   DEFAULT: full_kv_bytes_estimate, unchanged. Section 4's own zero-
 *            behavioral-change requirement for DEFAULT means the
 *            preflight must keep assuming the pre-Phase-12H
 *            KV-follows-weights behavior exactly as before this
 *            function existed.
 *   GPU:     full_kv_bytes_estimate, unchanged. The user explicitly
 *            asked for 100% GPU-resident KV, so the preflight
 *            correctly needs the full amount -- membrane_kv_residency_
 *            resolve()'s own GPU branch below still independently
 *            re-checks and fails closed if it doesn't fit, this is
 *            only about not needlessly rejecting/shrinking earlier.
 *   CPU:     0. No KV byte will ever be GPU-resident under this mode
 *            (membrane_kv_residency_resolve()'s CPU branch always
 *            succeeds, unconditionally, below) -- there is nothing to
 *            reserve GPU budget for.
 *   AUTO:    0. This module's own AUTO branch below places whatever
 *            GPU budget remains AFTER weights are already selected,
 *            and never fails for lack of it (falls back to more
 *            CPU-resident layers instead) -- reserving anything at the
 *            weight-preflight stage would double-count against this
 *            module's own later accounting for no safety benefit,
 *            only a needlessly smaller weight-layer budget.
 *
 * Any other value falls through to full_kv_bytes_estimate unchanged
 * (fail-safe: the pre-existing conservative behavior), matching this
 * module's own invalid-placement_mode handling in
 * membrane_kv_residency_resolve() rather than silently returning 0.
 */
uint64_t	membrane_kv_policy_preflight_reservation(int placement_mode,
				uint64_t full_kv_bytes_estimate);

# ifdef __cplusplus
}
# endif

#endif
