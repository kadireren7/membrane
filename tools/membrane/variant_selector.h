#ifndef MEMBRANE_VARIANT_SELECTOR_H
# define MEMBRANE_VARIANT_SELECTOR_H

# include <string>
# include <vector>
# include <cstdint>

# include "model_catalog.h"

/*
 * Mega Phase D, PR D2: given real hardware facts and a catalog family's
 * real GGUF variants, choose a sensible one automatically -- Section 9
 * of the task. Pure (no /proc read, no llama/ggml, no ggml_backend_dev_t)
 * so this is independently unit-testable with synthetic inputs, the
 * same established pattern host_memory_guard.h/gpu_policy.h already
 * use -- the caller (model_cmd.cpp) gathers the real facts via the
 * ALREADY-EXISTING membrane_read_host_meminfo() (runtime_session.h)
 * and reuses membrane_host_memory_guard_resolve() (host_memory_guard.h)
 * for the actual fit check, never a second, independently-drifting
 * "does this fit" implementation.
 *
 * Real, disclosed limitation (Section 9's own "GGUF variant sizes" is
 * the input, not an exact runtime footprint): the model is not
 * downloaded yet, so the catalog's own recorded size_bytes (the real,
 * verified download size) is used AS a proxy for host-resident weight
 * bytes -- close for these small/mid models at reasonable context
 * sizes, but not KV-cache-at-a-specific-context precise the way the
 * existing joint planner is once a model is actually loaded. See
 * docs/model-variant-selection.md.
 */

typedef struct s_membrane_variant_selector_input
{
	uint64_t	host_total_bytes;
	uint64_t	host_available_bytes;
	bool		host_available_known;	/* false = /proc unreadable --
										 * every variant fails closed
										 * (HOST_MEMORY_UNKNOWN), never
										 * assumed to fit */
}	membrane_variant_selector_input_t;

typedef struct s_membrane_variant_fit
{
	std::string	quant;
	bool		fits;
	std::string	reason_code;	/* a real MEMBRANE_HOST_GUARD_REASON_*
								 * value from host_memory_guard.h */
	std::string	reason;			/* human-readable detail */
}	membrane_variant_fit_t;

/*
 * The real, documented policy (Section 10 of the task):
 *   1. Evaluate every variant via host_memory_guard.h's own
 *      membrane_host_memory_guard_resolve(), treating the variant's
 *      catalog size_bytes as the host-resident weight-byte estimate.
 *   2. Among variants that fit, return the LARGEST (highest precision)
 *      -- never "smallest that fits," which would waste real available
 *      headroom for no reason.
 *   3. If none fit, returns NULL. out_all_considered (never NULL,
 *      always fully populated regardless of the outcome) reports every
 *      variant's own real fit/no-fit reason, so the caller can present
 *      real alternatives (Section 11: "fail with alternatives if none
 *      fit") rather than a bare "nothing works."
 * Deterministic: identical inputs always produce an identical result.
 */
const membrane_catalog_variant_t	*membrane_select_variant(
			const membrane_catalog_family_t &family,
			const membrane_variant_selector_input_t &hw,
			std::vector<membrane_variant_fit_t> *out_all_considered);

#endif
