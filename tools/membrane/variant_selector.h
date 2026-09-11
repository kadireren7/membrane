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

/*
 * Post-v1 product-polish, PR 1: the ONE authoritative "what should
 * `membrane model install` do with this variant's fit result" decision --
 * shared by both of `membrane model install`'s own call sites (a real,
 * manually-typed `--quant` override, and `membrane use`'s own internal
 * re-dispatch of the variant IT already selected via membrane_select_
 * variant() above) so they can no longer diverge.
 *
 * Root cause this closes (a real v1.0.0 user session): `membrane use`
 * previewed a variant as HOST_MEMORY_FIT (checked once, via membrane_
 * select_variant() above), then internally re-invoked `membrane model
 * install NAME --quant <that same variant>` -- which re-checks fit
 * against a FRESH host-memory read and, on a real host, found
 * HOST_MEMORY_INSUFFICIENT instead (available memory can genuinely
 * change between the two reads). Because that internal re-invocation is
 * indistinguishable, at the CLI-argument level, from a user who typed
 * `--quant` themselves, the fresh failure was silently treated as an
 * explicit override ("proceeding anyway because you asked for it
 * explicitly") -- even though the user never asked for that specific
 * variant; MEMBRANE's own automatic recommendation did.
 *
 * This function is the single place that distinguishes the two cases:
 *   - auto_selected == false (a real, manually-typed --quant): an
 *     unsafe variant is still honored -- MEMBRANE_VARIANT_INSTALL_FORCED
 *     -- Section 5's own "explicit user override must still work"
 *     requirement, unchanged from the pre-existing behavior.
 *   - auto_selected == true (MEMBRANE's own recommendation, re-checked
 *     immediately before the real install/download): an unsafe result
 *     here means available memory genuinely changed since the
 *     recommendation was made -- MEMBRANE_VARIANT_INSTALL_STALE --
 *     never silently downgraded to "the user forced it."
 * variant_fits is always considered.fits (see variant_selector.h's own
 * s_membrane_variant_fit) for the SAME family+quant already evaluated
 * by membrane_select_variant() -- never a second, independently-
 * computed fit boolean.
 * Pure, deterministic, no I/O -- directly unit-testable with synthetic
 * booleans, no real host meminfo or catalog/network involved.
 */
typedef enum e_membrane_variant_install_decision
{
	MEMBRANE_VARIANT_INSTALL_PROCEED,	/* fits (or fit not a concern) --
										 * install with no warning */
	MEMBRANE_VARIANT_INSTALL_STALE,	/* auto-selected, no longer fits --
										 * refuse; memory changed since
										 * the recommendation was made */
	MEMBRANE_VARIANT_INSTALL_FORCED,	/* a real, explicit --quant
										 * override does not fit -- warn,
										 * then proceed anyway */
}	membrane_variant_install_decision_t;

membrane_variant_install_decision_t	membrane_variant_install_decide(
			bool variant_fits, bool auto_selected);

#endif
