#ifndef MEMBRANE_RUN_CONTEXT_AUTO_CLI_H
# define MEMBRANE_RUN_CONTEXT_AUTO_CLI_H

# include <stddef.h>
# include <stdint.h>

# include "context_recommender.h"
# include "host_memory_guard.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Phase 35: the PURE, llama-free half of `--ctx auto`'s CLI adapter --
 * no model load, no tokenization, no device/host fact gathering, no
 * /proc read (Section 14 of the Phase 35 task: "Do not put planning
 * logic into CLI parsing" -- this module owns the small pieces of that
 * adapter that ARE pure arithmetic/mapping, kept separately testable
 * from the llama-aware parts that must live in main.cpp, matching
 * Section 39's own "keep parsing vs planning vs runtime adapter
 * boundaries testable" instruction). Real fact-gathering (GGUF scan,
 * vocab-only tokenization, host/device reads) lives in main.cpp, which
 * calls this module with already-gathered plain data -- exactly the
 * same pattern context_recommender.h itself already uses.
 */

/*
 * Section 6/7/18 of the Phase 35 task: minimum_required_context =
 * prompt_tokens + gen_tokens + the SAME fixed margin (8) main.cpp's
 * own pre-existing, unrelated "--ctx not given at all" default
 * (ctx_size = prompt_tokens.size() + gen_tokens + 8) already uses --
 * reused for consistency, not a new number. Checked/saturating
 * uint64_t arithmetic (Section 18): a pathological prompt_tokens or
 * gen_tokens value never wraps into a too-small minimum. Returns 0
 * (out left at UINT64_MAX, the unmistakably-invalid sentinel) only if
 * out is NULL; otherwise always succeeds (there is no failure mode for
 * addition alone -- overflow saturates rather than erroring, and a
 * saturated minimum will correctly fail MINIMUM_EXCEEDS_MODEL_MAX one
 * layer up in membrane_ctxrec_resolve() rather than silently wrapping
 * small).
 */
# define MEMBRANE_CTXAUTO_MARGIN_TOKENS	((uint64_t)8)

int	membrane_ctxauto_minimum_required_context(uint64_t prompt_tokens,
		uint64_t gen_tokens, uint64_t *out);

/*
 * Section 25 of the Phase 35 task: maps a membrane_ctxrec_result_t's
 * status (and, for the two statuses that need it, the caller's own
 * model_max_context/minimum_required_context) to a small, fixed set of
 * actionable suggestion strings -- never more than
 * MEMBRANE_CTXAUTO_MAX_SUGGESTIONS, never an impossible/inapplicable
 * fix (e.g. "use --kv q5" is never suggested when q5 was already the
 * explicit, failing request). out_count is always set, even to 0 (a
 * status this function has no specific suggestion for -- the caller's
 * own generic message still applies, just with no extra suggestions
 * appended).
 */
# define MEMBRANE_CTXAUTO_MAX_SUGGESTIONS	4
# define MEMBRANE_CTXAUTO_SUGGESTION_MAX	224

typedef struct s_membrane_ctxauto_suggestions
{
	char	text[MEMBRANE_CTXAUTO_MAX_SUGGESTIONS][MEMBRANE_CTXAUTO_SUGGESTION_MAX];
	size_t	count;
}	membrane_ctxauto_suggestions_t;

/*
 * status: the overall membrane_ctxrec_result_t::status.
 * representative_reason_code: the winning (or, on failure, first-
 * evaluated) candidate's own membrane_ctxrec_evaluated_t::reason_code
 * -- distinguishes a HOST_MEMORY_* cause from a generic joint-planner
 * rejection sharing the same top-level PLANNER_REJECTED_ALL status
 * (host-memory failures reuse that status rather than a dedicated
 * top-level one, per Phase 34's own taxonomy -- see context_
 * recommender.h). May be NULL/empty if not applicable (e.g. status
 * itself already fully determines the cause).
 * requested_precision_explicit/requested_placement_explicit: whether
 * --kv/--kv-placement were EXPLICITLY given (so this function never
 * suggests changing a dimension the user already fixed -- Section 25:
 * "Suggestions must be legal/contextual").
 */
void	membrane_ctxauto_suggest(const char *status,
			const char *representative_reason_code, int gen_tokens,
			int requested_precision_explicit,
			int requested_placement_explicit,
			membrane_ctxauto_suggestions_t *out);

# ifdef __cplusplus
}
# endif

#endif
