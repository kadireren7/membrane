#include "context_auto_cli.h"

#include <stdint.h>
#include <string.h>
#include <stdio.h>

static uint64_t	sat_add_u64(uint64_t a, uint64_t b)
{
	if (a > UINT64_MAX - b)
		return (UINT64_MAX);
	return (a + b);
}

int	membrane_ctxauto_minimum_required_context(uint64_t prompt_tokens,
			uint64_t gen_tokens, uint64_t *out)
{
	if (out == NULL)
		return (0);
	*out = sat_add_u64(sat_add_u64(prompt_tokens, gen_tokens),
			MEMBRANE_CTXAUTO_MARGIN_TOKENS);
	return (1);
}

static void	add_suggestion(membrane_ctxauto_suggestions_t *out,
				const char *text)
{
	if (out->count >= MEMBRANE_CTXAUTO_MAX_SUGGESTIONS)
		return ;
	snprintf(out->text[out->count], MEMBRANE_CTXAUTO_SUGGESTION_MAX, "%s",
		text);
	out->count++;
}

void	membrane_ctxauto_suggest(const char *status,
			const char *representative_reason_code, int gen_tokens,
			int requested_precision_explicit,
			int requested_placement_explicit,
			membrane_ctxauto_suggestions_t *out)
{
	if (out == NULL)
		return ;
	memset(out, 0, sizeof(*out));
	if (status == NULL)
		return ;
	if (strcmp(status, MEMBRANE_CTXREC_STATUS_PLANNER_REJECTED_ALL) == 0
		&& representative_reason_code != NULL
		&& strcmp(representative_reason_code,
			MEMBRANE_HOST_GUARD_REASON_UNKNOWN) == 0)
	{
		add_suggestion(out, "use an explicit --ctx N if you intentionally "
			"accept manual sizing without a host-memory check "
			"(preserves current safety policy -- --ctx auto never "
			"proceeds on unknown host memory)");
		return ;
	}
	if (strcmp(status, MEMBRANE_CTXREC_STATUS_NO_FEASIBLE_CONTEXT) == 0
		|| strcmp(status, MEMBRANE_CTXREC_STATUS_PLANNER_REJECTED_ALL) == 0)
	{
		if (gen_tokens > 0)
			add_suggestion(out, "reduce --gen-tokens");
		if (!requested_precision_explicit)
			add_suggestion(out, "try --kv q5 (smaller KV footprint than "
				"the default) if your model supports it");
		if (!requested_placement_explicit)
			add_suggestion(out, "try --kv-placement cpu if you are "
				"currently requesting GPU-resident KV");
		add_suggestion(out, "close memory-heavy applications and try again");
		return ;
	}
	if (strcmp(status, MEMBRANE_CTXREC_STATUS_MINIMUM_EXCEEDS_MODEL_MAX) == 0)
	{
		add_suggestion(out, "shorten the prompt");
		if (gen_tokens > 0)
			add_suggestion(out, "reduce --gen-tokens");
		add_suggestion(out, "choose a model with a larger context window");
		return ;
	}
	/* MODEL_MAX_CONTEXT_UNKNOWN/INVALID_MODEL_MAX_CONTEXT: the recovery
	 * (manual sizing) must not be framed as risk-free -- Section 25's
	 * own caveat ("only if this preserves current safety policy") --
	 * so this is deliberately the only suggestion offered, worded as an
	 * explicit trade-off rather than a plain fix. */
	if (strcmp(status, MEMBRANE_CTXREC_STATUS_MODEL_MAX_CONTEXT_UNKNOWN) == 0
		|| strcmp(status,
			MEMBRANE_CTXREC_STATUS_INVALID_MODEL_MAX_CONTEXT) == 0)
	{
		add_suggestion(out, "use an explicit --ctx N instead if you are "
			"willing to size context manually for this model");
		return ;
	}
}
