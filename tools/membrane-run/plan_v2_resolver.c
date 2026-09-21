#include <string.h>
#include <stdio.h>

#include "plan_v2_resolver.h"

static size_t	clamp_variant_count(size_t n)
{
	if (n > MEMBRANE_PLAN_V2_MAX_VARIANTS)
		return (MEMBRANE_PLAN_V2_MAX_VARIANTS);
	return (n);
}

static void	resolve_joint_candidate(const membrane_plan_v2_variant_t *vreq,
				const membrane_plan_v2_request_t *req,
				membrane_plan_t *plan)
{
	membrane_ctxrec_request_t	creq;
	membrane_ctxrec_result_t	rec;

	creq = vreq->ctxrec_req;
	/* Part 9 (enforced here, not merely assumed of the caller): every
	 * variant uses the exact same single hardware snapshot. */
	creq.host_total_bytes = req->host_total_bytes;
	creq.host_available_bytes = req->host_available_bytes;
	creq.host_available_known = req->host_available_known;
	creq.device_free_bytes = req->device_free_bytes;
	creq.device_total_bytes = req->device_total_bytes;
	membrane_ctxrec_resolve(&creq, &rec);
	membrane_plan_assemble(&rec, &creq, &vreq->ctxrec_meta, &vreq->identity,
		&vreq->variant, NULL, plan);
}

static void	resolve_coarse_candidate(const membrane_plan_v2_variant_t *vreq,
				const membrane_plan_v2_request_t *req,
				membrane_plan_t *plan)
{
	/* rec == NULL: membrane_plan_assemble() already pushes
	 * MEMBRANE_PLAN_REASON_MODEL_METADATA_UNAVAILABLE and sets
	 * feasible=1 by default (Part 5's own disclosure requirement,
	 * unchanged from G1's plan_catalog_only_model() path) -- this
	 * function only overrides feasibility when the coarse guard itself
	 * failed, exactly like G1's own push_no_variant_fits_note() call
	 * site did for a single selected variant. */
	membrane_plan_assemble(NULL, NULL, &vreq->ctxrec_meta, &vreq->identity,
		&vreq->variant, NULL, plan);
	plan->hardware.host_total_bytes = req->host_total_bytes;
	plan->hardware.host_available_bytes = req->host_available_bytes;
	plan->hardware.host_available_known = req->host_available_known;
	if (!vreq->coarse_fits)
	{
		/* limiting_resource stays within membrane_plan.h's own small
		 * top-level MEMBRANE_PLAN_REASON_* vocabulary (its own field
		 * comment: "MEMBRANE_PLAN_REASON_* or ''") -- same convention
		 * G1's plan_catalog_only_model() already used. The raw,
		 * detailed subsystem code (host_memory_guard.h's own
		 * MEMBRANE_HOST_GUARD_REASON_*, e.g. HOST_MEMORY_INSUFFICIENT)
		 * is only ever echoed verbatim in reasons[]::code, where a
		 * detailed, subsystem-specific code is the established
		 * contract (Part 12 of the G2 task: "use real existing reason
		 * codes"). */
		plan->feasibility.feasible = 0;
		snprintf(plan->feasibility.limiting_resource,
			sizeof(plan->feasibility.limiting_resource), "%s",
			MEMBRANE_PLAN_REASON_HOST_MEMORY_LIMIT);
		if (plan->reason_count < MEMBRANE_PLAN_MAX_REASONS)
		{
			membrane_plan_reason_t	*r = &plan->reasons[plan->reason_count];

			snprintf(r->code, sizeof(r->code), "%s",
				vreq->coarse_reason_code[0] != '\0'
					? vreq->coarse_reason_code
					: MEMBRANE_PLAN_REASON_HOST_MEMORY_LIMIT);
			snprintf(r->detail, sizeof(r->detail), "%s",
				vreq->coarse_reason[0] != '\0' ? vreq->coarse_reason
					: "coarse catalog-size host-memory estimate did not fit");
			plan->reason_count++;
		}
	}
}

static int	candidate_tier(const membrane_plan_t *plan)
{
	/* Higher is better: tier 1 = a real evaluated decision, tier 0 =
	 * feasible but only a coarse estimate, tier -1 = infeasible. */
	if (!plan->feasibility.feasible)
		return (-1);
	if (plan->decisions.has_decisions)
		return (1);
	return (0);
}

static void	pick_selected(membrane_plan_v2_result_t *out, size_t n)
{
	size_t	i;
	int		best_tier;
	int		best_index;

	best_tier = -1;
	best_index = -1;
	for (i = 0; i < n; ++i)
	{
		int	tier = candidate_tier(&out->candidates[i]);

		if (tier > best_tier)
		{
			best_tier = tier;
			best_index = (int)i;
		}
	}
	out->selected_index = best_index;
	out->has_selected = (best_index >= 0);
	out->feasible = out->has_selected
		&& out->candidates[best_index].feasibility.feasible;
}

static void	collect_alternatives(membrane_plan_v2_result_t *out, size_t n)
{
	size_t	k;
	int		pass;
	size_t	i;

	k = 0;
	/* Same tier ordering as selection itself (tier 1 candidates before
	 * tier 0 ones), Part 7's own bounded shortlist. */
	for (pass = 1; pass >= 0 && k < MEMBRANE_PLAN_V2_MAX_ALTERNATIVES; --pass)
	{
		for (i = 0; i < n && k < MEMBRANE_PLAN_V2_MAX_ALTERNATIVES; ++i)
		{
			if ((int)i == out->selected_index)
				continue ;
			if (candidate_tier(&out->candidates[i]) != pass)
				continue ;
			out->alternative_indices[k] = (int)i;
			k++;
		}
	}
	out->alternative_count = k;
}

static void	build_explanation(membrane_plan_v2_result_t *out, size_t n)
{
	if (out->has_selected)
	{
		const membrane_plan_t	*sel = &out->candidates[out->selected_index];

		snprintf(out->explanation, sizeof(out->explanation),
			"selected variant '%s' (candidate %d of %zu evaluated, %zu "
			"feasible alternative(s)) -- %s",
			sel->identity.variant_known ? sel->identity.variant : "unknown",
			out->selected_index + 1, n, out->alternative_count,
			sel->explanation[0] != '\0' ? sel->explanation
				: "see this candidate's own reasons");
	}
	else
		snprintf(out->explanation, sizeof(out->explanation),
			"no evaluated variant (%zu considered) has a feasible plan "
			"under current constraints -- see each candidate's own reasons",
			n);
}

int	membrane_plan_v2_resolve(const membrane_plan_v2_request_t *req,
		membrane_plan_v2_result_t *out)
{
	size_t	n;
	size_t	i;

	if (out == NULL)
		return (0);
	memset(out, 0, sizeof(*out));
	out->schema_version = MEMBRANE_PLAN_V2_SCHEMA_VERSION;
	snprintf(out->policy_version, sizeof(out->policy_version), "%s",
		MEMBRANE_PLAN_V2_POLICY_VERSION);
	out->selected_index = -1;
	if (req == NULL)
		return (0);
	n = clamp_variant_count(req->variant_count);
	out->candidate_count = n;
	for (i = 0; i < n; ++i)
	{
		const membrane_plan_v2_variant_t	*vreq = &req->variants[i];

		if (vreq->hparams_known)
			resolve_joint_candidate(vreq, req, &out->candidates[i]);
		else
			resolve_coarse_candidate(vreq, req, &out->candidates[i]);
	}
	pick_selected(out, n);
	collect_alternatives(out, n);
	build_explanation(out, n);
	return (out->feasible);
}
