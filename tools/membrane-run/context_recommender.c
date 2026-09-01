#include "context_recommender.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

size_t	membrane_ctxrec_generate_candidates(uint64_t model_max_context,
			uint64_t minimum_required_context, uint64_t *out_candidates,
			size_t max_out)
{
	uint64_t	floor_ctx;
	uint64_t	c;
	size_t		n;
	size_t		hard_max;

	if (out_candidates == NULL || max_out == 0 || model_max_context == 0)
		return (0);
	if (minimum_required_context > model_max_context)
		return (0);
	floor_ctx = MEMBRANE_CTXREC_MIN_CANDIDATE;
	if (minimum_required_context > floor_ctx)
		floor_ctx = minimum_required_context;
	if (floor_ctx > model_max_context)
		return (0);
	hard_max = max_out;
	if (hard_max > MEMBRANE_CTXREC_MAX_CANDIDATES)
		hard_max = MEMBRANE_CTXREC_MAX_CANDIDATES;
	n = 0;
	c = floor_ctx;
	/* Reserve the last slot for model_max_context's own guaranteed
	 * final entry (Section 7: "include model_max_context exactly as
	 * the final candidate when appropriate") whenever more than one
	 * slot is available; with exactly one slot, the loop body below
	 * never executes (c is never < model_max_context and still fits a
	 * doubling step within hard_max - 1 == 0), so the single slot is
	 * filled by the model_max_context append below instead. */
	while (n < hard_max && n + 1 < hard_max && c < model_max_context)
	{
		out_candidates[n] = c;
		n++;
		if (c > UINT64_MAX / 2)
			break ;
		c *= 2;
	}
	if (n < hard_max && (n == 0 || out_candidates[n - 1] != model_max_context))
	{
		out_candidates[n] = model_max_context;
		n++;
	}
	return (n);
}

static void	set_status(membrane_ctxrec_result_t *out, const char *status,
				const char *detail)
{
	snprintf(out->status, sizeof(out->status), "%s", status);
	snprintf(out->reason, sizeof(out->reason), "%s: %s", status, detail);
}

static void	build_joint_request(const membrane_ctxrec_request_t *req,
				const membrane_ctxrec_candidate_input_t *ci,
				membrane_joint_plan_request_t *jreq)
{
	memset(jreq, 0, sizeof(*jreq));
	jreq->n_layer_all = req->n_layer_all;
	jreq->bytes_per_layer = req->bytes_per_layer;
	jreq->output_role_bytes = req->output_role_bytes;
	jreq->arch_name = req->arch_name;
	jreq->n_embd = req->n_embd;
	jreq->n_head = req->n_head;
	jreq->n_head_kv = req->n_head_kv;
	jreq->ctx_size = (uint32_t)ci->ctx;
	jreq->device_free_bytes = req->device_free_bytes;
	jreq->device_total_bytes = req->device_total_bytes;
	jreq->kv_bytes_native = ci->kv_bytes_native;
	jreq->kv_bytes_q8 = ci->kv_bytes_q8;
	jreq->kv_bytes_q5 = ci->kv_bytes_q5;
	jreq->precision_request = req->precision_request;
	jreq->gpu_layers_request = req->gpu_layers_request;
	jreq->kv_placement_mode = req->kv_placement_mode;
	jreq->want_kv_budget = req->want_kv_budget;
	jreq->kv_budget_bytes = req->kv_budget_bytes;
}

/* Evaluates ONE candidate, filling out->evaluated[idx] and returning 1
 * iff it was feasible (out->selected_plan is populated by the caller
 * loop only for the best-so-far feasible candidate -- this function
 * has no notion of "best", it only evaluates in isolation, matching
 * Section 15's "no unsafe early break, evaluate the complete bounded
 * set" requirement). */
static int	evaluate_one(const membrane_ctxrec_request_t *req,
				const membrane_ctxrec_candidate_input_t *ci,
				membrane_ctxrec_evaluated_t *ev,
				membrane_joint_plan_result_t *jres_out)
{
	membrane_joint_plan_request_t	jreq;

	memset(ev, 0, sizeof(*ev));
	ev->ctx = ci->ctx;
	if (ci->ctx < req->minimum_required_context
		|| ci->ctx > req->model_max_context)
	{
		snprintf(ev->reason_code, sizeof(ev->reason_code), "%s",
			MEMBRANE_CTXREC_REASON_OUT_OF_DECLARED_BOUNDS);
		return (0);
	}
	if (ci->ctx > UINT32_MAX)
	{
		snprintf(ev->reason_code, sizeof(ev->reason_code), "%s",
			MEMBRANE_CTXREC_REASON_CTX_EXCEEDS_UINT32);
		return (0);
	}
	build_joint_request(req, ci, &jreq);
	membrane_joint_plan_resolve(&jreq, jres_out);
	ev->feasible = jres_out->ok;
	snprintf(ev->reason_code, sizeof(ev->reason_code), "%s",
		jres_out->reason_code);
	if (!jres_out->ok)
		return (0);
	{
		const membrane_joint_candidate_t	*win =
				&jres_out->candidates[jres_out->selected_index];

		ev->selected_gpu_layers = win->gpu_layers;
		ev->selected_kv_precision = win->kv_precision;
		ev->selected_kv_placement = win->kv_placement;
		ev->host_resident = (win->estimated_host_kv_bytes > 0)
			|| (win->gpu_layers < req->n_layer_all);
	}
	return (1);
}

int	membrane_ctxrec_resolve(const membrane_ctxrec_request_t *req,
			membrane_ctxrec_result_t *out)
{
	size_t	i;
	int		best;

	if (out == NULL)
		return (0);
	memset(out, 0, sizeof(*out));
	out->hardware_fit_index = -1;
	if (req == NULL || req->n_layer_all <= 0)
	{
		set_status(out, MEMBRANE_CTXREC_STATUS_INVALID_INPUT,
			"null request or non-positive n_layer_all");
		return (0);
	}
	if (!req->model_max_context_known)
	{
		set_status(out, MEMBRANE_CTXREC_STATUS_MODEL_MAX_CONTEXT_UNKNOWN,
			"model metadata did not expose a context-length ceiling");
		return (0);
	}
	if (req->model_max_context == 0)
	{
		set_status(out, MEMBRANE_CTXREC_STATUS_INVALID_MODEL_MAX_CONTEXT,
			"model_max_context is 0");
		return (0);
	}
	out->model_max_context = req->model_max_context;
	out->minimum_required_context = req->minimum_required_context;
	if (req->minimum_required_context > req->model_max_context)
	{
		set_status(out, MEMBRANE_CTXREC_STATUS_MINIMUM_EXCEEDS_MODEL_MAX,
			"minimum_required_context exceeds model_max_context");
		return (0);
	}
	if (req->candidate_count > MEMBRANE_CTXREC_MAX_CANDIDATES)
	{
		set_status(out, MEMBRANE_CTXREC_STATUS_INVALID_INPUT,
			"candidate_count exceeds MEMBRANE_CTXREC_MAX_CANDIDATES");
		return (0);
	}
	if (req->candidate_count == 0)
	{
		set_status(out, MEMBRANE_CTXREC_STATUS_NO_FEASIBLE_CONTEXT,
			"no candidates to evaluate (effective floor exceeds "
			"model_max_context, or the caller generated none)");
		return (0);
	}
	out->candidate_count = req->candidate_count;
	out->evaluated_count = req->candidate_count;
	best = -1;
	for (i = 0; i < req->candidate_count; ++i)
	{
		membrane_joint_plan_result_t	jres;

		if (evaluate_one(req, &req->candidates[i], &out->evaluated[i],
				&jres)
			&& (best == -1 || out->evaluated[i].ctx
				> out->evaluated[best].ctx))
		{
			best = (int)i;
			out->selected_plan = jres;
		}
	}
	if (best == -1)
	{
		set_status(out, MEMBRANE_CTXREC_STATUS_PLANNER_REJECTED_ALL,
			"every candidate was rejected by the joint planner");
		return (0);
	}
	out->hardware_fit_index = best;
	out->hardware_fit_context = out->evaluated[best].ctx;
	out->recommended_context = out->hardware_fit_context;
	snprintf(out->recommendation_policy, sizeof(out->recommendation_policy),
		"%s", MEMBRANE_CTXREC_POLICY_MAX_ESTIMATED_FIT);
	out->host_memory_unvalidated = out->evaluated[best].host_resident;
	out->ok = 1;
	snprintf(out->status, sizeof(out->status), "%s", MEMBRANE_CTXREC_STATUS_OK);
	snprintf(out->reason, sizeof(out->reason), "%s: hardware_fit_context=%llu "
		"is the largest of %zu evaluated candidates the joint planner "
		"accepted", MEMBRANE_CTXREC_STATUS_OK,
		(unsigned long long)out->hardware_fit_context, out->evaluated_count);
	snprintf(out->explanation, sizeof(out->explanation),
		"Recommended ctx=%llu (policy: %s) -- the largest of %zu evaluated "
		"candidates (model max %llu) with a legal plan under current "
		"memory estimates. This is an estimated fit, not a guaranteed "
		"absence of OOM (see docs/context-recommendation.md).",
		(unsigned long long)out->recommended_context,
		out->recommendation_policy, out->evaluated_count,
		(unsigned long long)out->model_max_context);
	return (1);
}
