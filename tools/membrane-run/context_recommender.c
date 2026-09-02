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

/* Phase 34: this candidate's own host-resident weight-byte estimate.
 *
 * total_weight_bytes (every real GGUF tensor, gpu_device.h's own
 * total_bytes) minus a GPU-offload credit -- but that credit
 * deliberately calls membrane_joint_estimate_gpu_weight_bytes() with
 * output_role_bytes forced to 0, NEVER crediting the output-role
 * tensor(s) to the GPU side for THIS subtraction, even though the real
 * joint-planner candidate correctly does credit it for its own GPU-
 * budget fit check. Why this is safe, not an inconsistency: for an
 * UNTIED-embedding model, output.weight is one real tensor already
 * counted once inside total_weight_bytes -- withholding its credit
 * here means this function OVER-counts host-resident bytes by
 * output_role_bytes when gpu_layers >= 1 (the safe, conservative
 * direction for a safety guard). For a TIED-embedding model,
 * llama.cpp creates a SEPARATE, GPU-eligible duplicate of
 * token_embd.weight for the output role (gpu_device.h's own
 * output_role_bytes field comment) -- a real allocation NOT present in
 * total_weight_bytes at all, so crediting it to the GPU side here
 * would UNDER-count (the original, always-CPU-resident token_embd
 * copy stays on the host regardless); withholding the credit gives the
 * EXACT right answer in that case instead. Since this module has no
 * cheap way to distinguish tied from untied here (that distinction
 * lives in gpu_device.cpp's own have_output local, not exposed as a
 * field), always withholding the credit is correct for tied models and
 * conservative (never wrong in the unsafe direction) for untied ones --
 * see docs/host-memory-guard.md's "Host memory components" section for
 * the full derivation. Checked subtraction, clamped to 0. */
static uint64_t	host_weight_bytes_for(const membrane_ctxrec_request_t *req,
					int32_t gpu_layers)
{
	uint64_t	gpu_credit;

	gpu_credit = membrane_joint_estimate_gpu_weight_bytes(gpu_layers,
			req->bytes_per_layer, 0);
	return (req->total_weight_bytes > gpu_credit
			? req->total_weight_bytes - gpu_credit : 0);
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
	snprintf(ev->reason_code, sizeof(ev->reason_code), "%s",
		jres_out->reason_code);
	if (!jres_out->ok)
		return (0);
	{
		const membrane_joint_candidate_t	*win =
				&jres_out->candidates[jres_out->selected_index];
		membrane_host_guard_request_t		hreq;
		membrane_host_guard_result_t		hres;

		ev->selected_gpu_layers = win->gpu_layers;
		ev->selected_kv_precision = win->kv_precision;
		ev->selected_kv_placement = win->kv_placement;
		ev->host_resident = (win->estimated_host_kv_bytes > 0)
			|| (win->gpu_layers < req->n_layer_all);
		memset(&hreq, 0, sizeof(hreq));
		hreq.host_total_bytes = req->host_total_bytes;
		hreq.host_available_bytes = req->host_available_bytes;
		hreq.host_available_known = req->host_available_known;
		hreq.host_weight_bytes = host_weight_bytes_for(req, win->gpu_layers);
		hreq.host_kv_bytes = win->estimated_host_kv_bytes;
		membrane_host_memory_guard_resolve(&hreq, &hres);
		ev->host_memory_checked = 1;
		ev->host_memory_fit = hres.ok;
		ev->host_required_bytes = hres.required_bytes;
		ev->host_reserve_bytes = hres.reserve_bytes;
		/* Section 11 of the Phase 34 task: a candidate the joint
		 * planner accepted is never counted feasible here unless the
		 * host-memory guard also accepts it -- this is what makes
		 * "OK unless host-memory feasibility was validated" true. On
		 * failure, the guard's own reason_code (HOST_MEMORY_*)
		 * overrides the joint planner's reason_code as the actual
		 * blocking cause. */
		ev->feasible = hres.ok;
		if (!hres.ok)
			snprintf(ev->reason_code, sizeof(ev->reason_code), "%s",
				hres.reason_code);
	}
	return (ev->feasible);
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
	out->host_memory_checked = out->evaluated[best].host_memory_checked;
	out->host_memory_fit = out->evaluated[best].host_memory_fit;
	out->host_required_bytes = out->evaluated[best].host_required_bytes;
	out->host_available_bytes = req->host_available_bytes;
	out->host_reserve_bytes = out->evaluated[best].host_reserve_bytes;
	out->ok = 1;
	snprintf(out->status, sizeof(out->status), "%s", MEMBRANE_CTXREC_STATUS_OK);
	snprintf(out->reason, sizeof(out->reason), "%s: hardware_fit_context=%llu "
		"is the largest of %zu evaluated candidates the joint planner "
		"accepted", MEMBRANE_CTXREC_STATUS_OK,
		(unsigned long long)out->hardware_fit_context, out->evaluated_count);
	snprintf(out->explanation, sizeof(out->explanation),
		"Recommended ctx=%llu (policy: %s) -- the largest of %zu evaluated "
		"candidates (model max %llu) with a legal plan under current GPU "
		"and host-memory estimates. This is an estimated fit, not a "
		"guaranteed absence of OOM (see docs/context-recommendation.md "
		"and docs/host-memory-guard.md).",
		(unsigned long long)out->recommended_context,
		out->recommendation_policy, out->evaluated_count,
		(unsigned long long)out->model_max_context);
	return (1);
}
