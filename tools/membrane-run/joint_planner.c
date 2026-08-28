#include "joint_planner.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "compat_check.h"
#include "gpu_policy.h"
#include "adaptive_kv_policy.h"
#include "kv_residency_policy.h"

/* Checked/saturating uint64 arithmetic (Section 25): a real device's
 * byte counts never come close to these limits, but a corrupted/
 * adversarial bytes_per_layer or n_layer_all must still never wrap
 * into a small value that lets an oversized candidate pass a budget
 * check -- saturate to UINT64_MAX instead. */
static uint64_t	sat_mul_u64(uint64_t a, uint64_t b)
{
	if (a != 0 && b > UINT64_MAX / a)
		return (UINT64_MAX);
	return (a * b);
}

static uint64_t	sat_add_u64(uint64_t a, uint64_t b)
{
	if (a > UINT64_MAX - b)
		return (UINT64_MAX);
	return (a + b);
}

uint64_t	membrane_joint_estimate_gpu_weight_bytes(int32_t v,
				uint64_t bytes_per_layer, uint64_t output_role_bytes)
{
	if (v <= 0)
		return (0);
	return (sat_add_u64(sat_mul_u64((uint64_t)(v - 1), bytes_per_layer),
			output_role_bytes));
}

/* Inverse of membrane_joint_estimate_gpu_weight_bytes(): largest V in
 * [0, n_layer_all] whose corrected estimate fits within budget_bytes.
 * budget_bytes < output_role_bytes means not even V=1 fits (the fixed
 * output-role cost alone exceeds budget) -- returns 0. bytes_per_layer
 * == 0 is defensively treated the same way (never divides by zero);
 * main.cpp's own pre-Phase-20 callers already fail closed before
 * calling this module at all when the pre-load estimate has no usable
 * bytes_per_layer, so this is a defensive fallback, not the expected
 * path. The "+1" is saturating and the result is clamped to
 * n_layer_all (itself always <= INT32_MAX, guaranteed by
 * membrane_joint_plan_resolve()'s own n_layer_all > 0 int32_t input)
 * BEFORE the int32_t conversion, so that cast is always safe. */
static int32_t	corrected_max_fit(uint64_t budget_bytes,
				uint64_t bytes_per_layer, uint64_t output_role_bytes,
				int32_t n_layer_all)
{
	uint64_t	quotient;
	uint64_t	v;

	if (budget_bytes < output_role_bytes || bytes_per_layer == 0)
		return (0);
	quotient = (budget_bytes - output_role_bytes) / bytes_per_layer;
	v = sat_add_u64(quotient, 1);
	if (v > (uint64_t)n_layer_all)
		v = (uint64_t)n_layer_all;
	return ((int32_t)v);
}

static uint64_t	kv_bytes_for(const membrane_joint_plan_request_t *req,
					int precision)
{
	if (precision == MEMBRANE_JOINT_KV_Q8)
		return (req->kv_bytes_q8);
	if (precision == MEMBRANE_JOINT_KV_Q5)
		return (req->kv_bytes_q5);
	return (req->kv_bytes_native);
}

/* How much of the WEIGHT budget this precision's KV bytes should be
 * assumed to consume, reusing kv_residency_policy.h's own preflight
 * function unchanged (Section 10: centralize fit calculation, never
 * double-apply reserve) -- 0 for cpu/auto placement (KV never/not-
 * guaranteed GPU-resident, so nothing to reserve weight-budget room
 * for), the full amount for default/gpu (KV follows weights onto the
 * GPU, or is explicitly required to). */
static uint64_t	kv_preflight_bytes(const membrane_joint_plan_request_t *req,
					int precision)
{
	return (membrane_kv_policy_preflight_reservation(req->kv_placement_mode,
			kv_bytes_for(req, precision)));
}

static uint64_t	weight_budget_for(uint64_t budget_bytes,
					uint64_t kv_preflight)
{
	return (budget_bytes > kv_preflight ? budget_bytes - kv_preflight : 0);
}

static void	check_compat(const membrane_joint_plan_request_t *req,
				int precision, int *out_ok, char *out_reason,
				size_t out_reason_size)
{
	membrane_compat_result_t	cr;

	if (precision == MEMBRANE_JOINT_KV_NATIVE)
	{
		*out_ok = 1;
		out_reason[0] = '\0';
		return ;
	}
	*out_ok = membrane_check_kv_compat(req->arch_name, req->n_embd,
			req->n_head, req->n_head_kv, req->ctx_size, precision, &cr);
	snprintf(out_reason, out_reason_size, "%s", cr.reason);
}

/* Resolves this candidate's KV placement via the SAME pure function
 * main.cpp's own (unchanged, still-authoritative, still post-load)
 * resolve_kv_placement() uses -- see joint_planner.h's own top comment
 * for why this is a SECOND, pre-load-estimate-based call for ranking
 * purposes, not a replacement for the authoritative one. */
static void	resolve_candidate_placement(const membrane_joint_plan_request_t *req,
				membrane_joint_candidate_t *c, uint64_t kv_bytes)
{
	membrane_kv_residency_result_t	kr;
	uint64_t	kv_bytes_per_layer;
	int			ok;

	c->kv_placement = MEMBRANE_JOINT_PLACEMENT_DEFAULT;
	c->estimated_kv_gpu_bytes = 0;
	c->estimated_host_kv_bytes = 0;
	c->fits_host = 1;
	if (req->kv_placement_mode == MEMBRANE_JOINT_PLACEMENT_DEFAULT
		|| c->gpu_layers <= 0)
	{
		/* DEFAULT: preserves the exact pre-Phase-20 contract (no
		 * kv_dev_override, llama.cpp's own upstream default decides) --
		 * never modeled as a placement candidate, matching main.cpp's
		 * own resolve_kv_placement() no-op for DEFAULT. Same trivial
		 * treatment when there are no GPU weight layers at all: KV has
		 * nowhere to be GPU-resident regardless of placement_mode. */
		if (c->gpu_layers <= 0)
			c->estimated_host_kv_bytes = kv_bytes;
		else
			c->estimated_kv_gpu_bytes = kv_bytes;
		return ;
	}
	kv_bytes_per_layer = (uint64_t)req->n_layer_all > 0
		? kv_bytes / (uint64_t)req->n_layer_all : 0;
	ok = membrane_kv_residency_resolve(req->kv_placement_mode,
			req->n_layer_all, kv_bytes_per_layer, req->device_free_bytes,
			req->device_total_bytes, c->estimated_weight_gpu_bytes, 0, &kr);
	c->fits_host = ok;
	if (!ok)
		return ;
	c->kv_placement = req->kv_placement_mode;
	c->estimated_kv_gpu_bytes = kr.gpu_kv_bytes;
	c->estimated_host_kv_bytes = kr.cpu_kv_bytes;
}

/* Fills every field of one candidate except reason_code/eligible
 * (left to the caller, which knows the broader context -- e.g.
 * whether this candidate lost to a sibling for a reason unrelated to
 * its own fit). gpu_layers == 0 is always a legal candidate shape
 * (weights entirely CPU-resident); reserve/budget arithmetic is
 * skipped for it (there is nothing to reserve GPU room for). */
static void	build_candidate(const membrane_joint_plan_request_t *req,
				int precision, int32_t gpu_layers, uint64_t reserve_bytes,
				uint64_t budget_bytes, membrane_joint_candidate_t *c)
{
	uint64_t	kv_bytes = kv_bytes_for(req, precision);
	uint64_t	weight_bytes = membrane_joint_estimate_gpu_weight_bytes(
			gpu_layers, req->bytes_per_layer, req->output_role_bytes);

	memset(c, 0, sizeof(*c));
	c->gpu_layers = gpu_layers;
	c->kv_precision = precision;
	c->estimated_weight_gpu_bytes = weight_bytes;
	c->available_gpu_bytes = req->device_free_bytes;
	c->safety_reserve_bytes = reserve_bytes;
	if (gpu_layers <= 0)
		c->fits_gpu = 1;
	else if (req->output_role_bytes == 0)
	{
		/* gpu_device.h's own contract: output_role_bytes == 0 means the
		 * output-role tensor(s) could not be identified (nonstandard/
		 * unrecognized tensor naming), NOT "the real output-role
		 * footprint happens to be zero bytes" -- llama.cpp always
		 * offloads *something* for the output role once gpu_layers >=
		 * 1 (see this file's own top comment). Treating an unavailable
		 * estimate as a real zero would let a 1-layer candidate pass
		 * the budget check while silently missing that allocation --
		 * exactly the kind of under-count Phase 20 exists to fix, not
		 * reintroduce. Fail conservatively instead of guessing. */
		c->fits_gpu = 0;
	}
	else
		c->fits_gpu = (weight_bytes + kv_preflight_bytes(req, precision)
			<= budget_bytes);
	resolve_candidate_placement(req, c, kv_bytes);
	c->estimated_total_gpu_bytes = c->estimated_weight_gpu_bytes
		+ c->estimated_kv_gpu_bytes;
}

static void	finalize_ineligible(membrane_joint_candidate_t *c,
				int compatible, const char *reason)
{
	c->compatible = compatible;
	c->eligible = 0;
	snprintf(c->reason_code, sizeof(c->reason_code), "%s", reason);
}

/* Single-precision path (explicit --kv native/q8/q5, any --gpu-layers
 * request). Ranking within this path is trivial (Section 12's
 * native-vs-q8-vs-q5 tradeoff never arises here -- precision is
 * already fixed): prefer more GPU-resident weight layers among
 * eligible candidates, matching this module's documented policy step
 * 5. */
static void	resolve_single_precision(const membrane_joint_plan_request_t *req,
				uint64_t reserve_bytes, uint64_t budget_bytes,
				membrane_joint_plan_result_t *out)
{
	int			compat_ok;
	char		compat_reason[256];
	int			n = 0;

	check_compat(req, req->precision_request, &compat_ok, compat_reason,
		sizeof(compat_reason));
	if (!compat_ok)
	{
		build_candidate(req, req->precision_request, 0, reserve_bytes,
			budget_bytes, &out->candidates[n]);
		finalize_ineligible(&out->candidates[n],
			0, MEMBRANE_JOINT_REASON_INCOMPATIBLE_ARCH);
		n++;
		out->candidate_count = n;
		return ;
	}
	if (req->gpu_layers_request >= 0
		|| req->gpu_layers_request == MEMBRANE_JOINT_GPU_LAYERS_REQUEST_ALL)
	{
		int32_t	v = req->gpu_layers_request
				== MEMBRANE_JOINT_GPU_LAYERS_REQUEST_ALL
			? req->n_layer_all : req->gpu_layers_request;

		if (v > req->n_layer_all)
			v = req->n_layer_all;
		build_candidate(req, req->precision_request, v, reserve_bytes,
			budget_bytes, &out->candidates[n]);
		out->candidates[n].compatible = 1;
		out->candidates[n].eligible = out->candidates[n].fits_gpu
			&& out->candidates[n].fits_host;
		snprintf(out->candidates[n].reason_code,
			sizeof(out->candidates[n].reason_code), "%s",
			out->candidates[n].eligible
				? MEMBRANE_JOINT_REASON_SELECTED
				: MEMBRANE_JOINT_REASON_GPU_MEMORY_INSUFFICIENT);
		n++;
	}
	else
	{
		int32_t	v_max = corrected_max_fit(
				weight_budget_for(budget_bytes,
					kv_preflight_bytes(req, req->precision_request)),
				req->bytes_per_layer, req->output_role_bytes,
				req->n_layer_all);

		if (v_max > 0)
		{
			build_candidate(req, req->precision_request, v_max,
				reserve_bytes, budget_bytes, &out->candidates[n]);
			out->candidates[n].compatible = 1;
			out->candidates[n].eligible = out->candidates[n].fits_gpu
				&& out->candidates[n].fits_host;
			snprintf(out->candidates[n].reason_code,
				sizeof(out->candidates[n].reason_code), "%s",
				out->candidates[n].eligible
					? MEMBRANE_JOINT_REASON_SELECTED
					: MEMBRANE_JOINT_REASON_GPU_MEMORY_INSUFFICIENT);
			n++;
		}
		build_candidate(req, req->precision_request, 0, reserve_bytes,
			budget_bytes, &out->candidates[n]);
		out->candidates[n].compatible = 1;
		out->candidates[n].eligible = out->candidates[n].fits_gpu
			&& out->candidates[n].fits_host;
		snprintf(out->candidates[n].reason_code,
			sizeof(out->candidates[n].reason_code), "%s",
			out->candidates[n].eligible ? MEMBRANE_JOINT_REASON_SELECTED
				: MEMBRANE_JOINT_REASON_GPU_MEMORY_INSUFFICIENT);
		n++;
	}
	out->candidate_count = n;
}

/* Adaptive path (--kv adaptive / bare --auto's own default precision).
 * Delegates the Q8-vs-Q5 decision to adaptive_kv_policy.h's own,
 * already-shipped, already-tested membrane_adaptive_kv_resolve() --
 * this module does NOT reimplement or second-guess its full-residency-
 * aware ordering (Section 19: reuse, no redundant parallel logic).
 * The only Phase 20 change here is what feeds it: each candidate's own
 * max-fit layer count now comes from the CORRECTED weight formula
 * instead of the pre-Phase-20 linear one. */
static void	resolve_adaptive_precision(const membrane_joint_plan_request_t *req,
				uint64_t reserve_bytes, uint64_t budget_bytes,
				membrane_joint_plan_result_t *out)
{
	int			compat_q8_ok;
	int			compat_q5_ok;
	char		reason_buf[256];
	membrane_adaptive_kv_candidate_t	cand_q8;
	membrane_adaptive_kv_candidate_t	cand_q5;
	membrane_adaptive_kv_result_t		ar;
	int32_t		v_q8;
	int32_t		v_q5;
	int			explicit_layers;
	int32_t		explicit_v = 0;
	int			n = 0;

	check_compat(req, MEMBRANE_JOINT_KV_Q8, &compat_q8_ok, reason_buf,
		sizeof(reason_buf));
	check_compat(req, MEMBRANE_JOINT_KV_Q5, &compat_q5_ok, reason_buf,
		sizeof(reason_buf));
	if (!compat_q8_ok && !compat_q5_ok)
	{
		build_candidate(req, MEMBRANE_JOINT_KV_Q8, 0, reserve_bytes,
			budget_bytes, &out->candidates[0]);
		finalize_ineligible(&out->candidates[0], 0,
			MEMBRANE_JOINT_REASON_INCOMPATIBLE_ARCH);
		out->candidate_count = 1;
		return ;
	}
	explicit_layers = (req->gpu_layers_request >= 0
		|| req->gpu_layers_request == MEMBRANE_JOINT_GPU_LAYERS_REQUEST_ALL);
	if (explicit_layers)
	{
		explicit_v = req->gpu_layers_request
				== MEMBRANE_JOINT_GPU_LAYERS_REQUEST_ALL
			? req->n_layer_all : req->gpu_layers_request;
		if (explicit_v > req->n_layer_all)
			explicit_v = req->n_layer_all;
		v_q8 = explicit_v;
		v_q5 = explicit_v;
	}
	else
	{
		v_q8 = corrected_max_fit(
				weight_budget_for(budget_bytes,
					kv_preflight_bytes(req, MEMBRANE_JOINT_KV_Q8)),
				req->bytes_per_layer, req->output_role_bytes,
				req->n_layer_all);
		v_q5 = corrected_max_fit(
				weight_budget_for(budget_bytes,
					kv_preflight_bytes(req, MEMBRANE_JOINT_KV_Q5)),
				req->bytes_per_layer, req->output_role_bytes,
				req->n_layer_all);
	}
	memset(&cand_q8, 0, sizeof(cand_q8));
	memset(&cand_q5, 0, sizeof(cand_q5));
	cand_q8.kv_bytes = req->kv_bytes_q8;
	cand_q8.selected_layers = v_q8;
	cand_q8.valid = compat_q8_ok
		&& (explicit_layers
			? membrane_joint_estimate_gpu_weight_bytes(v_q8,
					req->bytes_per_layer, req->output_role_bytes)
				+ kv_preflight_bytes(req, MEMBRANE_JOINT_KV_Q8) <= budget_bytes
			: v_q8 > 0)
		&& (!req->want_kv_budget || req->kv_bytes_q8 <= req->kv_budget_bytes);
	cand_q8.full_residency = cand_q8.valid && v_q8 == req->n_layer_all;
	cand_q5.kv_bytes = req->kv_bytes_q5;
	cand_q5.selected_layers = v_q5;
	cand_q5.valid = compat_q5_ok
		&& (explicit_layers
			? membrane_joint_estimate_gpu_weight_bytes(v_q5,
					req->bytes_per_layer, req->output_role_bytes)
				+ kv_preflight_bytes(req, MEMBRANE_JOINT_KV_Q5) <= budget_bytes
			: v_q5 > 0)
		&& (!req->want_kv_budget || req->kv_bytes_q5 <= req->kv_budget_bytes);
	cand_q5.full_residency = cand_q5.valid && v_q5 == req->n_layer_all;
	out->adaptive_evaluated = 1;
	out->adaptive_q8_kv_bytes = cand_q8.kv_bytes;
	out->adaptive_q8_layers = cand_q8.selected_layers;
	out->adaptive_q8_valid = cand_q8.valid;
	out->adaptive_q5_kv_bytes = cand_q5.kv_bytes;
	out->adaptive_q5_layers = cand_q5.selected_layers;
	out->adaptive_q5_valid = cand_q5.valid;
	membrane_adaptive_kv_resolve(&cand_q8, &cand_q5, 1, &ar);
	if (!ar.ok)
	{
		build_candidate(req, MEMBRANE_JOINT_KV_Q8, 0, reserve_bytes,
			budget_bytes, &out->candidates[0]);
		finalize_ineligible(&out->candidates[0], compat_q8_ok || compat_q5_ok,
			MEMBRANE_JOINT_REASON_GPU_MEMORY_INSUFFICIENT);
		out->candidate_count = 1;
		return ;
	}
	/* CodeRabbit review (PR #30): membrane_adaptive_kv_resolve() ranks
	 * Q8 vs Q5 using GPU-fit alone (cand_q8/cand_q5.valid above) -- it
	 * has no visibility into PLACEMENT feasibility (kv_residency_
	 * resolve()'s own runtime margin can reject a candidate GPU-fit
	 * alone would have accepted). Build and evaluate placement for
	 * BOTH the winning precision's full candidate and the runner-up's,
	 * so a placement-only rejection of the winner doesn't discard a
	 * runner-up that would have been fully eligible -- never
	 * overriding adaptive_kv_policy.h's own precision preference when
	 * the winner IS eligible, only falling back when it is not. */
	{
		int			winner_mode = ar.selected_mode;
		int32_t		winner_layers = ar.selected_layers;
		int			runner_up_mode = (winner_mode == MEMBRANE_JOINT_KV_Q8)
				? MEMBRANE_JOINT_KV_Q5 : MEMBRANE_JOINT_KV_Q8;
		int32_t		runner_up_layers = (runner_up_mode == MEMBRANE_JOINT_KV_Q8)
				? v_q8 : v_q5;
		int			runner_up_gpu_fit_valid = (runner_up_mode
				== MEMBRANE_JOINT_KV_Q8) ? cand_q8.valid : cand_q5.valid;
		membrane_joint_candidate_t	winner_full;
		membrane_joint_candidate_t	runner_up_full;
		int			have_runner_up = 0;

		build_candidate(req, winner_mode, winner_layers, reserve_bytes,
			budget_bytes, &winner_full);
		winner_full.compatible = 1;
		winner_full.eligible = winner_full.fits_gpu && winner_full.fits_host;
		snprintf(winner_full.reason_code, sizeof(winner_full.reason_code),
			"%s", ar.reason);
		if (runner_up_gpu_fit_valid)
		{
			build_candidate(req, runner_up_mode, runner_up_layers,
				reserve_bytes, budget_bytes, &runner_up_full);
			runner_up_full.compatible = 1;
			runner_up_full.eligible = runner_up_full.fits_gpu
				&& runner_up_full.fits_host;
			snprintf(runner_up_full.reason_code,
				sizeof(runner_up_full.reason_code), "%s",
				runner_up_full.eligible
					? MEMBRANE_JOINT_REASON_SELECTED
					: MEMBRANE_JOINT_REASON_GPU_MEMORY_INSUFFICIENT);
			have_runner_up = 1;
		}
		if (winner_full.eligible || !have_runner_up
			|| !runner_up_full.eligible)
		{
			out->candidates[n] = winner_full;
			n++;
			if (have_runner_up)
				out->candidates[n++] = runner_up_full;
		}
		else
		{
			/* Winner's placement failed but the runner-up's did not --
			 * the runner-up becomes THIS plan's primary candidate. */
			out->candidates[n] = runner_up_full;
			n++;
			out->candidates[n++] = winner_full;
		}
	}
	if (!explicit_layers && !(out->candidates[0].eligible))
	{
		/* Section 14: the fallback that lets a constrained-VRAM case
		 * (e.g. Qwen2.5 at capacity) still produce a legal plan within
		 * the SAME precision rather than failing outright, when the
		 * placement step (not the GPU-fit step, which build_candidate
		 * already checked) is what rejected every full-layer candidate
		 * above. Stays tied to adaptive_kv_resolve()'s own preferred
		 * mode (ar.selected_mode), matching this function's existing,
		 * deliberately-bounded scope -- see this function's own top
		 * comment. */
		build_candidate(req, ar.selected_mode, 0, reserve_bytes,
			budget_bytes, &out->candidates[n]);
		out->candidates[n].compatible = 1;
		out->candidates[n].eligible = out->candidates[n].fits_gpu
			&& out->candidates[n].fits_host;
		snprintf(out->candidates[n].reason_code,
			sizeof(out->candidates[n].reason_code), "%s",
			out->candidates[n].eligible ? MEMBRANE_JOINT_REASON_SELECTED
				: MEMBRANE_JOINT_REASON_GPU_MEMORY_INSUFFICIENT);
		n++;
	}
	out->candidate_count = n;
}

int	membrane_joint_plan_resolve(const membrane_joint_plan_request_t *req,
			membrane_joint_plan_result_t *out)
{
	uint64_t	reserve_bytes;
	uint64_t	budget_bytes;
	int			best = -1;
	int			i;

	if (out == NULL)
		return (0);
	memset(out, 0, sizeof(*out));
	out->selected_index = -1;
	if (req == NULL || req->n_layer_all <= 0)
	{
		snprintf(out->reason_code, sizeof(out->reason_code), "%s",
			MEMBRANE_JOINT_REASON_INVALID_CONFIG);
		snprintf(out->reason, sizeof(out->reason), "%s: n_layer_all must "
			"be positive", MEMBRANE_JOINT_REASON_INVALID_CONFIG);
		return (0);
	}
	reserve_bytes = membrane_gpu_reserve_bytes(req->device_total_bytes);
	budget_bytes = req->device_free_bytes > reserve_bytes
		? req->device_free_bytes - reserve_bytes : 0;
	if (req->precision_request == MEMBRANE_JOINT_PRECISION_REQUEST_AUTO)
		resolve_adaptive_precision(req, reserve_bytes, budget_bytes, out);
	else
		resolve_single_precision(req, reserve_bytes, budget_bytes, out);
	/* Deterministic selection: among eligible candidates (generation
	 * order is itself deterministic -- see the two resolve_* helpers
	 * above), prefer more gpu_layers; both resolve_* helpers already
	 * only ever produce candidates of a SINGLE precision family per
	 * call (Section 12's cross-precision tradeoff is adaptive_kv_
	 * policy.h's own, reused, responsibility -- see resolve_adaptive_
	 * precision()'s doc comment), so "more layers" is the entire
	 * ranking rule left to apply here. */
	for (i = 0; i < out->candidate_count; ++i)
	{
		if (!out->candidates[i].eligible)
			continue;
		if (best == -1
			|| out->candidates[i].gpu_layers > out->candidates[best].gpu_layers)
			best = i;
	}
	if (best == -1)
	{
		out->ok = 0;
		out->selected_index = -1;
		snprintf(out->reason_code, sizeof(out->reason_code), "%s",
			MEMBRANE_JOINT_REASON_NO_FEASIBLE_CANDIDATE);
		snprintf(out->reason, sizeof(out->reason), "%s: no candidate "
			"satisfied compatibility, memory fit, and placement together",
			MEMBRANE_JOINT_REASON_NO_FEASIBLE_CANDIDATE);
		return (0);
	}
	out->ok = 1;
	out->selected_index = best;
	snprintf(out->reason_code, sizeof(out->reason_code), "%s",
		out->candidates[best].reason_code);
	snprintf(out->reason, sizeof(out->reason), "%s",
		out->candidates[best].reason_code);
	return (1);
}
