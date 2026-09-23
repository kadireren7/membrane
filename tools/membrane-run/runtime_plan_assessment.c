#include <stdio.h>
#include <string.h>
#include <strings.h>

#include "kv_residency_policy.h"
#include "runtime_plan_assessment.h"

/* See runtime_plan_assessment.h's own top comment for the full contract. */

const char	*membrane_planning_level_name(membrane_planning_level_t l)
{
	if (l == MEMBRANE_PLANNING_LEVEL_CAPABILITY_ONLY)
		return ("capability_only");
	if (l == MEMBRANE_PLANNING_LEVEL_PLANNER_ESTIMATE)
		return ("planner_estimate");
	if (l == MEMBRANE_PLANNING_LEVEL_PLANNER_EXACT)
		return ("planner_exact");
	return ("unavailable");
}

const char	*membrane_runtime_actionability_name(
				membrane_runtime_actionability_t a)
{
	if (a == MEMBRANE_ACTIONABILITY_ADVISORY_ONLY)
		return ("advisory_only");
	if (a == MEMBRANE_ACTIONABILITY_PARTIALLY_ACTIONABLE)
		return ("partially_actionable");
	if (a == MEMBRANE_ACTIONABILITY_FULLY_ACTIONABLE)
		return ("fully_actionable");
	return ("unsupported");
}

const char	*membrane_dimension_applicability_name(
				membrane_dimension_applicability_t a)
{
	if (a == MEMBRANE_APPLICABILITY_UNSUPPORTED)
		return ("unsupported");
	if (a == MEMBRANE_APPLICABILITY_OBSERVABLE_ONLY)
		return ("observable_only");
	if (a == MEMBRANE_APPLICABILITY_PARTIALLY_CONTROLLABLE)
		return ("partially_controllable");
	if (a == MEMBRANE_APPLICABILITY_CONTROLLABLE)
		return ("controllable");
	return ("unknown");
}

const char	*membrane_assessment_value_source_name(
				membrane_assessment_value_source_t s)
{
	if (s == MEMBRANE_ASSESS_VALUE_PLANNER)
		return ("planner_v2");
	if (s == MEMBRANE_ASSESS_VALUE_EXPLICIT_REQUEST)
		return ("explicit_request");
	if (s == MEMBRANE_ASSESS_VALUE_RUNTIME_METADATA)
		return ("runtime_metadata");
	return ("none");
}

const char	*membrane_assessment_dimension_name(
				membrane_assessment_dimension_t d)
{
	static const char	*names[MEMBRANE_ASSESS_DIM_COUNT] = {
		"quant_variant", "context", "gpu_layers", "kv_precision",
		"kv_placement", "device_selection", "concurrency"};

	if ((int)d < 0 || d >= MEMBRANE_ASSESS_DIM_COUNT)
		return ("unknown");
	return (names[d]);
}

/* H1's negotiation dimension names, reused as reason-code prefixes. */
static const char	*dimension_reason_prefix(membrane_assessment_dimension_t d)
{
	if (d == MEMBRANE_ASSESS_DIM_QUANT_VARIANT)
		return (MEMBRANE_NEGOTIATION_DIM_QUANT_VARIANT_CONTROL);
	if (d == MEMBRANE_ASSESS_DIM_CONTEXT)
		return (MEMBRANE_NEGOTIATION_DIM_CONTEXT_CONTROL);
	if (d == MEMBRANE_ASSESS_DIM_GPU_LAYERS)
		return (MEMBRANE_NEGOTIATION_DIM_GPU_LAYERS_CONTROL);
	if (d == MEMBRANE_ASSESS_DIM_KV_PRECISION)
		return (MEMBRANE_NEGOTIATION_DIM_KV_PRECISION_CONTROL);
	if (d == MEMBRANE_ASSESS_DIM_KV_PLACEMENT)
		return (MEMBRANE_NEGOTIATION_DIM_KV_PLACEMENT_CONTROL);
	if (d == MEMBRANE_ASSESS_DIM_DEVICE_SELECTION)
		return ("DEVICE_SELECTION_CONTROL");
	return ("CONCURRENCY_CONTROL");
}

static void	copy_str(char *dst, size_t dst_size, const char *src)
{
	size_t	len;

	if (dst_size == 0)
		return ;
	if (src == NULL)
		src = "";
	len = strlen(src);
	if (len >= dst_size)
		len = dst_size - 1;
	memcpy(dst, src, len);
	dst[len] = '\0';
}

static void	push_reason(membrane_runtime_plan_assessment_t *out,
				const char *code, const char *detail)
{
	if (out->reason_count >= MEMBRANE_ASSESS_MAX_REASONS)
		return ;
	copy_str(out->reasons[out->reason_count].code,
		sizeof(out->reasons[0].code), code);
	copy_str(out->reasons[out->reason_count].detail,
		sizeof(out->reasons[0].detail), detail);
	out->reason_count++;
}

static int	is_observable(membrane_capability_state_t s)
{
	return (s == MEMBRANE_CAPABILITY_SUPPORTED
		|| s == MEMBRANE_CAPABILITY_PARTIAL);
}

/* The control capability, plus the ONLY observability capability that
 * genuinely reports the same dimension's value (quant: the runtime's
 * model metadata; context: its active-context telemetry). GPU layers,
 * KV precision/placement, device and concurrency have no matching
 * observability field in runtime_capabilities.h -- VRAM usage is not a
 * layer count, KV-cache usage is not a precision -- so they can never be
 * OBSERVABLE_ONLY. */
static void	dimension_capabilities(const membrane_runtime_capabilities_t *c,
				membrane_assessment_dimension_t d,
				membrane_capability_state_t *control,
				membrane_capability_state_t *observe)
{
	*observe = MEMBRANE_CAPABILITY_UNSUPPORTED;
	if (d == MEMBRANE_ASSESS_DIM_QUANT_VARIANT)
	{
		*control = c->quant_variant_control;
		*observe = c->model_metadata;
	}
	else if (d == MEMBRANE_ASSESS_DIM_CONTEXT)
	{
		*control = c->context_control;
		*observe = c->active_context;
	}
	else if (d == MEMBRANE_ASSESS_DIM_GPU_LAYERS)
		*control = c->gpu_layer_control;
	else if (d == MEMBRANE_ASSESS_DIM_KV_PRECISION)
		*control = c->kv_precision_control;
	else if (d == MEMBRANE_ASSESS_DIM_KV_PLACEMENT)
		*control = c->kv_placement_control;
	else if (d == MEMBRANE_ASSESS_DIM_DEVICE_SELECTION)
		*control = c->device_selection;
	else
		*control = c->concurrency_control;
}

static membrane_dimension_applicability_t	classify(
				membrane_capability_state_t control,
				membrane_capability_state_t observe)
{
	if (control == MEMBRANE_CAPABILITY_SUPPORTED)
		return (MEMBRANE_APPLICABILITY_CONTROLLABLE);
	if (control == MEMBRANE_CAPABILITY_PARTIAL)
		return (MEMBRANE_APPLICABILITY_PARTIALLY_CONTROLLABLE);
	if (control == MEMBRANE_CAPABILITY_UNSUPPORTED)
		return (is_observable(observe)
			? MEMBRANE_APPLICABILITY_OBSERVABLE_ONLY
			: MEMBRANE_APPLICABILITY_UNSUPPORTED);
	return (MEMBRANE_APPLICABILITY_UNKNOWN);
}

static const char	*kv_precision_label(int p)
{
	if (p == MEMBRANE_JOINT_KV_NATIVE)
		return ("native");
	if (p == MEMBRANE_JOINT_KV_Q8)
		return ("q8");
	if (p == MEMBRANE_JOINT_KV_Q5)
		return ("q5");
	return ("unknown");
}

static void	set_value(membrane_assessment_dimension_result_t *r,
				const char *value, membrane_assessment_value_source_t src,
				membrane_plan_source_t plan_src)
{
	r->value_known = 1;
	copy_str(r->value, sizeof(r->value), value);
	r->value_source = src;
	r->plan_source = plan_src;
}

static void	set_u64_value(membrane_assessment_dimension_result_t *r,
				uint64_t v, membrane_assessment_value_source_t src,
				membrane_plan_source_t plan_src)
{
	char	buf[32];

	snprintf(buf, sizeof(buf), "%llu", (unsigned long long)v);
	set_value(r, buf, src, plan_src);
}

static void	set_gpu_layers_value(membrane_assessment_dimension_result_t *r,
				int32_t v, membrane_assessment_value_source_t src,
				membrane_plan_source_t plan_src)
{
	char	buf[32];

	if (v == MEMBRANE_JOINT_GPU_LAYERS_REQUEST_ALL)
		copy_str(buf, sizeof(buf), "all");
	else
		snprintf(buf, sizeof(buf), "%d", v);
	set_value(r, buf, src, plan_src);
}

/* Required dimensions + values from a Planner v2 plan (same required-
 * dimension rule as membrane_runtime_negotiate_plan()). */
static void	fill_from_plan(const membrane_plan_t *p,
				membrane_assessment_dimension_result_t *dims)
{
	membrane_assessment_dimension_result_t	*r;

	if (p->identity.variant_known)
	{
		r = &dims[MEMBRANE_ASSESS_DIM_QUANT_VARIANT];
		r->required = 1;
		set_value(r, p->identity.variant, MEMBRANE_ASSESS_VALUE_PLANNER,
			p->identity.variant_source);
	}
	if (!p->decisions.has_decisions)
		return ;
	r = &dims[MEMBRANE_ASSESS_DIM_CONTEXT];
	r->required = 1;
	set_u64_value(r, p->decisions.context, MEMBRANE_ASSESS_VALUE_PLANNER,
		p->decisions.context_source);
	r = &dims[MEMBRANE_ASSESS_DIM_GPU_LAYERS];
	r->required = 1;
	set_gpu_layers_value(r, p->decisions.gpu_layers,
		MEMBRANE_ASSESS_VALUE_PLANNER, p->decisions.gpu_layers_source);
	r = &dims[MEMBRANE_ASSESS_DIM_KV_PRECISION];
	r->required = 1;
	set_value(r, kv_precision_label(p->decisions.kv_precision),
		MEMBRANE_ASSESS_VALUE_PLANNER, p->decisions.kv_precision_source);
	r = &dims[MEMBRANE_ASSESS_DIM_KV_PLACEMENT];
	r->required = 1;
	set_value(r, membrane_kv_placement_mode_name(p->decisions.kv_placement),
		MEMBRANE_ASSESS_VALUE_PLANNER, p->decisions.kv_placement_source);
}

/* Required dimensions + values when no plan exists: only explicit
 * requests become required; the runtime's own reported quant is shown as
 * a fact, never as a recommendation. */
static void	fill_from_request(const membrane_runtime_recommend_input_t *in,
				membrane_runtime_plan_assessment_t *out)
{
	membrane_assessment_dimension_result_t	*dims = out->dimensions;
	membrane_assessment_dimension_result_t	*r;
	int		has_model_quant;
	char	detail[MEMBRANE_ASSESS_DETAIL_MAX];

	has_model_quant = in->model_quantization != NULL
		&& in->model_quantization[0] != '\0';
	r = &dims[MEMBRANE_ASSESS_DIM_QUANT_VARIANT];
	if (in->requested_quant != NULL && in->requested_quant[0] != '\0')
	{
		r->required = 1;
		set_value(r, in->requested_quant,
			MEMBRANE_ASSESS_VALUE_EXPLICIT_REQUEST, MEMBRANE_PLAN_SOURCE_UNKNOWN);
		if (has_model_quant
			&& strcasecmp(in->requested_quant, in->model_quantization) != 0)
		{
			snprintf(detail, sizeof(detail), "requested quant %s, but this "
				"runtime model is %s; a different quant is a different "
				"runtime model, which MEMBRANE does not obtain",
				in->requested_quant, in->model_quantization);
			push_reason(out, MEMBRANE_ASSESS_REASON_REQUESTED_QUANT_DIFFERS,
				detail);
		}
	}
	else if (has_model_quant)
	{
		set_value(r, in->model_quantization,
			MEMBRANE_ASSESS_VALUE_RUNTIME_METADATA,
			MEMBRANE_PLAN_SOURCE_UNKNOWN);
		snprintf(detail, sizeof(detail), "quantization %s is fixed by the "
			"runtime model itself (reported by the runtime)",
			in->model_quantization);
		push_reason(out, MEMBRANE_ASSESS_REASON_QUANT_FIXED_BY_RUNTIME_MODEL,
			detail);
	}
	if (in->requested_context_known)
	{
		r = &dims[MEMBRANE_ASSESS_DIM_CONTEXT];
		r->required = 1;
		set_u64_value(r, in->requested_context,
			MEMBRANE_ASSESS_VALUE_EXPLICIT_REQUEST,
			MEMBRANE_PLAN_SOURCE_UNKNOWN);
		if (in->model_max_context_known
			&& in->requested_context > in->model_max_context)
		{
			snprintf(detail, sizeof(detail), "requested context %llu "
				"exceeds the model's reported maximum of %llu",
				(unsigned long long)in->requested_context,
				(unsigned long long)in->model_max_context);
			push_reason(out, MEMBRANE_ASSESS_REASON_CONTEXT_EXCEEDS_MODEL_MAX,
				detail);
		}
	}
	if (in->requested_gpu_layers_known)
	{
		r = &dims[MEMBRANE_ASSESS_DIM_GPU_LAYERS];
		r->required = 1;
		set_gpu_layers_value(r, in->requested_gpu_layers,
			MEMBRANE_ASSESS_VALUE_EXPLICIT_REQUEST,
			MEMBRANE_PLAN_SOURCE_UNKNOWN);
	}
	if (in->requested_kv_precision_known)
	{
		r = &dims[MEMBRANE_ASSESS_DIM_KV_PRECISION];
		r->required = 1;
		set_value(r, kv_precision_label(in->requested_kv_precision),
			MEMBRANE_ASSESS_VALUE_EXPLICIT_REQUEST,
			MEMBRANE_PLAN_SOURCE_UNKNOWN);
	}
}

static membrane_planning_level_t	planning_level(
				const membrane_runtime_recommend_input_t *in)
{
	if (in->runtime == NULL
		|| in->runtime->availability != MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE)
		return (MEMBRANE_PLANNING_LEVEL_UNAVAILABLE);
	if (in->plan != NULL)
	{
		if (in->plan->decisions.has_decisions
			&& !in->plan->identity.variant_estimate_only)
			return (MEMBRANE_PLANNING_LEVEL_PLANNER_EXACT);
		return (MEMBRANE_PLANNING_LEVEL_PLANNER_ESTIMATE);
	}
	if (in->model_known)
		return (MEMBRANE_PLANNING_LEVEL_CAPABILITY_ONLY);
	return (MEMBRANE_PLANNING_LEVEL_UNAVAILABLE);
}

static void	push_level_reasons(const membrane_runtime_recommend_input_t *in,
				membrane_runtime_plan_assessment_t *out)
{
	if (out->planning_level == MEMBRANE_PLANNING_LEVEL_UNAVAILABLE)
	{
		if (in->runtime == NULL || in->runtime->availability
			!= MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE)
			push_reason(out, MEMBRANE_ASSESS_REASON_RUNTIME_UNAVAILABLE,
				"the runtime is not available, so nothing can be assessed");
		else
			push_reason(out, MEMBRANE_ASSESS_REASON_MODEL_UNKNOWN,
				"no plan and no runtime model metadata were supplied");
		return ;
	}
	if (out->planning_level == MEMBRANE_PLANNING_LEVEL_CAPABILITY_ONLY)
	{
		push_reason(out, MEMBRANE_ASSESS_REASON_NO_PLANNER_PLAN,
			"no Planner v2 plan exists for this runtime model; only runtime "
			"capabilities and runtime-reported metadata are assessed");
		push_reason(out, MEMBRANE_ASSESS_REASON_EXACT_MEMORY_PLAN_UNAVAILABLE,
			"no exact context, GPU-layer or memory figure is computed: the "
			"inputs Planner v2's memory math needs are not available");
		return ;
	}
	if (out->planning_level == MEMBRANE_PLANNING_LEVEL_PLANNER_ESTIMATE)
		push_reason(out, MEMBRANE_ASSESS_REASON_PLAN_ESTIMATE_ONLY,
			"the Planner v2 plan is a disclosed estimate, not a plan from "
			"real model hparams");
	if (!in->plan->feasibility.feasible
		&& strcmp(in->plan->feasibility.limiting_resource,
			MEMBRANE_PLAN_REASON_MODEL_METADATA_UNAVAILABLE) != 0)
		push_reason(out, MEMBRANE_ASSESS_REASON_PLAN_NOT_FEASIBLE,
			"Planner v2 found no feasible plan on this host");
}

membrane_runtime_actionability_t	membrane_runtime_recommend_plan(
		const membrane_runtime_recommend_input_t *in,
		membrane_runtime_plan_assessment_t *out)
{
	membrane_capability_state_t	control;
	membrane_capability_state_t	observe;
	size_t						i;
	char						code[MEMBRANE_ASSESS_REASON_CODE_MAX];

	memset(out, 0, sizeof(*out));
	out->schema_version = MEMBRANE_RUNTIME_ASSESSMENT_SCHEMA_VERSION;
	if (in == NULL)
	{
		push_reason(out, MEMBRANE_ASSESS_REASON_RUNTIME_UNAVAILABLE,
			"no input");
		return (out->actionability);
	}
	if (in->runtime != NULL)
	{
		copy_str(out->runtime_id, sizeof(out->runtime_id), in->runtime->id);
		out->capability_provenance = in->runtime->capability_provenance;
	}
	if (in->plan != NULL)
		copy_str(out->runtime_model_id, sizeof(out->runtime_model_id),
			in->plan->identity.model_name);
	else if (in->runtime_model_id != NULL)
		copy_str(out->runtime_model_id, sizeof(out->runtime_model_id),
			in->runtime_model_id);
	out->planning_level = planning_level(in);
	push_level_reasons(in, out);
	i = 0;
	while (i < MEMBRANE_ASSESS_DIM_COUNT)
	{
		out->dimensions[i].dimension = (membrane_assessment_dimension_t)i;
		out->dimensions[i].planner_dimension
			= i < MEMBRANE_ASSESS_DIM_DEVICE_SELECTION;
		i++;
	}
	if (out->planning_level != MEMBRANE_PLANNING_LEVEL_UNAVAILABLE)
	{
		if (in->plan != NULL)
			fill_from_plan(in->plan, out->dimensions);
		else
			fill_from_request(in, out);
	}
	i = 0;
	while (i < MEMBRANE_ASSESS_DIM_COUNT)
	{
		membrane_assessment_dimension_result_t	*r = &out->dimensions[i];

		control = MEMBRANE_CAPABILITY_UNKNOWN;
		observe = MEMBRANE_CAPABILITY_UNKNOWN;
		if (in->runtime != NULL)
			dimension_capabilities(&in->runtime->capabilities, r->dimension,
				&control, &observe);
		r->capability = control;
		r->applicability = classify(control, observe);
		snprintf(code, sizeof(code), "%s_%s",
			dimension_reason_prefix(r->dimension),
			control == MEMBRANE_CAPABILITY_SUPPORTED ? "SUPPORTED"
			: control == MEMBRANE_CAPABILITY_PARTIAL ? "PARTIAL"
			: control == MEMBRANE_CAPABILITY_UNSUPPORTED ? "UNSUPPORTED"
			: "UNKNOWN");
		copy_str(r->reason_code, sizeof(r->reason_code), code);
		if (r->required)
		{
			out->required_count++;
			if (r->applicability == MEMBRANE_APPLICABILITY_CONTROLLABLE)
				out->controllable_count++;
		}
		i++;
	}
	if (out->planning_level == MEMBRANE_PLANNING_LEVEL_UNAVAILABLE)
		out->actionability = MEMBRANE_ACTIONABILITY_UNSUPPORTED;
	else if (out->required_count == 0)
	{
		out->actionability = MEMBRANE_ACTIONABILITY_ADVISORY_ONLY;
		push_reason(out, MEMBRANE_ASSESS_REASON_NO_REQUIRED_DIMENSIONS,
			"there is no planned or requested setting for the runtime to "
			"apply; the result is informational");
	}
	else if (out->controllable_count == out->required_count)
		out->actionability = MEMBRANE_ACTIONABILITY_FULLY_ACTIONABLE;
	else if (out->controllable_count > 0)
		out->actionability = MEMBRANE_ACTIONABILITY_PARTIALLY_ACTIONABLE;
	else
		out->actionability = MEMBRANE_ACTIONABILITY_ADVISORY_ONLY;
	return (out->actionability);
}
