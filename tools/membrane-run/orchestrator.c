#include "orchestrator.h"

#include <stdio.h>
#include <string.h>

/* See orchestrator.h's top comment. Pure. */

static void	copy(char *dst, size_t n, const char *src)
{
	if (n == 0)
		return ;
	snprintf(dst, n, "%s", src != NULL ? src : "");
}

static const char	*g_intents[4] = {"general", "interactive", "memory_saving",
	"long_context"};

const char	*membrane_orch_intent_name(membrane_orch_intent_t i)
{
	if ((int)i < 0 || (int)i > 3)
		return ("general");
	return (g_intents[i]);
}

int	membrane_orch_intent_parse(const char *s, membrane_orch_intent_t *out)
{
	int	i;

	for (i = 0; s != NULL && i < 4; ++i)
		if (strcmp(s, g_intents[i]) == 0)
		{
			*out = (membrane_orch_intent_t)i;
			return (1);
		}
	return (0);
}

const char	*membrane_orch_device_kind_name(membrane_orch_device_kind_t k)
{
	switch (k)
	{
		case MEMBRANE_DEV_KIND_CPU: return ("cpu");
		case MEMBRANE_DEV_KIND_DISCRETE: return ("discrete");
		case MEMBRANE_DEV_KIND_UMA: return ("uma");
		default: return ("unknown");
	}
}

const char	*membrane_orch_evidence_name(membrane_orch_evidence_t e)
{
	switch (e)
	{
		case MEMBRANE_EVIDENCE_INFEASIBLE: return ("infeasible");
		case MEMBRANE_EVIDENCE_UNKNOWN: return ("unknown");
		case MEMBRANE_EVIDENCE_ESTIMATE: return ("estimate");
		case MEMBRANE_EVIDENCE_RESIDENT: return ("resident");
		case MEMBRANE_EVIDENCE_PLANNED: return ("planned");
	}
	return ("unknown");
}

/* ------------------------------------------------------------------ */
/* topology                                                             */
/* ------------------------------------------------------------------ */

size_t	membrane_topology_gpu_count(const membrane_topology_t *t)
{
	size_t	i;
	size_t	n = 0;

	for (i = 0; i < t->device_count; ++i)
		if (t->devices[i].kind == MEMBRANE_DEV_KIND_DISCRETE
			|| t->devices[i].kind == MEMBRANE_DEV_KIND_UMA)
			n++;
	return (n);
}

int	membrane_topology_pick_device(const membrane_topology_t *t,
		uint64_t need_bytes)
{
	size_t	i;
	int		best = -1;

	for (i = 0; i < t->device_count; ++i)
	{
		const membrane_orch_device_t	*d = &t->devices[i];

		if (d->kind != MEMBRANE_DEV_KIND_DISCRETE || !d->free_known
			|| d->free_bytes < need_bytes)
			continue ;
		if (best < 0 || d->free_bytes > t->devices[best].free_bytes)
			best = (int)i;
	}
	return (best);
}

/* ------------------------------------------------------------------ */
/* candidates                                                           */
/* ------------------------------------------------------------------ */

static int	cap_score(membrane_capability_state_t s)
{
	return (s == MEMBRANE_CAPABILITY_SUPPORTED ? 2
		: s == MEMBRANE_CAPABILITY_PARTIAL ? 1 : 0);
}

static int	observation_quality(const membrane_runtime_capabilities_t *c)
{
	int	score = cap_score(c->current_model) + cap_score(c->active_context)
		+ cap_score(c->ram_usage) + cap_score(c->vram_usage)
		+ cap_score(c->kv_cache_usage) + cap_score(c->loaded_model_memory);

	return (score == 0 ? 0 : score <= 6 ? 1 : 2);
}

static void	add_limitation(membrane_orch_candidate_t *c, const char *dim,
				membrane_capability_state_t s)
{
	if (s == MEMBRANE_CAPABILITY_SUPPORTED || c->limitation_count >= 4)
		return ;
	snprintf(c->limitations[c->limitation_count++], MEMBRANE_ORCH_TEXT_MAX,
		"%s control: %s through MEMBRANE", dim,
		membrane_capability_state_name(s));
}

static void	exclude(membrane_orch_candidate_t *c, const char *code,
				const char *text)
{
	c->viable = 0;
	copy(c->exclusion_code, sizeof(c->exclusion_code), code);
	copy(c->exclusion_text, sizeof(c->exclusion_text), text);
}

static void	evaluate_evidence(membrane_orch_candidate_t *c,
				const membrane_orch_runtime_input_t *in,
				const membrane_topology_t *t)
{
	uint64_t	headroom = 0;
	uint64_t	limit = 0;
	size_t		i;

	if (in->plan_valid)
	{
		c->evidence = in->plan_feasible ? MEMBRANE_EVIDENCE_PLANNED
			: MEMBRANE_EVIDENCE_INFEASIBLE;
		copy(c->evidence_text, sizeof(c->evidence_text), in->plan_feasible
			? "Planner v2 found a feasible plan on this host"
			: "Planner v2 found no feasible plan on this host");
		return ;
	}
	if (in->resident_known && in->resident)
	{
		c->evidence = MEMBRANE_EVIDENCE_RESIDENT;
		copy(c->evidence_text, sizeof(c->evidence_text), "the runtime reports "
			"this model as loaded right now");
		return ;
	}
	if (in->size_known && t->ram_known)
	{
		headroom = t->ram_available_bytes;
		limit = t->ram_total_bytes;
		for (i = 0; i < t->device_count; ++i)
			if (t->devices[i].kind == MEMBRANE_DEV_KIND_DISCRETE)
			{
				limit += t->devices[i].total_bytes;
				if (t->devices[i].free_known)
					headroom += t->devices[i].free_bytes;
			}
		if (in->size_bytes > limit)
		{
			c->evidence = MEMBRANE_EVIDENCE_INFEASIBLE;
			copy(c->evidence_text, sizeof(c->evidence_text), "the reported "
				"model size exceeds all memory on this machine");
			return ;
		}
		if (in->size_bytes + in->size_bytes / 10 <= headroom)
		{
			c->evidence = MEMBRANE_EVIDENCE_ESTIMATE;
			copy(c->evidence_text, sizeof(c->evidence_text), "reported size "
				"fits measured free memory (weights only, no KV accounting)");
			return ;
		}
	}
	c->evidence = MEMBRANE_EVIDENCE_UNKNOWN;
	copy(c->evidence_text, sizeof(c->evidence_text), "nothing proves this "
		"model fits (no plan, not resident, no usable size/memory figures)");
}

static void	build_candidate(membrane_orch_candidate_t *c,
				const membrane_orch_runtime_input_t *in,
				const membrane_topology_t *t)
{
	const membrane_runtime_capabilities_t	*k = &in->caps;

	memset(c, 0, sizeof(*c));
	copy(c->runtime_id, sizeof(c->runtime_id), in->runtime_id);
	copy(c->display_name, sizeof(c->display_name), in->display_name);
	copy(c->model_id, sizeof(c->model_id), in->model_id);
	c->viable = 1;
	c->gpu_layers = -1;
	c->resident = in->resident_known && in->resident;
	c->control_count = (k->context_control == MEMBRANE_CAPABILITY_SUPPORTED)
		+ (k->gpu_layer_control == MEMBRANE_CAPABILITY_SUPPORTED)
		+ (k->kv_precision_control == MEMBRANE_CAPABILITY_SUPPORTED)
		+ (k->kv_placement_control == MEMBRANE_CAPABILITY_SUPPORTED);
	c->observation_quality = observation_quality(k);
	add_limitation(c, "context", k->context_control);
	add_limitation(c, "GPU-layer", k->gpu_layer_control);
	add_limitation(c, "KV-precision", k->kv_precision_control);
	add_limitation(c, "KV-placement", k->kv_placement_control);
	c->context_max_known = in->context_max_known;
	c->context_max = in->context_max;
	if (in->plan_valid)
	{
		c->context_known = in->plan_context > 0;
		c->context = in->plan_context;
		c->gpu_layers = in->plan_gpu_layers;
		copy(c->kv_precision, sizeof(c->kv_precision), in->plan_kv_precision);
		copy(c->kv_placement, sizeof(c->kv_placement), in->plan_kv_placement);
	}
	if (in->plan_valid && in->plan_mem_known)
	{
		c->mem_known = 1;
		c->mem_bytes = in->plan_host_required_bytes;
		copy(c->mem_provenance, sizeof(c->mem_provenance), "planner_v2");
	}
	else if (in->size_known)
	{
		c->mem_known = 1;
		c->mem_bytes = in->size_bytes;
		copy(c->mem_provenance, sizeof(c->mem_provenance),
			"runtime_reported");
	}
	if (in->availability != MEMBRANE_RUNTIME_AVAILABILITY_AVAILABLE)
	{
		exclude(c, MEMBRANE_ORCH_REASON_UNAVAILABLE,
			in->unavailable_reason[0] ? in->unavailable_reason
				: "the runtime is not available");
		return ;
	}
	if (!in->model_present)
	{
		exclude(c, MEMBRANE_ORCH_REASON_NO_MODEL, "this runtime does not "
			"have the requested model");
		return ;
	}
	evaluate_evidence(c, in, t);
	if (c->evidence == MEMBRANE_EVIDENCE_INFEASIBLE)
		exclude(c, MEMBRANE_ORCH_REASON_INFEASIBLE, c->evidence_text);
}

/* ------------------------------------------------------------------ */
/* ranking                                                              */
/* ------------------------------------------------------------------ */

typedef enum e_key
{
	KEY_NONE = 0,
	KEY_EVIDENCE,
	KEY_RESIDENT,
	KEY_OBSERVATION,
	KEY_MEMORY,
	KEY_CONTROL,
	KEY_CONTEXT,
	KEY_ID
}	key_t_;

static const key_t_	g_chain[4][6] = {
	{KEY_EVIDENCE, KEY_RESIDENT, KEY_OBSERVATION, KEY_ID, KEY_NONE, KEY_NONE},
	{KEY_RESIDENT, KEY_EVIDENCE, KEY_OBSERVATION, KEY_ID, KEY_NONE, KEY_NONE},
	{KEY_MEMORY, KEY_CONTROL, KEY_EVIDENCE, KEY_ID, KEY_NONE, KEY_NONE},
	{KEY_CONTEXT, KEY_EVIDENCE, KEY_RESIDENT, KEY_ID, KEY_NONE, KEY_NONE}};

/* <0: a better, >0: b better, 0: equal on this key */
static int	cmp_key(key_t_ k, const membrane_orch_candidate_t *a,
				const membrane_orch_candidate_t *b)
{
	switch (k)
	{
		case KEY_EVIDENCE:
			return ((int)b->evidence - (int)a->evidence);
		case KEY_RESIDENT:
			return (b->resident - a->resident);
		case KEY_OBSERVATION:
			return (b->observation_quality - a->observation_quality);
		case KEY_CONTROL:
			return (b->control_count - a->control_count);
		case KEY_MEMORY:
			if (a->mem_known != b->mem_known)
				return (b->mem_known - a->mem_known);
			if (!a->mem_known)
				return (0);
			return (a->mem_bytes < b->mem_bytes ? -1
				: a->mem_bytes > b->mem_bytes ? 1 : 0);
		case KEY_CONTEXT:
			if (a->context_max_known != b->context_max_known)
				return (b->context_max_known - a->context_max_known);
			if (!a->context_max_known)
				return (0);
			return (a->context_max > b->context_max ? -1
				: a->context_max < b->context_max ? 1 : 0);
		case KEY_ID:
			return (strcmp(a->runtime_id, b->runtime_id));
		default:
			return (0);
	}
}

/* returns the deciding key through *deciding (KEY_NONE when equal) */
static int	compare(membrane_orch_intent_t intent,
				const membrane_orch_candidate_t *a,
				const membrane_orch_candidate_t *b, key_t_ *deciding)
{
	int	i;

	for (i = 0; i < 6 && g_chain[intent][i] != KEY_NONE; ++i)
	{
		int	c = cmp_key(g_chain[intent][i], a, b);

		if (c != 0)
		{
			if (deciding != NULL)
				*deciding = g_chain[intent][i];
			return (c);
		}
	}
	if (deciding != NULL)
		*deciding = KEY_NONE;
	return (0);
}

static void	add_reason(membrane_orch_decision_t *d, const char *code)
{
	size_t	i;

	for (i = 0; i < d->reason_count; ++i)
		if (strcmp(d->reasons[i], code) == 0)
			return ;
	if (d->reason_count < MEMBRANE_ORCH_MAX_REASONS)
		copy(d->reasons[d->reason_count++], sizeof(d->reasons[0]), code);
}

void	membrane_orchestrate(membrane_orch_intent_t intent,
			const membrane_topology_t *topology,
			const membrane_orch_runtime_input_t *inputs, size_t n,
			membrane_orch_decision_t *out)
{
	size_t	i;
	size_t	j;

	memset(out, 0, sizeof(*out));
	out->schema_version = MEMBRANE_ORCH_SCHEMA_VERSION;
	out->intent = intent;
	copy(out->confidence, sizeof(out->confidence), "low");
	if (n > MEMBRANE_ORCH_MAX_RUNTIMES)
		n = MEMBRANE_ORCH_MAX_RUNTIMES;
	for (i = 0; i < n; ++i)
		build_candidate(&out->candidates[i], &inputs[i], topology);
	out->candidate_count = n;
	for (i = 0; i < n; ++i)
		if (out->candidates[i].viable)
			out->order[out->order_count++] = (int)i;
	/* stable insertion sort by the intent's chain (ends in runtime id, so
	 * the order is total and input-order independent) */
	for (i = 1; i < out->order_count; ++i)
		for (j = i; j > 0; --j)
		{
			const membrane_orch_candidate_t	*x
				= &out->candidates[out->order[j - 1]];
			const membrane_orch_candidate_t	*y = &out->candidates[out->order[j]];

			if (compare(intent, x, y, NULL) <= 0)
				break ;
			{
				int	t = out->order[j];

				out->order[j] = out->order[j - 1];
				out->order[j - 1] = t;
			}
		}
	if (out->order_count == 0)
	{
		add_reason(out, MEMBRANE_ORCH_REASON_NO_CANDIDATE);
		for (i = 0; i < n; ++i)
			if (out->candidates[i].exclusion_code[0])
				add_reason(out, out->candidates[i].exclusion_code);
		snprintf(out->summary, sizeof(out->summary), "No runtime can serve "
			"this model right now (%zu considered).", n);
		return ;
	}
	out->has_selection = 1;
	out->selected = out->order[0];
	{
		const membrane_orch_candidate_t	*s = &out->candidates[out->selected];
		key_t_							deciding = KEY_NONE;

		if (out->order_count == 1)
			add_reason(out, MEMBRANE_ORCH_REASON_ONLY_CANDIDATE);
		else
			compare(intent, s, &out->candidates[out->order[1]], &deciding);
		if (s->evidence == MEMBRANE_EVIDENCE_PLANNED)
			add_reason(out, MEMBRANE_ORCH_REASON_PLANNED);
		if (s->resident)
			add_reason(out, MEMBRANE_ORCH_REASON_RESIDENT);
		if (out->order_count > 1)
			switch (deciding)
			{
				case KEY_EVIDENCE:
					add_reason(out, MEMBRANE_ORCH_REASON_STRONGER_EVIDENCE);
					break ;
				case KEY_RESIDENT:
					add_reason(out, MEMBRANE_ORCH_REASON_RESIDENT);
					break ;
				case KEY_OBSERVATION:
					add_reason(out, MEMBRANE_ORCH_REASON_OBSERVATION);
					break ;
				case KEY_MEMORY:
					add_reason(out, MEMBRANE_ORCH_REASON_SMALLEST_MEMORY);
					break ;
				case KEY_CONTROL:
					add_reason(out, MEMBRANE_ORCH_REASON_MORE_CONTROL);
					break ;
				case KEY_CONTEXT:
					add_reason(out, MEMBRANE_ORCH_REASON_LARGEST_CONTEXT);
					break ;
				default:
					add_reason(out, MEMBRANE_ORCH_REASON_TIEBREAK);
					break ;
			}
		for (i = 0; i < n; ++i)
		{
			if (out->candidates[i].viable
				&& out->candidates[i].evidence == MEMBRANE_EVIDENCE_UNKNOWN)
				add_reason(out, MEMBRANE_ORCH_REASON_UNKNOWN_FIT);
		}
		/* confidence: how strong is the evidence, and was it a real
		 * difference or a coin-flip-by-name */
		if (out->order_count > 1 && deciding == KEY_ID)
		{
			copy(out->confidence, sizeof(out->confidence), "low");
			copy(out->confidence_text, sizeof(out->confidence_text),
				"candidates are indistinguishable on the available facts; "
				"the choice fell back to a stable runtime-id order");
		}
		else if (s->evidence >= MEMBRANE_EVIDENCE_RESIDENT)
		{
			copy(out->confidence, sizeof(out->confidence), "high");
			copy(out->confidence_text, sizeof(out->confidence_text),
				s->evidence == MEMBRANE_EVIDENCE_PLANNED
				? "backed by a Planner v2 plan for this host"
				: "the runtime reports the model loaded");
		}
		else if (s->evidence == MEMBRANE_EVIDENCE_ESTIMATE)
		{
			copy(out->confidence, sizeof(out->confidence), "medium");
			copy(out->confidence_text, sizeof(out->confidence_text),
				"based on a reported model size versus measured free memory "
				"only; no KV accounting");
		}
		else
		{
			copy(out->confidence, sizeof(out->confidence), "low");
			copy(out->confidence_text, sizeof(out->confidence_text),
				"it is not known whether the model fits");
		}
		snprintf(out->summary, sizeof(out->summary), "%s (%s) is the best fit "
			"for '%.60s' under intent '%s': %.100s.", s->display_name,
			s->runtime_id, s->model_id, membrane_orch_intent_name(intent),
			s->evidence_text);
	}
}
