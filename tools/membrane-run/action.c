#include "action.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* See action.h's top comment for the contract. Pure except
 * membrane_action_run(), which only talks to the caller's executor. */

static const char	*g_action_names[MEMBRANE_ACTION_CODE_COUNT] = {
	"UNLOAD_MODEL", "SET_CONTEXT", "SET_KV_PRECISION", "SET_KV_PLACEMENT",
	"SET_GPU_LAYERS", "UNPIN_MODEL", "ACTIVATE_MODEL", "PIN_MODEL"
};

const char	*membrane_action_code_name(membrane_action_code_t c)
{
	if ((int)c < 0 || c >= MEMBRANE_ACTION_CODE_COUNT)
		return ("UNKNOWN");
	return (g_action_names[c]);
}

const char	*membrane_action_origin_name(membrane_action_origin_t o)
{
	return (o == MEMBRANE_ACTION_ORIGIN_RECOMMENDATION
		? "recommendation" : "user_request");
}

const char	*membrane_action_risk_name(membrane_action_risk_t r)
{
	if (r == MEMBRANE_ACTION_RISK_HIGH)
		return ("high");
	if (r == MEMBRANE_ACTION_RISK_MEDIUM)
		return ("medium");
	return ("low");
}

const char	*membrane_apply_status_name(membrane_apply_status_t s)
{
	switch (s)
	{
		case MEMBRANE_APPLY_NOTHING_TO_APPLY: return ("nothing_to_apply");
		case MEMBRANE_APPLY_APPLIED: return ("applied");
		case MEMBRANE_APPLY_PARTIALLY_APPLIED: return ("partially_applied");
		case MEMBRANE_APPLY_ROLLED_BACK: return ("rolled_back");
		case MEMBRANE_APPLY_ROLLBACK_FAILED: return ("rollback_failed");
		case MEMBRANE_APPLY_REFUSED: return ("refused");
		case MEMBRANE_APPLY_UNSUPPORTED: return ("unsupported");
		case MEMBRANE_APPLY_STALE_PRECONDITION: return ("stale_precondition");
		case MEMBRANE_APPLY_FAILED: return ("failed");
	}
	return ("unknown");
}

const char	*membrane_action_outcome_name(membrane_action_outcome_t o)
{
	switch (o)
	{
		case MEMBRANE_OUTCOME_NOT_RUN: return ("not_run");
		case MEMBRANE_OUTCOME_APPLIED: return ("applied");
		case MEMBRANE_OUTCOME_FAILED: return ("failed");
		case MEMBRANE_OUTCOME_ROLLED_BACK: return ("rolled_back");
		case MEMBRANE_OUTCOME_ROLLBACK_FAILED: return ("rollback_failed");
	}
	return ("unknown");
}

static void	copy(char *dst, size_t size, const char *src)
{
	if (size == 0)
		return ;
	if (src == NULL)
		src = "";
	snprintf(dst, size, "%s", src);
}

/* ------------------------------------------------------------------ */
/* Fingerprint / stale check                                            */
/* ------------------------------------------------------------------ */

static void	state_add(membrane_apply_state_t *st, const char *key,
				const char *value)
{
	if (st->count >= MEMBRANE_ACTION_MAX_STATE)
		return ;
	copy(st->key[st->count], sizeof(st->key[0]), key);
	copy(st->value[st->count], sizeof(st->value[0]), value);
	st->count++;
}

void	membrane_apply_fingerprint(const membrane_apply_live_t *live,
			membrane_apply_state_t *out)
{
	size_t	order[MEMBRANE_ACTION_MAX_RESIDENT];
	size_t	i;
	size_t	j;
	char	buf[MEMBRANE_ACTION_STATE_VAL_MAX];

	memset(out, 0, sizeof(*out));
	state_add(out, "runtime.reachable", live->runtime_reachable ? "1" : "0");
	state_add(out, "default_model", live->default_model);
	state_add(out, "override.context", live->ctx_known ? live->ctx : "?");
	state_add(out, "override.kv_precision",
		live->kv_precision_known ? live->kv_precision : "?");
	state_add(out, "override.kv_placement",
		live->kv_placement_known ? live->kv_placement : "?");
	state_add(out, "override.gpu_layers",
		live->gpu_layers_known ? live->gpu_layers : "?");
	snprintf(buf, sizeof(buf), "%zu", live->resident_count);
	state_add(out, "resident.count", buf);
	for (i = 0; i < live->resident_count; ++i)
		order[i] = i;
	for (i = 1; i < live->resident_count; ++i)
		for (j = i; j > 0 && strcmp(live->resident[order[j]].name,
				live->resident[order[j - 1]].name) < 0; --j)
		{
			size_t	t = order[j];

			order[j] = order[j - 1];
			order[j - 1] = t;
		}
	for (i = 0; i < live->resident_count; ++i)
	{
		const membrane_apply_resident_t	*r = &live->resident[order[i]];
		char	key[MEMBRANE_ACTION_STATE_KEY_MAX];

		snprintf(key, sizeof(key), "resident.%zu", i);
		snprintf(buf, sizeof(buf), "%.120s|%s|%s", r->name,
			r->activity_known ? (r->pinned ? "pinned" : "unpinned")
				: "pin?",
			r->activity_known ? (r->generating ? "generating" : "idle")
				: "activity?");
		state_add(out, key, buf);
	}
}

static const char	*state_find(const membrane_apply_state_t *st,
						const char *key)
{
	size_t	i;

	for (i = 0; i < st->count; ++i)
		if (strcmp(st->key[i], key) == 0)
			return (st->value[i]);
	return (NULL);
}

size_t	membrane_action_check_stale(const membrane_apply_state_t *basis,
			const membrane_apply_state_t *now, char *drift_out,
			size_t drift_size)
{
	size_t	n = 0;
	size_t	i;
	size_t	len = 0;

	if (drift_out != NULL && drift_size > 0)
		drift_out[0] = '\0';
	for (i = 0; i < basis->count + now->count; ++i)
	{
		int			from_basis = i < basis->count;
		const char	*key = from_basis ? basis->key[i]
			: now->key[i - basis->count];
		const char	*a = state_find(basis, key);
		const char	*b = state_find(now, key);

		if (!from_basis && a != NULL)
			continue ;			/* already compared via the basis pass */
		if (a == NULL || b == NULL || strcmp(a, b) != 0)
		{
			n++;
			if (drift_out != NULL && len + 1 < drift_size)
				len += (size_t)snprintf(drift_out + len, drift_size - len,
						"%s%s", len ? "; " : "", key);
		}
	}
	return (n);
}

int	membrane_action_recommendations_still_hold(
		const membrane_action_plan_t *plan,
		const membrane_memory_intelligence_t *fresh, char *missing,
		size_t missing_size)
{
	size_t	i;
	size_t	j;

	if (missing != NULL && missing_size > 0)
		missing[0] = '\0';
	for (i = 0; i < plan->action_count; ++i)
	{
		const membrane_action_t	*a = &plan->actions[i];
		int						found = 0;

		if (a->origin != MEMBRANE_ACTION_ORIGIN_RECOMMENDATION
			|| a->source_recommendation[0] == '\0')
			continue ;
		for (j = 0; fresh != NULL && j < fresh->recommendation_count; ++j)
			if (strcmp(fresh->recommendations[j].code,
					a->source_recommendation) == 0)
				found = 1;
		if (!found)
		{
			copy(missing, missing_size, a->source_recommendation);
			return (0);
		}
	}
	return (1);
}

/* ------------------------------------------------------------------ */
/* Candidate selection                                                  */
/* ------------------------------------------------------------------ */

int	membrane_action_pick_unload_candidate(const membrane_apply_live_t *live,
		char *why, size_t why_size)
{
	int		best = -1;
	size_t	i;
	size_t	protected_n = 0;
	size_t	unknown_n = 0;
	size_t	default_n = 0;

	for (i = 0; i < live->resident_count; ++i)
	{
		const membrane_apply_resident_t	*r = &live->resident[i];

		if (live->default_model[0] != '\0'
			&& strcmp(r->name, live->default_model) == 0)
		{
			default_n++;
			continue ;
		}
		if (!r->activity_known)
		{
			unknown_n++;
			continue ;
		}
		if (r->pinned || r->generating)
		{
			protected_n++;
			continue ;
		}
		if (best < 0
			|| r->size_bytes > live->resident[best].size_bytes
			|| (r->size_bytes == live->resident[best].size_bytes
				&& strcmp(r->name, live->resident[best].name) < 0))
			best = (int)i;
	}
	if (best < 0 && why != NULL)
		snprintf(why, why_size, "no resident model qualifies: %zu is the "
			"configured default, %zu pinned or generating, %zu with "
			"unknown activity (MEMBRANE never auto-selects a model it "
			"cannot prove idle)", default_n, protected_n, unknown_n);
	return (best);
}

/* ------------------------------------------------------------------ */
/* Compiler                                                             */
/* ------------------------------------------------------------------ */

static void	add_unapplied(membrane_action_plan_t *p, const char *source,
				const char *reason_code, const char *text)
{
	membrane_action_unapplied_t	*u;

	if (p->unapplied_count >= MEMBRANE_ACTION_MAX_UNAPPLIED)
		return ;
	u = &p->unapplied[p->unapplied_count++];
	copy(u->source, sizeof(u->source), source);
	copy(u->reason_code, sizeof(u->reason_code), reason_code);
	copy(u->text, sizeof(u->text), text);
}

/* Returns 1 iff the surface lets `code` become an action; otherwise
 * records why (fail closed on UNKNOWN). */
static int	surface_allows(membrane_action_plan_t *p,
				const membrane_apply_surface_t *surface,
				membrane_action_code_t code, const char *source)
{
	membrane_capability_state_t	st = MEMBRANE_CAPABILITY_UNKNOWN;
	char						text[MEMBRANE_ACTION_TEXT_MAX];

	if (surface != NULL)
		st = surface->state[code];
	if (st == MEMBRANE_CAPABILITY_SUPPORTED)
		return (1);
	snprintf(text, sizeof(text), "%s is %s on %s: %.120s",
		membrane_action_code_name(code), st == MEMBRANE_CAPABILITY_UNKNOWN
			? "not known to be controllable" : "not controllable",
		p->runtime_id, surface != NULL && surface->reason[code][0]
			? surface->reason[code] : "no control primitive");
	add_unapplied(p, source, st == MEMBRANE_CAPABILITY_UNKNOWN
		? MEMBRANE_UNAPPLIED_UNKNOWN_CAP : MEMBRANE_UNAPPLIED_UNSUPPORTED,
		text);
	return (0);
}

static membrane_action_t	*new_action(membrane_action_plan_t *p,
								membrane_action_code_t code,
								const membrane_apply_surface_t *surface)
{
	membrane_action_t	*a;

	if (p->action_count >= MEMBRANE_ACTION_MAX_ACTIONS)
		return (NULL);
	a = &p->actions[p->action_count++];
	memset(a, 0, sizeof(*a));
	a->code = code;
	copy(a->runtime_id, sizeof(a->runtime_id), p->runtime_id);
	a->capability = MEMBRANE_CAPABILITY_SUPPORTED;
	a->executable = 1;
	a->mutates_state = 1;
	a->requires_reload = surface != NULL ? surface->requires_reload[code] : 0;
	if (a->requires_reload)
		p->any_requires_reload = 1;
	return (a);
}

static int	all_digits(const char *s)
{
	if (*s == '\0')
		return (0);
	for (; *s; ++s)
		if (*s < '0' || *s > '9')
			return (0);
	return (1);
}

static const char	*rec_for(membrane_action_code_t code)
{
	switch (code)
	{
		case MEMBRANE_ACTION_SET_CONTEXT:
			return (MEMBRANE_INTEL_REC_REDUCE_CONTEXT);
		case MEMBRANE_ACTION_SET_KV_PRECISION:
			return (MEMBRANE_INTEL_REC_USE_LOWER_KV_PRECISION);
		case MEMBRANE_ACTION_SET_KV_PLACEMENT:
			return (MEMBRANE_INTEL_REC_CHANGE_KV_PLACEMENT);
		case MEMBRANE_ACTION_SET_GPU_LAYERS:
			return (MEMBRANE_INTEL_REC_REDUCE_GPU_LAYERS);
		case MEMBRANE_ACTION_UNLOAD_MODEL:
			return (MEMBRANE_INTEL_REC_UNLOAD_UNUSED_MODEL);
		default:
			return ("");
	}
}

static int	intel_has(const membrane_memory_intelligence_t *intel,
				const char *code)
{
	size_t	i;

	if (intel == NULL || code[0] == '\0')
		return (0);
	for (i = 0; i < intel->recommendation_count; ++i)
		if (strcmp(intel->recommendations[i].code, code) == 0)
			return (1);
	return (0);
}

/* One SET_* request. */
static void	compile_set(membrane_action_plan_t *p,
				const membrane_memory_intelligence_t *intel,
				const membrane_apply_surface_t *surface,
				const membrane_apply_live_t *live,
				const membrane_action_validation_t *validation,
				membrane_action_code_t code, const char *value)
{
	const char			*cur = "";
	int					cur_known = 0;
	const char			*what = membrane_action_code_name(code);
	membrane_action_t	*a;
	char				text[MEMBRANE_ACTION_TEXT_MAX];
	int					ok = 0;

	switch (code)
	{
		case MEMBRANE_ACTION_SET_CONTEXT:
			ok = (all_digits(value) && strlen(value) <= 9
				&& atol(value) >= 16) || !strcmp(value, "automatic");
			cur = live->ctx; cur_known = live->ctx_known;
			break ;
		case MEMBRANE_ACTION_SET_KV_PRECISION:
			ok = !strcmp(value, "native") || !strcmp(value, "q8")
				|| !strcmp(value, "q5") || !strcmp(value, "automatic");
			cur = live->kv_precision; cur_known = live->kv_precision_known;
			break ;
		case MEMBRANE_ACTION_SET_KV_PLACEMENT:
			ok = !strcmp(value, "default") || !strcmp(value, "cpu")
				|| !strcmp(value, "gpu") || !strcmp(value, "auto")
				|| !strcmp(value, "automatic");
			cur = live->kv_placement; cur_known = live->kv_placement_known;
			break ;
		default:
			ok = !strcmp(value, "all") || !strcmp(value, "auto")
				|| (all_digits(value) && strlen(value) <= 5)
				|| !strcmp(value, "automatic");
			cur = live->gpu_layers; cur_known = live->gpu_layers_known;
			break ;
	}
	if (!surface_allows(p, surface, code, what))
		return ;
	if (!ok)
	{
		snprintf(text, sizeof(text), "%s: '%.60s' is not a valid value",
			what, value);
		add_unapplied(p, what, MEMBRANE_UNAPPLIED_REJECTED, text);
		return ;
	}
	if (cur_known && strcmp(cur, value) == 0)
	{
		snprintf(text, sizeof(text), "%s is already %.60s", what, value);
		add_unapplied(p, what, MEMBRANE_UNAPPLIED_NO_CHANGE, text);
		return ;
	}
	if (validation == NULL || !validation->checked || !validation->feasible)
	{
		snprintf(text, sizeof(text), "%s %.60s was not accepted by Planner "
			"v2: %.100s", what, value, validation != NULL
				&& validation->detail[0] ? validation->detail
				: "validation unavailable");
		add_unapplied(p, what, MEMBRANE_UNAPPLIED_NOT_VALIDATED, text);
		return ;
	}
	a = new_action(p, code, surface);
	if (a == NULL)
		return ;
	copy(a->requested, sizeof(a->requested), value);
	a->current_known = cur_known;
	copy(a->current, sizeof(a->current), cur_known ? cur : "");
	a->origin = MEMBRANE_ACTION_ORIGIN_USER_REQUEST;
	if (intel_has(intel, rec_for(code)))
		copy(a->source_recommendation, sizeof(a->source_recommendation),
			rec_for(code));
	copy(a->reason, sizeof(a->reason), validation->detail);
	a->risk = MEMBRANE_ACTION_RISK_MEDIUM;
	a->reversible = cur_known && surface->reversible[code];
	a->rollback_known = a->reversible;
	copy(a->rollback_value, sizeof(a->rollback_value), cur_known ? cur : "");
	snprintf(a->summary, sizeof(a->summary), "%s: %s -> %s%s", what,
		cur_known ? cur : "(unknown)", value,
		a->requires_reload ? " (takes effect when the model is next "
			"loaded)" : "");
}

static int	find_resident(const membrane_apply_live_t *live, const char *name)
{
	size_t	i;

	for (i = 0; i < live->resident_count; ++i)
		if (strcmp(live->resident[i].name, name) == 0)
			return ((int)i);
	return (-1);
}

static void	compile_unload(membrane_action_plan_t *p,
				const membrane_apply_surface_t *surface,
				const membrane_apply_live_t *live, const char *model,
				membrane_action_origin_t origin, const char *source_rec,
				const char *why_chosen)
{
	int					idx;
	membrane_action_t	*a;
	char				text[MEMBRANE_ACTION_TEXT_MAX];

	if (!surface_allows(p, surface, MEMBRANE_ACTION_UNLOAD_MODEL,
			source_rec[0] ? source_rec : "UNLOAD_MODEL"))
		return ;
	idx = find_resident(live, model);
	if (idx < 0)
	{
		snprintf(text, sizeof(text), "UNLOAD_MODEL: '%.100s' is not "
			"resident", model);
		add_unapplied(p, "UNLOAD_MODEL", MEMBRANE_UNAPPLIED_REJECTED, text);
		return ;
	}
	if (live->resident[idx].activity_known
		&& (live->resident[idx].pinned || live->resident[idx].generating))
	{
		snprintf(text, sizeof(text), "UNLOAD_MODEL: '%.100s' is %s -- "
			"MEMBRANE never unloads %s", model,
			live->resident[idx].pinned ? "pinned" : "generating",
			live->resident[idx].pinned ? "a pinned model silently"
				: "a model that is generating");
		add_unapplied(p, "UNLOAD_MODEL", MEMBRANE_UNAPPLIED_REJECTED, text);
		return ;
	}
	a = new_action(p, MEMBRANE_ACTION_UNLOAD_MODEL, surface);
	if (a == NULL)
		return ;
	copy(a->model, sizeof(a->model), model);
	copy(a->requested, sizeof(a->requested), "unloaded");
	a->current_known = 1;
	copy(a->current, sizeof(a->current), "resident");
	a->origin = origin;
	copy(a->source_recommendation, sizeof(a->source_recommendation),
		source_rec);
	copy(a->reason, sizeof(a->reason), why_chosen);
	/* Unload is recoverable only by reloading (slow, may itself fail
	 * under the very pressure that motivated it) -- medium risk, and
	 * for a runtime that does not report activity, never auto-chosen. */
	a->risk = MEMBRANE_ACTION_RISK_MEDIUM;
	a->reversible = surface->reversible[MEMBRANE_ACTION_UNLOAD_MODEL];
	a->rollback_known = a->reversible;
	snprintf(a->rollback_value, sizeof(a->rollback_value), "reload");
	snprintf(a->summary, sizeof(a->summary), "UNLOAD_MODEL %.100s%s",
		model, live->resident[idx].activity_known ? ""
			: " (this runtime does not report whether it is generating)");
}

static void	compile_pin(membrane_action_plan_t *p,
				const membrane_apply_surface_t *surface,
				const membrane_apply_live_t *live, const char *model,
				int pin)
{
	membrane_action_code_t	code = pin ? MEMBRANE_ACTION_PIN_MODEL
		: MEMBRANE_ACTION_UNPIN_MODEL;
	const char				*what = membrane_action_code_name(code);
	int						idx;
	membrane_action_t		*a;
	char					text[MEMBRANE_ACTION_TEXT_MAX];

	if (!surface_allows(p, surface, code, what))
		return ;
	idx = find_resident(live, model);
	if (idx < 0 || !live->resident[idx].activity_known)
	{
		snprintf(text, sizeof(text), "%s: '%.100s' is not resident (or its "
			"pin state is unknown)", what, model);
		add_unapplied(p, what, MEMBRANE_UNAPPLIED_REJECTED, text);
		return ;
	}
	if ((live->resident[idx].pinned != 0) == (pin != 0))
	{
		snprintf(text, sizeof(text), "%.100s is already %s", model,
			pin ? "pinned" : "unpinned");
		add_unapplied(p, what, MEMBRANE_UNAPPLIED_NO_CHANGE, text);
		return ;
	}
	a = new_action(p, code, surface);
	if (a == NULL)
		return ;
	copy(a->model, sizeof(a->model), model);
	copy(a->requested, sizeof(a->requested), pin ? "pinned" : "unpinned");
	a->current_known = 1;
	copy(a->current, sizeof(a->current), pin ? "unpinned" : "pinned");
	a->origin = MEMBRANE_ACTION_ORIGIN_USER_REQUEST;
	copy(a->reason, sizeof(a->reason), "explicit request");
	a->risk = MEMBRANE_ACTION_RISK_LOW;
	a->reversible = surface->reversible[code];
	a->rollback_known = a->reversible;
	copy(a->rollback_value, sizeof(a->rollback_value), a->current);
	snprintf(a->summary, sizeof(a->summary), "%s %.100s", what, model);
}

static void	compile_activate(membrane_action_plan_t *p,
				const membrane_apply_surface_t *surface,
				const membrane_apply_live_t *live, const char *model,
				int reload)
{
	int					idx = find_resident(live, model);
	membrane_action_t	*a;
	char				text[MEMBRANE_ACTION_TEXT_MAX];

	if (!surface_allows(p, surface, MEMBRANE_ACTION_ACTIVATE_MODEL,
			"ACTIVATE_MODEL"))
		return ;
	if (reload && idx < 0)
	{
		snprintf(text, sizeof(text), "reload: '%.100s' is not resident",
			model);
		add_unapplied(p, "ACTIVATE_MODEL", MEMBRANE_UNAPPLIED_REJECTED, text);
		return ;
	}
	if (reload && live->resident[idx].activity_known
		&& (live->resident[idx].pinned || live->resident[idx].generating))
	{
		snprintf(text, sizeof(text), "reload: '%.100s' is %s -- not "
			"reloaded", model, live->resident[idx].pinned ? "pinned"
				: "generating");
		add_unapplied(p, "ACTIVATE_MODEL", MEMBRANE_UNAPPLIED_REJECTED, text);
		return ;
	}
	if (!reload && idx >= 0)
	{
		snprintf(text, sizeof(text), "%.100s is already resident", model);
		add_unapplied(p, "ACTIVATE_MODEL", MEMBRANE_UNAPPLIED_NO_CHANGE, text);
		return ;
	}
	a = new_action(p, MEMBRANE_ACTION_ACTIVATE_MODEL, surface);
	if (a == NULL)
		return ;
	copy(a->model, sizeof(a->model), model);
	copy(a->requested, sizeof(a->requested), reload ? "reloaded" : "resident");
	a->current_known = 1;
	copy(a->current, sizeof(a->current), idx >= 0 ? "resident" : "not resident");
	a->origin = MEMBRANE_ACTION_ORIGIN_USER_REQUEST;
	copy(a->reason, sizeof(a->reason), reload
		? "explicit reload so applied settings take effect now"
		: "explicit request");
	a->risk = MEMBRANE_ACTION_RISK_MEDIUM;
	if (reload)
	{
		/* unload + activate. If it fails after the unload, the engine
		 * restores the model (after reverting settings). */
		a->reversible = surface->reversible[MEMBRANE_ACTION_ACTIVATE_MODEL];
		a->rollback_known = a->reversible;
		a->restore_on_failure = a->reversible;
		copy(a->rollback_value, sizeof(a->rollback_value), "reactivate");
	}
	else
	{
		a->reversible = 0;	/* loading may evict another resident model;
							 * that eviction is not undone automatically */
		a->rollback_known = 0;
	}
	snprintf(a->summary, sizeof(a->summary), "%s %.100s%s",
		reload ? "RELOAD (unload + activate)" : "ACTIVATE_MODEL", model,
		reload ? "" : " (may evict another resident model)");
}

static void	sort_actions(membrane_action_plan_t *p)
{
	size_t	i;
	size_t	j;

	for (i = 1; i < p->action_count; ++i)
		for (j = i; j > 0; --j)
		{
			membrane_action_t	*x = &p->actions[j - 1];
			membrane_action_t	*y = &p->actions[j];

			if (x->code < y->code || (x->code == y->code
					&& strcmp(x->model, y->model) <= 0))
				break ;
			{
				membrane_action_t	t = *x;

				*x = *y;
				*y = t;
			}
		}
}

void	membrane_action_compile(const membrane_memory_intelligence_t *intel,
		const membrane_apply_surface_t *surface,
		const membrane_apply_live_t *live,
		const membrane_action_request_t *request,
		const membrane_action_validation_t *validation,
		membrane_action_plan_t *out)
{
	size_t				i;
	int					unload_requested = request->unload[0] != '\0';
	char				text[MEMBRANE_ACTION_TEXT_MAX];
	static const membrane_action_code_t	set_codes[4] = {
		MEMBRANE_ACTION_SET_CONTEXT, MEMBRANE_ACTION_SET_KV_PRECISION,
		MEMBRANE_ACTION_SET_KV_PLACEMENT, MEMBRANE_ACTION_SET_GPU_LAYERS};
	const char			*set_vals[4];
	int					set_done[MEMBRANE_ACTION_CODE_COUNT] = {0};

	memset(out, 0, sizeof(*out));
	out->schema_version = MEMBRANE_ACTION_SCHEMA_VERSION;
	copy(out->runtime_id, sizeof(out->runtime_id), request->runtime_id[0]
		? request->runtime_id : intel != NULL ? intel->runtime_id : "");
	membrane_apply_fingerprint(live, &out->basis);
	set_vals[0] = request->context;
	set_vals[1] = request->kv_precision;
	set_vals[2] = request->kv_placement;
	set_vals[3] = request->gpu_layers;

	/* 1. Explicit user requests (J10). */
	for (i = 0; i < 4; ++i)
		if (set_vals[i][0] != '\0')
		{
			compile_set(out, intel, surface, live, validation, set_codes[i],
				set_vals[i]);
			set_done[set_codes[i]] = 1;
		}
	if (request->unload[0] != '\0')
		compile_unload(out, surface, live, request->unload,
			MEMBRANE_ACTION_ORIGIN_USER_REQUEST,
			intel_has(intel, MEMBRANE_INTEL_REC_UNLOAD_UNUSED_MODEL)
				? MEMBRANE_INTEL_REC_UNLOAD_UNUSED_MODEL : "",
			"explicit request");
	if (request->unpin[0] != '\0')
		compile_pin(out, surface, live, request->unpin, 0);
	if (request->activate[0] != '\0')
		compile_activate(out, surface, live, request->activate, 0);
	if (request->pin[0] != '\0')
		compile_pin(out, surface, live, request->pin, 1);
	if (request->reload)
	{
		const char	*target = request->model;

		if (target[0] == '\0' && live->resident_count == 1)
			target = live->resident[0].name;
		if (target[0] == '\0')
			add_unapplied(out, "ACTIVATE_MODEL", MEMBRANE_UNAPPLIED_REJECTED,
				"--reload needs --model NAME when zero or several models "
				"are resident");
		else
			compile_activate(out, surface, live, target, 1);
	}

	/* 2. Recommendations. Never produce a value out of thin air. */
	for (i = 0; intel != NULL && i < intel->recommendation_count; ++i)
	{
		const membrane_intel_recommendation_t	*r = &intel->recommendations[i];
		size_t									k;
		int										dup = 0;
		membrane_action_code_t					code = MEMBRANE_ACTION_CODE_COUNT;

		for (k = 0; k < i; ++k)
			if (strcmp(intel->recommendations[k].code, r->code) == 0)
				dup = 1;
		if (dup || strcmp(r->code, MEMBRANE_INTEL_REC_NO_ACTION_NEEDED) == 0)
			continue ;
		for (k = 0; k < 4; ++k)
			if (strcmp(r->code, rec_for(set_codes[k])) == 0)
				code = set_codes[k];
		if (code != MEMBRANE_ACTION_CODE_COUNT)
		{
			if (set_done[code])
				continue ;				/* satisfied by the user request */
			if (!surface_allows(out, surface, code, r->code))
				continue ;
			snprintf(text, sizeof(text), "%.40s names a direction, not a "
				"value: no deterministic target, so nothing is applied "
				"automatically. Pass the matching `membrane apply` flag "
				"to choose one (validated by Planner v2).", r->code);
			add_unapplied(out, r->code, MEMBRANE_UNAPPLIED_NO_TARGET, text);
			continue ;
		}
		if (strcmp(r->code, MEMBRANE_INTEL_REC_UNLOAD_UNUSED_MODEL) == 0)
		{
			char	why[MEMBRANE_ACTION_TEXT_MAX] = "";
			int		idx;

			if (unload_requested)
				continue ;
			if (!surface_allows(out, surface, MEMBRANE_ACTION_UNLOAD_MODEL,
					r->code))
				continue ;
			idx = membrane_action_pick_unload_candidate(live, why,
					sizeof(why));
			if (idx < 0)
			{
				add_unapplied(out, r->code, MEMBRANE_UNAPPLIED_NO_CANDIDATE,
					why);
				continue ;
			}
			snprintf(text, sizeof(text), "largest idle, unpinned, non-"
				"default resident model (ties: name order)");
			compile_unload(out, surface, live, live->resident[idx].name,
				MEMBRANE_ACTION_ORIGIN_RECOMMENDATION, r->code, text);
			continue ;
		}
		add_unapplied(out, r->code, MEMBRANE_UNAPPLIED_NOT_ACTIONABLE,
			"no MEMBRANE control primitive implements this recommendation; "
			"it stays advisory");
	}
	sort_actions(out);
}

/* ------------------------------------------------------------------ */
/* Transaction engine                                                   */
/* ------------------------------------------------------------------ */

void	membrane_action_run(const membrane_action_plan_t *plan,
		const membrane_action_executor_t *exec,
		membrane_action_result_t *out)
{
	membrane_apply_live_t	live;
	membrane_apply_state_t	now;
	char					drift[120];
	size_t					i;
	long					failed = -1;
	int						rollback_failed = 0;
	int						irreversible_left = 0;
	char					err[MEMBRANE_ACTION_TEXT_MAX];

	memset(out, 0, sizeof(*out));
	if (plan->action_count == 0)
	{
		out->status = MEMBRANE_APPLY_NOTHING_TO_APPLY;
		copy(out->detail, sizeof(out->detail), "no executable action");
		return ;
	}
	memset(&live, 0, sizeof(live));
	if (exec == NULL || exec->read_live == NULL || exec->execute == NULL
		|| !exec->read_live(exec->ctx, &live) || !live.runtime_reachable)
	{
		out->status = MEMBRANE_APPLY_REFUSED;
		copy(out->detail, sizeof(out->detail), "runtime state could not be "
			"re-read; nothing was changed");
		return ;
	}
	membrane_apply_fingerprint(&live, &now);
	if (membrane_action_check_stale(&plan->basis, &now, drift,
			sizeof(drift)) > 0)
	{
		out->status = MEMBRANE_APPLY_STALE_PRECONDITION;
		snprintf(out->detail, sizeof(out->detail), "live state changed "
			"since the plan was made (%s); nothing was changed -- rerun "
			"`membrane advise` / `membrane apply`", drift);
		return ;
	}
	for (i = 0; i < plan->action_count; ++i)
	{
		err[0] = '\0';
		if (exec->execute(exec->ctx, &plan->actions[i], err, sizeof(err)))
		{
			out->outcome[i] = MEMBRANE_OUTCOME_APPLIED;
			out->applied_count++;
			continue ;
		}
		out->outcome[i] = MEMBRANE_OUTCOME_FAILED;
		copy(out->error[i], sizeof(out->error[i]), err[0] ? err
			: "action failed");
		failed = (long)i;
		break ;
	}
	if (failed < 0)
	{
		out->status = MEMBRANE_APPLY_APPLIED;
		copy(out->detail, sizeof(out->detail), "all actions applied");
		return ;
	}
	/* Roll back already-applied reversible actions, newest first. */
	for (i = (size_t)failed; i-- > 0; )
	{
		const membrane_action_t	*a = &plan->actions[i];

		if (out->outcome[i] != MEMBRANE_OUTCOME_APPLIED)
			continue ;
		if (!a->reversible || !a->rollback_known || exec->rollback == NULL)
		{
			irreversible_left++;
			continue ;
		}
		err[0] = '\0';
		if (exec->rollback(exec->ctx, a, err, sizeof(err)))
		{
			out->outcome[i] = MEMBRANE_OUTCOME_ROLLED_BACK;
			out->applied_count--;
		}
		else
		{
			out->outcome[i] = MEMBRANE_OUTCOME_ROLLBACK_FAILED;
			copy(out->error[i], sizeof(out->error[i]), err[0] ? err
				: "rollback failed");
			rollback_failed++;
		}
	}
	if (plan->actions[failed].restore_on_failure && exec->rollback != NULL)
	{
		err[0] = '\0';
		if (exec->rollback(exec->ctx, &plan->actions[failed], err,
				sizeof(err)))
			out->outcome[failed] = MEMBRANE_OUTCOME_ROLLED_BACK;
		else
		{
			out->outcome[failed] = MEMBRANE_OUTCOME_ROLLBACK_FAILED;
			if (err[0])
				snprintf(out->error[failed] + strlen(out->error[failed]),
					sizeof(out->error[failed]) - strlen(out->error[failed]),
					" | restore: %s", err);
			rollback_failed++;
		}
	}
	if (rollback_failed)
		out->status = MEMBRANE_APPLY_ROLLBACK_FAILED;
	else if (irreversible_left)
		out->status = MEMBRANE_APPLY_PARTIALLY_APPLIED;
	else if (out->applied_count == 0 && failed == 0
		&& out->outcome[0] != MEMBRANE_OUTCOME_ROLLED_BACK)
		out->status = MEMBRANE_APPLY_FAILED;
	else if (out->applied_count == 0 && failed > 0)
		out->status = MEMBRANE_APPLY_ROLLED_BACK;
	else if (out->applied_count == 0)
		out->status = MEMBRANE_APPLY_ROLLED_BACK;
	else
		out->status = MEMBRANE_APPLY_PARTIALLY_APPLIED;
	snprintf(out->detail, sizeof(out->detail), "action %ld (%s) failed; %zu "
		"action(s) remain applied", failed + 1,
		membrane_action_code_name(plan->actions[failed].code),
		out->applied_count);
}
