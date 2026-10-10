#include "adaptive.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* See adaptive.h's top comment. Pure: no clock, no I/O, no threads. */

#define GIB	(1024ull * 1024 * 1024)

static void	copy(char *dst, size_t n, const char *src)
{
	if (n == 0)
		return ;
	snprintf(dst, n, "%s", src != NULL ? src : "");
}

/* ------------------------------------------------------------------ */
/* names / parsing                                                      */
/* ------------------------------------------------------------------ */

static const char	*g_modes[4] = {"off", "observe", "recommend", "apply"};

const char	*membrane_adaptive_mode_name(membrane_adaptive_mode_t m)
{
	if ((int)m < 0 || (int)m > 3)
		return ("off");
	return (g_modes[m]);
}

int	membrane_adaptive_mode_parse(const char *s, membrane_adaptive_mode_t *out)
{
	int	i;

	for (i = 0; s != NULL && i < 4; ++i)
		if (strcmp(s, g_modes[i]) == 0)
		{
			*out = (membrane_adaptive_mode_t)i;
			return (1);
		}
	return (0);
}

static const struct { const char *name; int bit; }	g_classes[5] = {
	{"context", MEMBRANE_ADAPTIVE_CLASS_CONTEXT},
	{"kv_precision", MEMBRANE_ADAPTIVE_CLASS_KV_PRECISION},
	{"kv_placement", MEMBRANE_ADAPTIVE_CLASS_KV_PLACEMENT},
	{"gpu_layers", MEMBRANE_ADAPTIVE_CLASS_GPU_LAYERS},
	{"unload", MEMBRANE_ADAPTIVE_CLASS_UNLOAD}};

int	membrane_adaptive_classes_parse(const char *csv)
{
	int			mask = 0;
	const char	*cur = csv;

	if (csv == NULL || csv[0] == '\0')
		return (-1);
	while (1)
	{
		const char	*end = strchr(cur, ',');
		size_t		len = end != NULL ? (size_t)(end - cur) : strlen(cur);
		int			i;
		int			found = 0;

		for (i = 0; i < 5; ++i)
			if (len == strlen(g_classes[i].name)
				&& strncmp(cur, g_classes[i].name, len) == 0)
			{
				mask |= g_classes[i].bit;
				found = 1;
			}
		if (!found)
			return (-1);
		if (end == NULL)
			break ;
		cur = end + 1;
	}
	return (mask);
}

void	membrane_adaptive_classes_format(int mask, char *out, size_t n)
{
	int		i;
	size_t	len = 0;

	if (n == 0)
		return ;
	out[0] = '\0';
	for (i = 0; i < 5; ++i)
		if ((mask & g_classes[i].bit) && len + 1 < n)
			len += (size_t)snprintf(out + len, n - len, "%s%s",
					len ? "," : "", g_classes[i].name);
}

const char	*membrane_pressure_name(membrane_pressure_t p)
{
	switch (p)
	{
		case MEMBRANE_PRESSURE_HEALTHY: return ("healthy");
		case MEMBRANE_PRESSURE_WARNING: return ("warning");
		case MEMBRANE_PRESSURE_CRITICAL: return ("critical");
		default: return ("unknown");
	}
}

const char	*membrane_adaptive_decision_name(
				membrane_adaptive_decision_kind_t k)
{
	switch (k)
	{
		case MEMBRANE_DECISION_NONE: return ("none");
		case MEMBRANE_DECISION_OBSERVED: return ("observed");
		case MEMBRANE_DECISION_TRANSIENT: return ("transient");
		case MEMBRANE_DECISION_COOLDOWN: return ("cooldown");
		case MEMBRANE_DECISION_RATE_LIMITED: return ("rate_limited");
		case MEMBRANE_DECISION_SUPPRESSED: return ("suppressed");
		case MEMBRANE_DECISION_MANUAL_OVERRIDE: return ("manual_override");
		case MEMBRANE_DECISION_DEFERRED: return ("deferred");
		case MEMBRANE_DECISION_RECOMMEND: return ("recommend");
		case MEMBRANE_DECISION_ACT: return ("act");
		case MEMBRANE_DECISION_NO_SAFE_STEP: return ("no_safe_step");
	}
	return ("none");
}

/* ------------------------------------------------------------------ */
/* policy                                                               */
/* ------------------------------------------------------------------ */

void	membrane_adaptive_policy_defaults(membrane_adaptive_policy_t *p)
{
	memset(p, 0, sizeof(*p));
	p->enabled = 0;
	p->mode = MEMBRANE_ADAPTIVE_OFF;
	copy(p->runtime_scope, sizeof(p->runtime_scope), MEMBRANE_RUNTIME_ID_NATIVE);
	p->interval_s = 30;
	p->cooldown_s = 600;
	p->sustained_samples = 3;
	p->recovery_samples = 3;
	p->warning_exit_permille = 200;
	p->critical_exit_permille = 100;
	p->ram_exit_abs_bytes = 1 * GIB;
	p->max_actions_per_window = 2;
	p->window_s = 3600;
	p->allowed_classes = MEMBRANE_ADAPTIVE_CLASS_CONTEXT
		| MEMBRANE_ADAPTIVE_CLASS_KV_PRECISION;
	p->min_pressure = MEMBRANE_PRESSURE_WARNING;
	p->allow_estimate_only = 0;
	p->auto_restore = 0;
	p->restore_after_s = 900;
	p->thrash_reversals = 3;
	p->thrash_window_s = 3600;
	p->suppress_s = 1800;
	p->history_max_samples = 120;
	p->history_max_age_s = 3600;
}

const char	*membrane_adaptive_policy_validate(
				const membrane_adaptive_policy_t *p)
{
	if (p->interval_s < MEMBRANE_ADAPTIVE_MIN_INTERVAL_S
		|| p->interval_s > 86400)
		return ("interval_s must be 5..86400");
	if (p->cooldown_s < 0 || p->cooldown_s > 86400)
		return ("cooldown_s must be 0..86400");
	if (p->sustained_samples < 1 || p->sustained_samples > 100
		|| p->recovery_samples < 1 || p->recovery_samples > 100)
		return ("sustained/recovery samples must be 1..100");
	if (p->warning_exit_permille <= 150 || p->warning_exit_permille > 1000)
		return ("warning_exit_permille must exceed the 150 enter threshold");
	if (p->critical_exit_permille <= 50
		|| p->critical_exit_permille > p->warning_exit_permille)
		return ("critical_exit_permille must exceed 50 and not exceed the "
			"warning exit");
	if (p->max_actions_per_window < 0 || p->max_actions_per_window > 100
		|| p->window_s < 1 || p->window_s > 604800)
		return ("action rate limit out of range");
	if ((p->allowed_classes & ~MEMBRANE_ADAPTIVE_CLASS_ALL) != 0)
		return ("unknown action class bit");
	if (p->min_pressure != MEMBRANE_PRESSURE_WARNING
		&& p->min_pressure != MEMBRANE_PRESSURE_CRITICAL)
		return ("min_pressure must be warning or critical");
	if (p->thrash_reversals < 2 || p->thrash_reversals > 20
		|| p->thrash_window_s < 1 || p->suppress_s < 1)
		return ("anti-thrashing settings out of range");
	if (p->history_max_samples < 1
		|| p->history_max_samples > MEMBRANE_ADAPTIVE_MAX_SAMPLES
		|| p->history_max_age_s < 1)
		return ("history bounds out of range");
	if (p->runtime_scope[0] == '\0')
		return ("runtime_scope is required");
	return ("");
}

membrane_adaptive_mode_t	membrane_adaptive_effective_mode(
				const membrane_adaptive_policy_t *p)
{
	return (p->enabled ? p->mode : MEMBRANE_ADAPTIVE_OFF);
}

/* ------------------------------------------------------------------ */
/* samples                                                              */
/* ------------------------------------------------------------------ */

static int	permille(uint64_t part, uint64_t whole)
{
	if (whole == 0)
		return (-1);
	if (part > whole)
		part = whole;
	return ((int)((part * 1000) / whole));
}

void	membrane_adaptive_make_sample(
			const membrane_observation_snapshot_t *snap,
			const membrane_memory_intelligence_t *intel,
			const membrane_apply_live_t *live, int64_t now_ms,
			membrane_adaptive_sample_t *out)
{
	size_t	i;

	memset(out, 0, sizeof(*out));
	out->t_ms = now_ms;
	out->ram_permille = -1;
	out->vram_permille = -1;
	out->measured = MEMBRANE_PRESSURE_UNKNOWN;
	out->valid = snap != NULL && snap->status != MEMBRANE_OBS_STATUS_UNAVAILABLE
		&& intel != NULL;
	if (!out->valid)
		return ;
	out->measured = MEMBRANE_PRESSURE_HEALTHY;
	if (snap->ram_total_bytes.known && snap->ram_headroom_bytes.known)
	{
		out->ram_permille = permille(snap->ram_headroom_bytes.value,
				snap->ram_total_bytes.value);
		out->ram_free_bytes = snap->ram_headroom_bytes.value;
		out->ram_known = 1;
	}
	if (snap->vram_total_bytes.known && snap->vram_headroom_bytes.known)
		out->vram_permille = permille(snap->vram_headroom_bytes.value,
				snap->vram_total_bytes.value);
	for (i = 0; i < intel->finding_count; ++i)
	{
		const membrane_intel_finding_t	*f = &intel->findings[i];

		if (f->provenance == MEMBRANE_OBS_PROV_MEASURED
			&& (f->dimension == MEMBRANE_INTEL_DIM_HOST_MEMORY
				|| f->dimension == MEMBRANE_INTEL_DIM_VRAM)
			&& f->severity >= MEMBRANE_INTEL_SEVERITY_WARNING)
		{
			membrane_pressure_t	lvl = f->severity
				== MEMBRANE_INTEL_SEVERITY_CRITICAL
				? MEMBRANE_PRESSURE_CRITICAL : MEMBRANE_PRESSURE_WARNING;

			if (lvl > out->measured)
				out->measured = lvl;
			if (f->dimension == MEMBRANE_INTEL_DIM_VRAM)
				out->vram_pressured = 1;
			else
				out->ram_pressured = 1;
		}
		else if (f->provenance == MEMBRANE_OBS_PROV_ESTIMATED
			&& f->severity >= MEMBRANE_INTEL_SEVERITY_NOTICE)
			out->estimate_only = 1;
	}
	for (i = 0; live != NULL && i < live->resident_count; ++i)
		if (live->resident[i].activity_known && live->resident[i].generating)
			out->generating = 1;
}

void	membrane_adaptive_state_init(membrane_adaptive_state_t *s,
			int64_t now_ms)
{
	memset(s, 0, sizeof(*s));
	s->started_ms = now_ms;
	s->last_action_ms = -1;
	s->pressure = MEMBRANE_PRESSURE_UNKNOWN;
}

static int	min_permille(const membrane_adaptive_sample_t *x)
{
	int	m = 1000;
	int	any = 0;

	if (x->ram_permille >= 0)
	{
		m = x->ram_permille;
		any = 1;
	}
	if (x->vram_permille >= 0 && (!any || x->vram_permille < m))
	{
		m = x->vram_permille;
		any = 1;
	}
	return (any ? m : -1);
}

static int	exit_ok(const membrane_adaptive_policy_t *p,
				membrane_pressure_t from,
				const membrane_adaptive_sample_t *x)
{
	int	m = min_permille(x);

	if (m < 0)
		return (0);			/* cannot prove recovery without headroom data */
	if (from == MEMBRANE_PRESSURE_CRITICAL)
		return (x->measured <= MEMBRANE_PRESSURE_WARNING
			&& m >= p->critical_exit_permille);
	return (x->measured == MEMBRANE_PRESSURE_HEALTHY
		&& m >= p->warning_exit_permille
		&& (!x->ram_known || x->ram_free_bytes >= p->ram_exit_abs_bytes));
}

membrane_pressure_t	membrane_adaptive_record_sample(
			const membrane_adaptive_policy_t *p,
			membrane_adaptive_state_t *s,
			const membrane_adaptive_sample_t *x)
{
	size_t	keep_from = 0;
	size_t	i;
	int		crit_run = 0;
	int		need_crit;

	/* append, then prune by age and by count (oldest dropped first) */
	if (s->sample_count >= MEMBRANE_ADAPTIVE_MAX_SAMPLES)
	{
		memmove(&s->samples[0], &s->samples[1],
			(MEMBRANE_ADAPTIVE_MAX_SAMPLES - 1) * sizeof(s->samples[0]));
		s->sample_count = MEMBRANE_ADAPTIVE_MAX_SAMPLES - 1;
	}
	s->samples[s->sample_count++] = *x;
	for (i = 0; i < s->sample_count; ++i)
		if (x->t_ms - s->samples[i].t_ms <= (int64_t)p->history_max_age_s * 1000)
			break ;
	keep_from = i;
	if (s->sample_count - keep_from > (size_t)p->history_max_samples)
		keep_from = s->sample_count - (size_t)p->history_max_samples;
	if (keep_from > 0)
	{
		memmove(&s->samples[0], &s->samples[keep_from],
			(s->sample_count - keep_from) * sizeof(s->samples[0]));
		s->sample_count -= keep_from;
	}
	if (!x->valid)
		return (s->pressure);

	/* consecutive-run counters */
	if (x->measured >= MEMBRANE_PRESSURE_WARNING)
	{
		s->above_count++;
	}
	else
		s->above_count = 0;
	for (i = s->sample_count; i-- > 0; )
	{
		if (!s->samples[i].valid)
			continue ;
		if (s->samples[i].measured != MEMBRANE_PRESSURE_CRITICAL)
			break ;
		crit_run++;
	}
	need_crit = p->sustained_samples / 2;
	if (need_crit < 1)
		need_crit = 1;

	if (x->measured == MEMBRANE_PRESSURE_CRITICAL && crit_run >= need_crit)
	{
		s->pressure = MEMBRANE_PRESSURE_CRITICAL;
		s->recovered_count = 0;
	}
	else if (x->measured >= MEMBRANE_PRESSURE_WARNING
		&& s->above_count >= p->sustained_samples
		&& s->pressure < MEMBRANE_PRESSURE_WARNING)
	{
		s->pressure = MEMBRANE_PRESSURE_WARNING;
		s->recovered_count = 0;
	}
	else if (s->pressure >= MEMBRANE_PRESSURE_WARNING)
	{
		/* hysteresis: leaving a pressured state needs the EXIT thresholds,
		 * not merely "I3 no longer fires" */
		if (exit_ok(p, s->pressure, x))
		{
			s->recovered_count++;
			if (s->recovered_count >= p->recovery_samples)
			{
				s->pressure = s->pressure == MEMBRANE_PRESSURE_CRITICAL
					? MEMBRANE_PRESSURE_WARNING : MEMBRANE_PRESSURE_HEALTHY;
				s->recovered_count = 0;
				if (s->pressure == MEMBRANE_PRESSURE_HEALTHY)
					s->healthy_since_ms = x->t_ms;
			}
		}
		else
			s->recovered_count = 0;
	}
	else if (x->measured == MEMBRANE_PRESSURE_HEALTHY
		&& s->pressure != MEMBRANE_PRESSURE_HEALTHY)
	{
		s->pressure = MEMBRANE_PRESSURE_HEALTHY;
		s->healthy_since_ms = x->t_ms;
	}
	return (s->pressure);
}

/* ------------------------------------------------------------------ */
/* baseline / manual-override detection                                 */
/* ------------------------------------------------------------------ */

static void	sorted_resident_sig(const membrane_apply_live_t *live, char *out,
				size_t n)
{
	size_t	order[MEMBRANE_ACTION_MAX_RESIDENT];
	size_t	i;
	size_t	j;
	size_t	len = 0;

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
	if (n > 0)
		out[0] = '\0';
	for (i = 0; i < live->resident_count && len + 2 < n; ++i)
		len += (size_t)snprintf(out + len, n - len, "%s%.100s",
				i ? "|" : "", live->resident[order[i]].name);
}

static void	baseline_add(membrane_apply_state_t *b, const char *k,
				const char *v)
{
	if (b->count >= MEMBRANE_ACTION_MAX_STATE)
		return ;
	copy(b->key[b->count], sizeof(b->key[0]), k);
	copy(b->value[b->count], sizeof(b->value[0]), v);
	b->count++;
}

static void	build_baseline(const membrane_apply_live_t *live,
				membrane_apply_state_t *b, char *sig, size_t sig_n)
{
	memset(b, 0, sizeof(*b));
	baseline_add(b, "default_model", live->default_model);
	baseline_add(b, "override.context", live->ctx_known ? live->ctx : "?");
	baseline_add(b, "override.kv_precision",
		live->kv_precision_known ? live->kv_precision : "?");
	baseline_add(b, "override.kv_placement",
		live->kv_placement_known ? live->kv_placement : "?");
	baseline_add(b, "override.gpu_layers",
		live->gpu_layers_known ? live->gpu_layers : "?");
	sorted_resident_sig(live, sig, sig_n);
	baseline_add(b, "residents", sig);
}

void	membrane_adaptive_adopt_baseline(membrane_adaptive_state_t *s,
			const membrane_apply_live_t *live)
{
	char	sig[160];

	build_baseline(live, &s->baseline, sig, sizeof(sig));
	s->baseline_valid = 1;
}

static int	numeric_dir(const char *from, const char *to)
{
	char	*e1;
	char	*e2;
	long	a = strtol(from, &e1, 10);
	long	b = strtol(to, &e2, 10);

	if (*from == '\0' || *e1 != '\0' || *to == '\0' || *e2 != '\0')
		return (0);
	return (b < a ? -1 : b > a ? 1 : 0);
}

static void	push_action(membrane_adaptive_state_t *s,
				const membrane_adaptive_action_rec_t *r)
{
	if (s->action_count >= MEMBRANE_ADAPTIVE_MAX_ACTIONS)
	{
		memmove(&s->actions[0], &s->actions[1],
			(MEMBRANE_ADAPTIVE_MAX_ACTIONS - 1) * sizeof(s->actions[0]));
		s->action_count = MEMBRANE_ADAPTIVE_MAX_ACTIONS - 1;
	}
	s->actions[s->action_count++] = *r;
}

/* Returns the drifted keys count; records manual-change history. */
static size_t	detect_manual(membrane_adaptive_state_t *s, int64_t now_ms,
					const membrane_apply_live_t *live, char *keys, size_t kn)
{
	membrane_apply_state_t	now;
	char					sig[160];
	size_t					i;
	size_t					n = 0;
	size_t					len = 0;

	build_baseline(live, &now, sig, sizeof(sig));
	if (kn > 0)
		keys[0] = '\0';
	for (i = 0; i < now.count && i < s->baseline.count; ++i)
	{
		membrane_adaptive_action_rec_t	r;

		if (strcmp(s->baseline.key[i], now.key[i]) == 0
			&& strcmp(s->baseline.value[i], now.value[i]) == 0)
			continue ;
		n++;
		if (len + 1 < kn)
			len += (size_t)snprintf(keys + len, kn - len, "%s%s",
					len ? ", " : "", now.key[i]);
		memset(&r, 0, sizeof(r));
		r.t_ms = now_ms;
		r.manual = 1;
		r.ok = 1;
		copy(r.code, sizeof(r.code), now.key[i]);
		copy(r.from, sizeof(r.from), s->baseline.value[i]);
		copy(r.to, sizeof(r.to), now.value[i]);
		r.direction = numeric_dir(r.from, r.to);
		if (strcmp(now.key[i], "residents") == 0)
			r.direction = strlen(r.to) > strlen(r.from) ? 1 : -1;
		push_action(s, &r);
	}
	return (n);
}

/* ------------------------------------------------------------------ */
/* decision helpers                                                     */
/* ------------------------------------------------------------------ */

#if defined(__GNUC__) || defined(__clang__)
# define ADAPTIVE_PRINTF_FMT(a, b)	__attribute__((format(printf, a, b)))
#else
# define ADAPTIVE_PRINTF_FMT(a, b)
#endif
static void	chain_add(membrane_adaptive_decision_t *d, const char *stage,
				const char *fmt, ...) ADAPTIVE_PRINTF_FMT(3, 4);

static void	chain_add(membrane_adaptive_decision_t *d, const char *stage,
				const char *fmt, ...)
{
	va_list	ap;

	if (d->chain_count >= MEMBRANE_ADAPTIVE_MAX_CHAIN)
		return ;
	copy(d->chain[d->chain_count].stage, sizeof(d->chain[0].stage), stage);
	va_start(ap, fmt);
	vsnprintf(d->chain[d->chain_count].text, sizeof(d->chain[0].text), fmt, ap);
	va_end(ap);
	d->chain_count++;
}

static void	set_kind(membrane_adaptive_decision_t *d,
				membrane_adaptive_decision_kind_t k, const char *reason)
{
	d->kind = k;
	copy(d->reason, sizeof(d->reason), reason);
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

static size_t	count_in_window(const membrane_adaptive_policy_t *p,
					const membrane_adaptive_state_t *s, int64_t now_ms)
{
	size_t	i;
	size_t	n = 0;

	for (i = 0; i < s->action_count; ++i)
		if (!s->actions[i].manual
			&& now_ms - s->actions[i].t_ms <= (int64_t)p->window_s * 1000)
			n++;
	return (n);
}

/* Would taking `dir` on (code, model) make the recent sequence flip
 * direction `thrash_reversals` times? Manual changes count too: an
 * adaptive step undoing a user's change (or the reverse) is thrash. */
static int	would_thrash(const membrane_adaptive_policy_t *p,
				const membrane_adaptive_state_t *s, int64_t now_ms,
				const char *code, const char *model, int dir)
{
	int		last = 0;
	int		rev = 0;
	size_t	i;

	for (i = 0; i < s->action_count; ++i)
	{
		const membrane_adaptive_action_rec_t	*a = &s->actions[i];

		if (a->direction == 0 || strcmp(a->code, code) != 0
			|| strcmp(a->model, model) != 0
			|| now_ms - a->t_ms > (int64_t)p->thrash_window_s * 1000)
			continue ;
		if (last != 0 && a->direction != last)
			rev++;
		last = a->direction;
	}
	if (last != 0 && dir != 0 && dir != last)
		rev++;
	return (rev >= p->thrash_reversals);
}

static int	surface_ok(const membrane_apply_surface_t *sf,
				membrane_action_code_t c)
{
	return (sf != NULL && sf->state[c] == MEMBRANE_CAPABILITY_SUPPORTED);
}

static void	fill_set(membrane_adaptive_decision_t *d, int cls,
				membrane_action_code_t code, const char *value, int reload)
{
	d->action_class = cls;
	d->needs_reload = reload;
	switch (code)
	{
		case MEMBRANE_ACTION_SET_CONTEXT:
			copy(d->request.context, sizeof(d->request.context), value);
			break ;
		case MEMBRANE_ACTION_SET_KV_PRECISION:
			copy(d->request.kv_precision, sizeof(d->request.kv_precision),
				value);
			break ;
		case MEMBRANE_ACTION_SET_KV_PLACEMENT:
			copy(d->request.kv_placement, sizeof(d->request.kv_placement),
				value);
			break ;
		default:
			copy(d->request.gpu_layers, sizeof(d->request.gpu_layers), value);
			break ;
	}
}

/* ------------------------------------------------------------------ */
/* decide                                                               */
/* ------------------------------------------------------------------ */

typedef struct s_step
{
	int						viable;
	int						cls;
	membrane_action_code_t	code;
	char					value[MEMBRANE_ACTION_VALUE_MAX];
	char					model[MEMBRANE_ACTION_NAME_MAX];
	int						reload;
	int						dir;
	char					from[MEMBRANE_ACTION_VALUE_MAX];
	char					text[MEMBRANE_ADAPTIVE_TEXT_MAX];
}	step_t;

static const char	*kv_current(const membrane_apply_live_t *live,
						const membrane_observation_snapshot_t *snap)
{
	if (live->kv_precision_known && strcmp(live->kv_precision, "automatic"))
		return (live->kv_precision);
	if (snap != NULL && snap->kv_planned_precision.known)
		return (snap->kv_planned_precision.value);
	return ("");
}

void	membrane_adaptive_decide(const membrane_adaptive_policy_t *p,
			membrane_adaptive_state_t *s, int64_t now_ms,
			const membrane_observation_snapshot_t *snap,
			const membrane_memory_intelligence_t *intel,
			const membrane_apply_live_t *live,
			const membrane_apply_surface_t *surface,
			membrane_adaptive_decision_t *out)
{
	membrane_adaptive_mode_t	mode = membrane_adaptive_effective_mode(p);
	const membrane_adaptive_sample_t	*last = s->sample_count
		? &s->samples[s->sample_count - 1] : NULL;
	char	keys[160];
	step_t	step;
	int		i;

	memset(out, 0, sizeof(*out));
	out->pressure = s->pressure;
	if (mode == MEMBRANE_ADAPTIVE_OFF)
	{
		set_kind(out, MEMBRANE_DECISION_NONE, "");
		return ;
	}
	if (last != NULL)
	{
		char	ram[24];
		char	vram[24];

		if (last->ram_permille < 0)
			snprintf(ram, sizeof(ram), "unknown");
		else
			snprintf(ram, sizeof(ram), "%d.%d%%", last->ram_permille / 10,
				last->ram_permille % 10);
		if (last->vram_permille < 0)
			snprintf(vram, sizeof(vram), "unknown");
		else
			snprintf(vram, sizeof(vram), "%d.%d%%", last->vram_permille / 10,
				last->vram_permille % 10);
		chain_add(out, "observed", "pressure %s (measured); free RAM %s, "
			"free VRAM %s", membrane_pressure_name(s->pressure), ram, vram);
	}
	if (last == NULL || !last->valid)
	{
		set_kind(out, MEMBRANE_DECISION_NONE,
			MEMBRANE_ADAPT_REASON_UNOBSERVABLE);
		return ;
	}
	/* manual override: adopt the user's state, never undo it */
	if (!s->baseline_valid)
		membrane_adaptive_adopt_baseline(s, live);
	else if (detect_manual(s, now_ms, live, keys, sizeof(keys)) > 0)
	{
		membrane_adaptive_adopt_baseline(s, live);
		s->last_action_ms = now_ms;		/* a quiet period, not an undo */
		chain_add(out, "policy", "changed outside adaptive control (%s): "
			"baseline adopted, nothing undone", keys);
		set_kind(out, MEMBRANE_DECISION_MANUAL_OVERRIDE,
			MEMBRANE_ADAPT_REASON_MANUAL);
		return ;
	}
	if (mode == MEMBRANE_ADAPTIVE_OBSERVE)
	{
		set_kind(out, MEMBRANE_DECISION_OBSERVED, "");
		return ;
	}
	if (s->suppressed_until_ms > now_ms)
	{
		chain_add(out, "policy", "suppressed for another %lld s (%s)",
			(long long)((s->suppressed_until_ms - now_ms + 999) / 1000),
			s->suppress_reason);
		set_kind(out, MEMBRANE_DECISION_SUPPRESSED, s->suppress_reason);
		out->cooldown_remaining_s = (s->suppressed_until_ms - now_ms + 999)
			/ 1000;
		return ;
	}
	/* pending reload clears when the resident set changed or emptied */
	if (s->pending_reload)
	{
		char	sig[160];
		membrane_apply_state_t	tmp;

		build_baseline(live, &tmp, sig, sizeof(sig));
		if (live->resident_count == 0 || strcmp(sig, s->resident_sig) != 0)
			s->pending_reload = 0;
	}

	/* ---- pressure not (yet) sustained ------------------------------ */
	if (s->pressure < MEMBRANE_PRESSURE_WARNING
		&& last->measured >= MEMBRANE_PRESSURE_WARNING)
	{
		chain_add(out, "policy", "pressure seen in %d of %d samples needed "
			"to count as sustained", s->above_count, p->sustained_samples);
		set_kind(out, MEMBRANE_DECISION_TRANSIENT,
			MEMBRANE_ADAPT_REASON_TRANSIENT);
		return ;
	}
	/* ---- healthy: nothing to adapt (maybe restore) ---------------- */
	if (s->pressure < MEMBRANE_PRESSURE_WARNING)
	{
		const membrane_adaptive_action_rec_t	*r = NULL;
		size_t									k;

		set_kind(out, MEMBRANE_DECISION_NONE, "");
		if (!p->auto_restore || s->pressure != MEMBRANE_PRESSURE_HEALTHY
			|| now_ms - s->healthy_since_ms < (int64_t)p->restore_after_s * 1000
			|| (s->last_action_ms >= 0 && now_ms - s->last_action_ms
				< (int64_t)p->cooldown_s * 1000))
			return ;
		for (k = s->action_count; k-- > 0; )
			if (!s->actions[k].manual && s->actions[k].ok
				&& s->actions[k].direction < 0)
			{
				r = &s->actions[k];
				break ;
			}
		if (r == NULL || (strcmp(r->code, "SET_CONTEXT") != 0
				&& strcmp(r->code, "SET_KV_PRECISION") != 0))
			return ;
		if (strcmp(r->code, "SET_CONTEXT") == 0
			&& strcmp(live->ctx, r->to) != 0)
			return ;			/* the user changed it since: leave alone */
		if (strcmp(r->code, "SET_KV_PRECISION") == 0
			&& strcmp(live->kv_precision, r->to) != 0)
			return ;
		if (would_thrash(p, s, now_ms, r->code, r->model, 1))
		{
			s->suppressed_until_ms = now_ms + (int64_t)p->suppress_s * 1000;
			copy(s->suppress_reason, sizeof(s->suppress_reason),
				MEMBRANE_ADAPT_REASON_THRASH);
			set_kind(out, MEMBRANE_DECISION_SUPPRESSED,
				MEMBRANE_ADAPT_REASON_THRASH);
			chain_add(out, "policy", "restoring %s would reverse direction "
				"too often: adaptation suppressed", r->code);
			return ;
		}
		out->is_restore = 1;
		fill_set(out, strcmp(r->code, "SET_CONTEXT") == 0
			? MEMBRANE_ADAPTIVE_CLASS_CONTEXT
			: MEMBRANE_ADAPTIVE_CLASS_KV_PRECISION,
			strcmp(r->code, "SET_CONTEXT") == 0 ? MEMBRANE_ACTION_SET_CONTEXT
			: MEMBRANE_ACTION_SET_KV_PRECISION, r->from, r->needs_reload);
		snprintf(out->step_text, sizeof(out->step_text), "restore %s: %s -> "
			"%s (pressure cleared for %d s)", r->code, r->to, r->from,
			p->restore_after_s);
		chain_add(out, "policy", "pressure healthy since the exit threshold "
			"was met; auto_restore is on");
		chain_add(out, "action", "%s", out->step_text);
		set_kind(out, mode == MEMBRANE_ADAPTIVE_APPLY
			? MEMBRANE_DECISION_ACT : MEMBRANE_DECISION_RECOMMEND, "");
		return ;
	}

	/* ---- pressured ------------------------------------------------ */
	if (s->above_count < p->sustained_samples
		&& s->pressure < MEMBRANE_PRESSURE_WARNING)
	{
		set_kind(out, MEMBRANE_DECISION_TRANSIENT,
			MEMBRANE_ADAPT_REASON_TRANSIENT);
		return ;
	}
	if (s->pressure < p->min_pressure)
	{
		set_kind(out, MEMBRANE_DECISION_NONE,
			MEMBRANE_ADAPT_REASON_BELOW_MIN);
		return ;
	}
	chain_add(out, "finding", "sustained %s memory pressure from measured "
		"findings (%d consecutive samples)", membrane_pressure_name(
			s->pressure), s->above_count);
	if (last->measured < MEMBRANE_PRESSURE_WARNING && last->estimate_only
		&& !p->allow_estimate_only)
	{
		set_kind(out, MEMBRANE_DECISION_RECOMMEND,
			MEMBRANE_ADAPT_REASON_ESTIMATE);
		return ;
	}
	if (s->last_action_ms >= 0 && now_ms - s->last_action_ms
		< (int64_t)p->cooldown_s * 1000)
	{
		out->cooldown_remaining_s = ((int64_t)p->cooldown_s * 1000
				- (now_ms - s->last_action_ms) + 999) / 1000;
		chain_add(out, "policy", "cooldown: %lld s remaining",
			(long long)out->cooldown_remaining_s);
		set_kind(out, MEMBRANE_DECISION_COOLDOWN,
			MEMBRANE_ADAPT_REASON_COOLDOWN);
		return ;
	}
	if ((int)count_in_window(p, s, now_ms) >= p->max_actions_per_window)
	{
		chain_add(out, "policy", "at most %d adaptive actions per %d s",
			p->max_actions_per_window, p->window_s);
		set_kind(out, MEMBRANE_DECISION_RATE_LIMITED,
			MEMBRANE_ADAPT_REASON_RATE);
		return ;
	}

	/* ---- the least-disruptive-first hierarchy --------------------- */
	memset(&step, 0, sizeof(step));
	for (i = 1; i <= 5 && !step.viable; ++i)
	{
		step_t	c;

		memset(&c, 0, sizeof(c));
		if (i == 1)
		{
			const char	*cur = live->ctx_known && all_digits(live->ctx)
				? live->ctx : "";
			unsigned long long	n = 0;

			c.cls = MEMBRANE_ADAPTIVE_CLASS_CONTEXT;
			c.code = MEMBRANE_ACTION_SET_CONTEXT;
			c.dir = -1;
			if (cur[0] != '\0')
				n = strtoull(cur, NULL, 10);
			else if (snap != NULL && snap->context_planned.known)
				n = snap->context_planned.value;
			if (n == 0)
				chain_add(out, "policy", "context: current value unknown, "
					"skipped");
			else
			{
				unsigned long long	t = (n / 2) / 256 * 256;

				if (t < MEMBRANE_ADAPTIVE_MIN_CONTEXT)
					chain_add(out, "policy", "context %llu is already near "
						"the %d-token floor, skipped", n,
						MEMBRANE_ADAPTIVE_MIN_CONTEXT);
				else
				{
					snprintf(c.value, sizeof(c.value), "%llu", t);
					snprintf(c.from, sizeof(c.from), "%s",
						cur[0] ? cur : "automatic");
					snprintf(c.text, sizeof(c.text), "SET_CONTEXT %llu -> "
						"%llu (halve, applies to the next request)", n, t);
					c.viable = 1;
				}
			}
		}
		else if (i == 2)
		{
			const char	*cur = kv_current(live, snap);

			c.cls = MEMBRANE_ADAPTIVE_CLASS_KV_PRECISION;
			c.code = MEMBRANE_ACTION_SET_KV_PRECISION;
			c.reload = 1;
			c.dir = -1;
			if (strcmp(cur, "native") == 0 || strcmp(cur, "q8") == 0)
			{
				copy(c.value, sizeof(c.value),
					strcmp(cur, "native") == 0 ? "q8" : "q5");
				copy(c.from, sizeof(c.from), live->kv_precision_known
					? live->kv_precision : "automatic");
				snprintf(c.text, sizeof(c.text), "SET_KV_PRECISION %s -> %s "
					"(one step, applies at the next model load)", cur,
					c.value);
				c.viable = 1;
			}
			else
				chain_add(out, "policy", "kv precision: %s, no lower step "
					"known, skipped", cur[0] ? cur : "unknown");
		}
		else if (i == 3)
		{
			c.cls = MEMBRANE_ADAPTIVE_CLASS_KV_PLACEMENT;
			c.code = MEMBRANE_ACTION_SET_KV_PLACEMENT;
			c.reload = 1;
			c.dir = -1;
			if (last->vram_pressured && strcmp(live->kv_placement, "cpu"))
			{
				copy(c.value, sizeof(c.value), "cpu");
				copy(c.from, sizeof(c.from), live->kv_placement_known
					? live->kv_placement : "automatic");
				snprintf(c.text, sizeof(c.text), "SET_KV_PLACEMENT -> cpu "
					"(measured VRAM pressure, applies at the next model "
					"load)");
				c.viable = 1;
			}
			else
				chain_add(out, "policy", "kv placement: not under measured "
					"VRAM pressure (or already cpu), skipped");
		}
		else if (i == 4)
		{
			int	cur = -1;
			size_t	k;

			c.cls = MEMBRANE_ADAPTIVE_CLASS_GPU_LAYERS;
			c.code = MEMBRANE_ACTION_SET_GPU_LAYERS;
			c.reload = 1;
			c.dir = -1;
			if (live->gpu_layers_known && all_digits(live->gpu_layers))
				cur = atoi(live->gpu_layers);
			for (k = 0; cur < 0 && k < live->resident_count; ++k)
				if (live->resident[k].gpu_layers_known
					&& live->resident[k].gpu_layers > 0)
					cur = live->resident[k].gpu_layers;
			if (last->vram_pressured && cur > 0)
			{
				int	t = cur * 3 / 4;

				if (t >= cur)
					t = cur - 1;
				snprintf(c.value, sizeof(c.value), "%d", t);
				snprintf(c.from, sizeof(c.from), "%s", live->gpu_layers_known
					? live->gpu_layers : "automatic");
				snprintf(c.text, sizeof(c.text), "SET_GPU_LAYERS %d -> %d "
					"(-25%%, measured VRAM pressure, applies at the next "
					"model load)", cur, t);
				c.viable = 1;
			}
			else
				chain_add(out, "policy", "gpu layers: no VRAM pressure or no "
					"offloaded layers known, skipped");
		}
		else
		{
			char	why[MEMBRANE_ACTION_TEXT_MAX] = "";
			int		idx = membrane_action_pick_unload_candidate(live, why,
					sizeof(why));

			c.cls = MEMBRANE_ADAPTIVE_CLASS_UNLOAD;
			c.code = MEMBRANE_ACTION_UNLOAD_MODEL;
			c.dir = -1;
			if (idx >= 0)
			{
				copy(c.model, sizeof(c.model), live->resident[idx].name);
				copy(c.value, sizeof(c.value), "unloaded");
				copy(c.from, sizeof(c.from), "resident");
				snprintf(c.text, sizeof(c.text), "UNLOAD_MODEL %.100s (idle, "
					"unpinned, non-default)", c.model);
				c.viable = 1;
			}
			else
				chain_add(out, "policy", "unload: %.160s", why);
		}
		if (!c.viable)
			continue ;
		if (!(p->allowed_classes & c.cls))
		{
			chain_add(out, "policy", "%.150s: class not allowed by policy",
				c.text);
			continue ;
		}
		if (!surface_ok(surface, c.code))
		{
			chain_add(out, "policy", "%.150s: runtime cannot control it",
				c.text);
			continue ;
		}
		if (c.reload || c.code == MEMBRANE_ACTION_UNLOAD_MODEL)
			if (s->pending_reload)
			{
				chain_add(out, "policy", "an earlier change is still "
					"waiting for the model to be reloaded: not escalating");
				set_kind(out, MEMBRANE_DECISION_DEFERRED,
					MEMBRANE_ADAPT_REASON_PENDING);
				return ;
			}
		step = c;
	}
	if (!step.viable)
	{
		chain_add(out, "policy", "no allowed, supported, non-destructive "
			"step remains");
		set_kind(out, MEMBRANE_DECISION_NO_SAFE_STEP,
			MEMBRANE_ADAPT_REASON_NO_STEP);
		return ;
	}
	if (would_thrash(p, s, now_ms, membrane_action_code_name(step.code),
			step.model, step.dir))
	{
		s->suppressed_until_ms = now_ms + (int64_t)p->suppress_s * 1000;
		copy(s->suppress_reason, sizeof(s->suppress_reason),
			MEMBRANE_ADAPT_REASON_THRASH);
		chain_add(out, "policy", "%s keeps reversing direction: adaptation "
			"suppressed for %d s", membrane_action_code_name(step.code),
			p->suppress_s);
		set_kind(out, MEMBRANE_DECISION_SUPPRESSED,
			MEMBRANE_ADAPT_REASON_THRASH);
		out->cooldown_remaining_s = p->suppress_s;
		return ;
	}
	if (intel != NULL)
	{
		size_t	k;

		for (k = 0; k < intel->recommendation_count; ++k)
			if (intel->recommendations[k].applicability
				!= MEMBRANE_INTEL_APPLICABILITY_ADVISORY_ONLY)
			{
				chain_add(out, "recommendation", "%s",
					intel->recommendations[k].code);
				break ;
			}
	}
	out->action_class = step.cls;
	out->needs_reload = step.reload;
	switch (step.code)
	{
		case MEMBRANE_ACTION_UNLOAD_MODEL:
			copy(out->request.unload, sizeof(out->request.unload), step.model);
			break ;
		default:
			fill_set(out, step.cls, step.code, step.value, step.reload);
			break ;
	}
	copy(out->step_text, sizeof(out->step_text), step.text);
	chain_add(out, "policy", "mode %s, class allowed, runtime supports it, "
		"cooldown/rate/thrash clear", membrane_adaptive_mode_name(mode));
	chain_add(out, "action", "%s", step.text);
	set_kind(out, mode == MEMBRANE_ADAPTIVE_APPLY ? MEMBRANE_DECISION_ACT
		: MEMBRANE_DECISION_RECOMMEND, "");
	(void)step.from;
}

void	membrane_adaptive_note_action(const membrane_adaptive_policy_t *p,
			membrane_adaptive_state_t *s, int64_t now_ms,
			const membrane_adaptive_decision_t *d, int ok,
			const membrane_apply_live_t *live_after)
{
	membrane_adaptive_action_rec_t	r;
	const char						*code = "";

	(void)p;
	memset(&r, 0, sizeof(r));
	r.t_ms = now_ms;
	r.ok = ok;
	r.needs_reload = d->needs_reload;
	r.direction = d->is_restore ? 1 : -1;
	if (d->request.context[0])
		code = "SET_CONTEXT";
	else if (d->request.kv_precision[0])
		code = "SET_KV_PRECISION";
	else if (d->request.kv_placement[0])
		code = "SET_KV_PLACEMENT";
	else if (d->request.gpu_layers[0])
		code = "SET_GPU_LAYERS";
	else if (d->request.unload[0])
	{
		code = "UNLOAD_MODEL";
		copy(r.model, sizeof(r.model), d->request.unload);
	}
	copy(r.code, sizeof(r.code), code);
	/* from/to are filled from the decision text's request values */
	copy(r.to, sizeof(r.to), d->request.context[0] ? d->request.context
		: d->request.kv_precision[0] ? d->request.kv_precision
		: d->request.kv_placement[0] ? d->request.kv_placement
		: d->request.gpu_layers[0] ? d->request.gpu_layers : "unloaded");
	/* the previous value is what the baseline held for that override */
	{
		size_t	i;
		const char	*key = d->request.context[0] ? "override.context"
			: d->request.kv_precision[0] ? "override.kv_precision"
			: d->request.kv_placement[0] ? "override.kv_placement"
			: d->request.gpu_layers[0] ? "override.gpu_layers" : "";

		for (i = 0; key[0] && i < s->baseline.count; ++i)
			if (strcmp(s->baseline.key[i], key) == 0)
				copy(r.from, sizeof(r.from), s->baseline.value[i]);
		if (d->request.unload[0])
			copy(r.from, sizeof(r.from), "resident");
	}
	push_action(s, &r);
	s->last_action_ms = now_ms;
	if (ok && d->needs_reload && live_after != NULL)
	{
		membrane_apply_state_t	tmp;

		s->pending_reload = 1;
		build_baseline(live_after, &tmp, s->resident_sig,
			sizeof(s->resident_sig));
	}
	if (live_after != NULL)
		membrane_adaptive_adopt_baseline(s, live_after);
}
