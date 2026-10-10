#include <string.h>

#include "adaptive.h"
#include "test_helpers.h"

/*
 * Milestone K: adaptive.h/.c tests. Synthetic timelines with injected
 * timestamps (no sleeps, no host). The intelligence is the REAL I3
 * assembler fed synthetic snapshots, so "when is there pressure" is
 * exactly I3's threshold policy -- these tests only add the hysteresis /
 * cooldown / rate / thrash behavior on top.
 */

#define GIB	(1024ull * 1024 * 1024)
#define STEP_MS	30000

typedef struct s_rig
{
	membrane_adaptive_policy_t	p;
	membrane_adaptive_state_t	s;
	membrane_apply_surface_t	sf;
	membrane_apply_live_t		lv;
	int64_t						now;
	membrane_observation_snapshot_t	snap;
	membrane_memory_intelligence_t	intel;
}	rig_t;

static void	rig_init(rig_t *r, membrane_adaptive_mode_t mode)
{
	int	i;

	memset(r, 0, sizeof(*r));
	membrane_adaptive_policy_defaults(&r->p);
	r->p.enabled = 1;
	r->p.mode = mode;
	r->now = 1000000;
	membrane_adaptive_state_init(&r->s, r->now);
	for (i = 0; i < MEMBRANE_ACTION_CODE_COUNT; ++i)
	{
		r->sf.state[i] = MEMBRANE_CAPABILITY_SUPPORTED;
		r->sf.reversible[i] = 1;
	}
	r->lv.runtime_reachable = 1;
	r->lv.ctx_known = r->lv.kv_precision_known = r->lv.kv_placement_known
		= r->lv.gpu_layers_known = 1;
	snprintf(r->lv.ctx, sizeof(r->lv.ctx), "automatic");
	snprintf(r->lv.kv_precision, sizeof(r->lv.kv_precision), "automatic");
	snprintf(r->lv.kv_placement, sizeof(r->lv.kv_placement), "automatic");
	snprintf(r->lv.gpu_layers, sizeof(r->lv.gpu_layers), "automatic");
	snprintf(r->lv.default_model, sizeof(r->lv.default_model), "main");
}

static void	add_res(rig_t *r, const char *name, int gen, int pinned,
				uint64_t size, int layers)
{
	membrane_apply_resident_t	*x = &r->lv.resident[r->lv.resident_count++];

	snprintf(x->name, sizeof(x->name), "%s", name);
	x->activity_known = 1;
	x->generating = gen;
	x->pinned = pinned;
	x->size_bytes = size;
	x->gpu_layers_known = layers > 0;
	x->gpu_layers = layers;
}

static void	make_snap(membrane_observation_snapshot_t *s, int ram_pm,
				int vram_pm)
{
	membrane_obs_snapshot_init(s, MEMBRANE_RUNTIME_ID_NATIVE);
	s->runtime_observable = 1;
	membrane_obs_str_set(&s->runtime_availability, "available",
		MEMBRANE_OBS_PROV_STATIC_METADATA, "t");
	membrane_obs_u64_set(&s->ram_total_bytes, 8 * GIB,
		MEMBRANE_OBS_PROV_MEASURED, "t");
	membrane_obs_u64_set(&s->ram_available_bytes, (8 * GIB * (uint64_t)ram_pm + 999) / 1000,
		MEMBRANE_OBS_PROV_MEASURED, "t");
	membrane_obs_u64_set(&s->gpu_device_count, 1, MEMBRANE_OBS_PROV_MEASURED,
		"t");
	membrane_obs_u64_set(&s->vram_total_bytes, 8 * GIB,
		MEMBRANE_OBS_PROV_MEASURED, "t");
	membrane_obs_u64_set(&s->vram_free_bytes, (8 * GIB * (uint64_t)vram_pm + 999) / 1000,
		MEMBRANE_OBS_PROV_MEASURED, "t");
	membrane_obs_u64_set(&s->context_planned, 8192,
		MEMBRANE_OBS_PROV_ESTIMATED, "planner_v2");
	membrane_obs_u64_set(&s->context_model_max, 32768,
		MEMBRANE_OBS_PROV_STATIC_METADATA, "gguf");
	membrane_obs_str_set(&s->kv_planned_precision, "native",
		MEMBRANE_OBS_PROV_ESTIMATED, "planner_v2");
	membrane_obs_u64_set(&s->resident_model_count, 1,
		MEMBRANE_OBS_PROV_RUNTIME_REPORTED, "t");
	membrane_obs_snapshot_finalize(s);
}

/* One full controller tick: observe -> sample -> record -> decide. */
static membrane_adaptive_decision_t	tick(rig_t *r, int ram_pm, int vram_pm)
{
	membrane_adaptive_sample_t		smp;
	membrane_adaptive_decision_t	d;
	membrane_runtime_capabilities_t	caps;

	memset(&caps, 0, sizeof(caps));
	make_snap(&r->snap, ram_pm, vram_pm);
	membrane_memory_intelligence_assemble(&r->snap, &caps, &r->intel);
	r->now += STEP_MS;
	membrane_adaptive_make_sample(&r->snap, &r->intel, &r->lv, r->now, &smp);
	membrane_adaptive_record_sample(&r->p, &r->s, &smp);
	membrane_adaptive_decide(&r->p, &r->s, r->now, &r->snap, &r->intel,
		&r->lv, &r->sf, &d);
	return (d);
}

/* Pretend the controller executed `d` successfully (what the driver does
 * after membrane_apply returns APPLIED), including the live-state change. */
static void	did(rig_t *r, const membrane_adaptive_decision_t *d)
{
	if (d->request.context[0])
		snprintf(r->lv.ctx, sizeof(r->lv.ctx), "%s", d->request.context);
	if (d->request.kv_precision[0])
		snprintf(r->lv.kv_precision, sizeof(r->lv.kv_precision), "%s",
			d->request.kv_precision);
	if (d->request.kv_placement[0])
		snprintf(r->lv.kv_placement, sizeof(r->lv.kv_placement), "%s",
			d->request.kv_placement);
	if (d->request.gpu_layers[0])
		snprintf(r->lv.gpu_layers, sizeof(r->lv.gpu_layers), "%s",
			d->request.gpu_layers);
	if (d->request.unload[0])
	{
		size_t	i;

		for (i = 0; i < r->lv.resident_count; ++i)
			if (!strcmp(r->lv.resident[i].name, d->request.unload))
			{
				r->lv.resident[i] = r->lv.resident[--r->lv.resident_count];
				break ;
			}
	}
	membrane_adaptive_note_action(&r->p, &r->s, r->now, d, 1, &r->lv);
}

#define WARN_PM		100		/* I3 warning: ratio < 0.15 */
#define CRIT_PM		30		/* I3 critical: ratio < 0.05 */
#define HEALTHY_PM	700

/* Drives samples until a decision other than TRANSIENT/NONE appears. */
static membrane_adaptive_decision_t	pressure_until_decision(rig_t *r, int pm,
										int max_ticks)
{
	membrane_adaptive_decision_t	d;
	int								i;

	memset(&d, 0, sizeof(d));
	for (i = 0; i < max_ticks; ++i)
	{
		d = tick(r, pm, 900);
		if (d.kind != MEMBRANE_DECISION_TRANSIENT
			&& d.kind != MEMBRANE_DECISION_NONE)
			break ;
	}
	return (d);
}

/* --- policy ---------------------------------------------------------- */

static void	test_defaults_off_and_conservative(void)
{
	membrane_adaptive_policy_t	p;

	membrane_adaptive_policy_defaults(&p);
	TEST_ASSERT(!p.enabled && p.mode == MEMBRANE_ADAPTIVE_OFF, "OFF by default");
	TEST_ASSERT(membrane_adaptive_effective_mode(&p) == MEMBRANE_ADAPTIVE_OFF,
		"effective mode off");
	p.mode = MEMBRANE_ADAPTIVE_APPLY;
	TEST_ASSERT(membrane_adaptive_effective_mode(&p) == MEMBRANE_ADAPTIVE_OFF,
		"a stray mode=apply without enabled is still off");
	TEST_ASSERT(!strcmp(membrane_adaptive_policy_validate(&p), ""),
		"defaults validate");
	TEST_ASSERT(p.allowed_classes == (MEMBRANE_ADAPTIVE_CLASS_CONTEXT
			| MEMBRANE_ADAPTIVE_CLASS_KV_PRECISION)
		&& !(p.allowed_classes & MEMBRANE_ADAPTIVE_CLASS_UNLOAD),
		"default classes never unload");
	TEST_ASSERT(p.cooldown_s >= 300 && p.interval_s >= 30
		&& p.max_actions_per_window <= 2 && !p.auto_restore
		&& !p.allow_estimate_only, "conservative defaults");
}

static void	test_policy_validation(void)
{
	membrane_adaptive_policy_t	p;

	membrane_adaptive_policy_defaults(&p);
	p.interval_s = 1;
	TEST_ASSERT(*membrane_adaptive_policy_validate(&p), "interval floor");
	membrane_adaptive_policy_defaults(&p);
	p.warning_exit_permille = 150;
	TEST_ASSERT(*membrane_adaptive_policy_validate(&p), "exit must exceed "
		"the enter threshold (no hysteresis otherwise)");
	membrane_adaptive_policy_defaults(&p);
	p.critical_exit_permille = 40;
	TEST_ASSERT(*membrane_adaptive_policy_validate(&p), "critical exit");
	membrane_adaptive_policy_defaults(&p);
	p.history_max_samples = 9999;
	TEST_ASSERT(*membrane_adaptive_policy_validate(&p), "history bound");
	membrane_adaptive_policy_defaults(&p);
	p.min_pressure = MEMBRANE_PRESSURE_HEALTHY;
	TEST_ASSERT(*membrane_adaptive_policy_validate(&p), "min pressure");
	TEST_ASSERT(membrane_adaptive_classes_parse("context,kv_precision")
			== (MEMBRANE_ADAPTIVE_CLASS_CONTEXT
				| MEMBRANE_ADAPTIVE_CLASS_KV_PRECISION)
		&& membrane_adaptive_classes_parse("context,bogus") == -1
		&& membrane_adaptive_classes_parse("") == -1, "class parsing");
	{
		char	buf[128];

		membrane_adaptive_classes_format(MEMBRANE_ADAPTIVE_CLASS_ALL, buf,
			sizeof(buf));
		TEST_ASSERT(!strcmp(buf, "context,kv_precision,kv_placement,"
				"gpu_layers,unload"), "class formatting is stable");
	}
}

/* --- modes ----------------------------------------------------------- */

static void	test_modes_never_mutate_below_apply(void)
{
	membrane_adaptive_mode_t		modes[3] = {MEMBRANE_ADAPTIVE_OFF,
		MEMBRANE_ADAPTIVE_OBSERVE, MEMBRANE_ADAPTIVE_RECOMMEND};
	int								m;
	int								i;
	rig_t							*r = malloc(sizeof(*r));

	for (m = 0; m < 3; ++m)
	{
		rig_init(r, modes[m]);
		for (i = 0; i < 12; ++i)
		{
			membrane_adaptive_decision_t	d = tick(r, WARN_PM, 900);

			TEST_ASSERT(d.kind != MEMBRANE_DECISION_ACT, "mode below apply "
				"never ACTs");
			if (modes[m] == MEMBRANE_ADAPTIVE_OFF)
				TEST_ASSERT(d.kind == MEMBRANE_DECISION_NONE, "off: nothing");
			if (modes[m] == MEMBRANE_ADAPTIVE_OBSERVE)
				TEST_ASSERT(d.kind == MEMBRANE_DECISION_OBSERVED
					&& !d.request.context[0], "observe: samples only");
		}
		if (modes[m] == MEMBRANE_ADAPTIVE_RECOMMEND)
		{
			membrane_adaptive_decision_t	d = pressure_until_decision(r,
					WARN_PM, 3);

			TEST_ASSERT(d.kind == MEMBRANE_DECISION_RECOMMEND
				&& d.request.context[0], "recommend: a concrete step, no act");
		}
	}
	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	r->p.enabled = 0;
	for (i = 0; i < 8; ++i)
		TEST_ASSERT(tick(r, CRIT_PM, 900).kind == MEMBRANE_DECISION_NONE,
			"apply mode but not enabled: nothing");
	free(r);
}

/* --- sustained vs transient ------------------------------------------ */

static void	test_transient_vs_sustained(void)
{
	rig_t							*r = malloc(sizeof(*r));
	membrane_adaptive_decision_t	d;

	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	d = tick(r, WARN_PM, 900);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_TRANSIENT
		&& !strcmp(d.reason, MEMBRANE_ADAPT_REASON_TRANSIENT), "1st sample");
	d = tick(r, WARN_PM, 900);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_TRANSIENT, "2nd sample");
	d = tick(r, HEALTHY_PM, 900);			/* a blip that clears */
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_NONE
		&& r->s.pressure == MEMBRANE_PRESSURE_HEALTHY, "blip never acted");
	d = tick(r, WARN_PM, 900);
	d = tick(r, WARN_PM, 900);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_TRANSIENT, "run restarted");
	d = tick(r, WARN_PM, 900);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_ACT
		&& r->s.pressure == MEMBRANE_PRESSURE_WARNING, "3rd consecutive => "
			"sustained => act");
	free(r);
}

/* --- hysteresis ------------------------------------------------------- */

static void	test_hysteresis_boundaries(void)
{
	rig_t	*r = malloc(sizeof(*r));
	int		i;

	rig_init(r, MEMBRANE_ADAPTIVE_RECOMMEND);
	for (i = 0; i < 3; ++i)
		tick(r, WARN_PM, 900);
	TEST_ASSERT(r->s.pressure == MEMBRANE_PRESSURE_WARNING, "entered");
	/* I3 stops firing at >= 150 permille, but the EXIT threshold is 200:
	 * 150..199 must NOT recover. */
	for (i = 0; i < 10; ++i)
		tick(r, 199, 900);
	TEST_ASSERT(r->s.pressure == MEMBRANE_PRESSURE_WARNING,
		"199 permille (healthy per I3, below exit) stays WARNING");
	/* exactly the exit threshold: recovery_samples consecutive */
	tick(r, 200, 900);
	tick(r, 200, 900);
	TEST_ASSERT(r->s.pressure == MEMBRANE_PRESSURE_WARNING, "needs 3");
	tick(r, 199, 900);						/* a dip resets the run */
	tick(r, 200, 900);
	tick(r, 200, 900);
	TEST_ASSERT(r->s.pressure == MEMBRANE_PRESSURE_WARNING, "run was reset");
	tick(r, 200, 900);
	TEST_ASSERT(r->s.pressure == MEMBRANE_PRESSURE_HEALTHY, "exactly 200 x3 "
		"recovers");
	/* re-entering needs sustained pressure again, not one dip */
	tick(r, 140, 900);
	TEST_ASSERT(r->s.pressure == MEMBRANE_PRESSURE_HEALTHY, "no flapping");
	/* host RAM also needs the absolute floor to recover */
	rig_init(r, MEMBRANE_ADAPTIVE_RECOMMEND);
	r->p.ram_exit_abs_bytes = 3 * GIB;		/* 8 GiB * 0.3 < 3 GiB? no: 2.4 */
	for (i = 0; i < 3; ++i)
		tick(r, WARN_PM, 900);
	for (i = 0; i < 6; ++i)
		tick(r, 300, 900);					/* 2.4 GiB < 3 GiB floor */
	TEST_ASSERT(r->s.pressure == MEMBRANE_PRESSURE_WARNING,
		"absolute RAM floor blocks recovery");
	for (i = 0; i < 3; ++i)
		tick(r, 400, 900);					/* 3.2 GiB */
	TEST_ASSERT(r->s.pressure == MEMBRANE_PRESSURE_HEALTHY, "floor met");
	/* critical -> warning -> healthy, stepping down one level at a time */
	rig_init(r, MEMBRANE_ADAPTIVE_RECOMMEND);
	tick(r, CRIT_PM, 900);
	TEST_ASSERT(r->s.pressure == MEMBRANE_PRESSURE_CRITICAL, "critical needs "
		"only sustained/2 samples");
	for (i = 0; i < 3; ++i)
		tick(r, 99, 900);
	TEST_ASSERT(r->s.pressure == MEMBRANE_PRESSURE_CRITICAL, "99 < 100 exit");
	for (i = 0; i < 3; ++i)
		tick(r, 100, 900);
	TEST_ASSERT(r->s.pressure == MEMBRANE_PRESSURE_WARNING, "critical exit "
		"lands in WARNING, not HEALTHY");
	free(r);
}

/* --- cooldown / rate -------------------------------------------------- */

static void	test_cooldown_boundary(void)
{
	rig_t							*r = malloc(sizeof(*r));
	membrane_adaptive_decision_t	d;
	int64_t							t0;

	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	r->p.cooldown_s = 100;
	d = pressure_until_decision(r, WARN_PM, 5);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_ACT, "first act");
	did(r, &d);
	t0 = r->now;
	/* STEP_MS = 30 s: 3 ticks = 90 s < 100 s */
	d = tick(r, WARN_PM, 900);
	d = tick(r, WARN_PM, 900);
	d = tick(r, WARN_PM, 900);
	TEST_ASSERT(r->now - t0 == 90000 && d.kind == MEMBRANE_DECISION_COOLDOWN
		&& d.cooldown_remaining_s == 10, "90 s in: 10 s remain");
	d = tick(r, WARN_PM, 900);				/* 120 s >= 100 s */
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_ACT, "cooldown over");
	/* exact boundary with direct timestamps */
	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	r->p.cooldown_s = 60;
	r->s.pressure = MEMBRANE_PRESSURE_WARNING;
	r->s.above_count = 9;
	r->s.last_action_ms = r->now;
	{
		membrane_adaptive_sample_t	smp;

		make_snap(&r->snap, WARN_PM, 900);
		memset(&r->intel, 0, sizeof(r->intel));
		membrane_adaptive_make_sample(&r->snap, &r->intel, &r->lv, r->now
			+ 59999, &smp);
		smp.measured = MEMBRANE_PRESSURE_WARNING;
		membrane_adaptive_record_sample(&r->p, &r->s, &smp);
		membrane_adaptive_decide(&r->p, &r->s, r->now + 59999, &r->snap,
			&r->intel, &r->lv, &r->sf, &d);
		TEST_ASSERT(d.kind == MEMBRANE_DECISION_COOLDOWN, "59.999 s: cooling");
		membrane_adaptive_make_sample(&r->snap, &r->intel, &r->lv, r->now
			+ 60000, &smp);
		smp.measured = MEMBRANE_PRESSURE_WARNING;
		membrane_adaptive_record_sample(&r->p, &r->s, &smp);
		membrane_adaptive_decide(&r->p, &r->s, r->now + 60000, &r->snap,
			&r->intel, &r->lv, &r->sf, &d);
		TEST_ASSERT(d.kind == MEMBRANE_DECISION_ACT, "exactly 60 s: allowed");
	}
	free(r);
}

static void	test_rate_limit(void)
{
	rig_t							*r = malloc(sizeof(*r));
	membrane_adaptive_decision_t	d;
	int								acts = 0;
	int								i;

	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	r->p.cooldown_s = 0;
	r->p.max_actions_per_window = 2;
	r->p.window_s = 3600;
	r->p.allowed_classes = MEMBRANE_ADAPTIVE_CLASS_ALL;
	for (i = 0; i < 40; ++i)
	{
		d = tick(r, WARN_PM, 900);
		if (d.kind == MEMBRANE_DECISION_ACT)
		{
			acts++;
			did(r, &d);
		}
	}
	TEST_ASSERT(acts == 2, "no more than 2 actions per window");
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_RATE_LIMITED
		&& !strcmp(d.reason, MEMBRANE_ADAPT_REASON_RATE), "reason code");
	/* the window slides */
	r->now += 3600 * 1000;
	for (i = 0; i < 4; ++i)
		d = tick(r, WARN_PM, 900);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_ACT || d.kind
		== MEMBRANE_DECISION_NO_SAFE_STEP, "window expired => not rate "
			"limited");
	free(r);
}

/* --- hierarchy -------------------------------------------------------- */

static void	test_hierarchy_and_skips(void)
{
	rig_t							*r = malloc(sizeof(*r));
	membrane_adaptive_decision_t	d;

	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	r->p.cooldown_s = 0;
	r->p.max_actions_per_window = 50;
	r->p.allowed_classes = MEMBRANE_ADAPTIVE_CLASS_ALL;
	add_res(r, "main", 0, 0, 9000, 20);
	add_res(r, "spare", 0, 0, 3000, 0);
	d = pressure_until_decision(r, WARN_PM, 5);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_ACT && !strcmp(d.request.context,
			"4096"), "1st: halve the planned 8192 context (no reload)");
	TEST_ASSERT(!d.needs_reload && d.chain_count >= 3, "explainable chain");
	did(r, &d);
	d = tick(r, WARN_PM, 900);
	TEST_ASSERT(!strcmp(d.request.context, "2048"), "context keeps halving");
	did(r, &d);
	d = tick(r, WARN_PM, 900);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_ACT && !strcmp(d.request.context,
			"1024") , "to the floor");
	did(r, &d);
	/* floor reached: next step is KV precision (reload class) */
	d = tick(r, WARN_PM, 900);
	TEST_ASSERT(!strcmp(d.request.kv_precision, "q8") && d.needs_reload,
		"context at floor => KV precision native->q8, needs reload");
	did(r, &d);
	/* the change is pending: do not escalate past it */
	d = tick(r, WARN_PM, 900);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_DEFERRED
		&& !strcmp(d.reason, MEMBRANE_ADAPT_REASON_PENDING),
		"pending reload blocks escalation (deferred)");
	/* the model is reloaded (resident set changed): pending clears */
	r->lv.resident_count = 0;
	add_res(r, "main", 0, 0, 9000, 20);
	d = tick(r, WARN_PM, 900);					/* manual-looking change */
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_MANUAL_OVERRIDE ||
		d.kind == MEMBRANE_DECISION_ACT, "resident set changed");
	free(r);
}

static void	test_vram_only_steps_and_unload_last(void)
{
	rig_t							*r = malloc(sizeof(*r));
	membrane_adaptive_decision_t	d;

	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	r->p.cooldown_s = 0;
	r->p.max_actions_per_window = 50;
	r->p.allowed_classes = MEMBRANE_ADAPTIVE_CLASS_KV_PLACEMENT
		| MEMBRANE_ADAPTIVE_CLASS_GPU_LAYERS | MEMBRANE_ADAPTIVE_CLASS_UNLOAD;
	snprintf(r->lv.ctx, sizeof(r->lv.ctx), "1024");
	add_res(r, "main", 0, 0, 9000, 20);
	add_res(r, "spare", 0, 0, 3000, 0);
	/* host RAM pressure only: placement/gpu steps do not apply */
	d = pressure_until_decision(r, WARN_PM, 5);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_ACT && !strcmp(d.request.unload,
			"spare"), "RAM-only pressure skips VRAM-specific steps and "
			"unloads the idle non-default model");
	/* VRAM pressure: placement first */
	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	r->p.cooldown_s = 0;
	r->p.max_actions_per_window = 50;
	r->p.allowed_classes = MEMBRANE_ADAPTIVE_CLASS_ALL;
	snprintf(r->lv.ctx, sizeof(r->lv.ctx), "1024");
	snprintf(r->lv.kv_precision, sizeof(r->lv.kv_precision), "q5");
	add_res(r, "main", 0, 0, 9000, 20);
	{
		int	i;

		for (i = 0; i < 4; ++i)
			d = tick(r, 900, WARN_PM);
	}
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_ACT && !strcmp(
		d.request.kv_placement, "cpu"), "measured VRAM pressure => KV to cpu");
	did(r, &d);
	r->s.pending_reload = 0;
	d = tick(r, 900, WARN_PM);
	TEST_ASSERT(!strcmp(d.request.gpu_layers, "15"), "then -25% GPU layers "
		"(20 -> 15)");
	did(r, &d);
	r->s.pending_reload = 0;
	d = tick(r, 900, WARN_PM);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_ACT && !strcmp(
		d.request.gpu_layers, "11"), "layers keep stepping (15 -> 11)");
	/* no offloaded layers known, default model only: nothing safe left */
	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	r->p.cooldown_s = 0;
	r->p.allowed_classes = MEMBRANE_ADAPTIVE_CLASS_ALL;
	snprintf(r->lv.ctx, sizeof(r->lv.ctx), "1024");
	snprintf(r->lv.kv_precision, sizeof(r->lv.kv_precision), "q5");
	snprintf(r->lv.kv_placement, sizeof(r->lv.kv_placement), "cpu");
	add_res(r, "main", 0, 0, 9000, 0);
	{
		int	i;

		for (i = 0; i < 4; ++i)
			d = tick(r, 900, WARN_PM);
	}
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_NO_SAFE_STEP
		&& !strcmp(d.reason, MEMBRANE_ADAPT_REASON_NO_STEP),
		"nothing left that is safe (main is the default): ask the user");
	free(r);
}

static void	test_class_and_capability_gates(void)
{
	rig_t							*r = malloc(sizeof(*r));
	membrane_adaptive_decision_t	d;
	int								i;

	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	r->p.allowed_classes = MEMBRANE_ADAPTIVE_CLASS_KV_PRECISION;
	for (i = 0; i < 4; ++i)
		d = tick(r, WARN_PM, 900);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_ACT && !d.request.context[0]
		&& d.request.kv_precision[0], "context class not allowed => skipped "
			"(chain says so), next allowed class used");
	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	r->sf.state[MEMBRANE_ACTION_SET_CONTEXT] = MEMBRANE_CAPABILITY_UNSUPPORTED;
	r->sf.state[MEMBRANE_ACTION_SET_KV_PRECISION] = MEMBRANE_CAPABILITY_UNKNOWN;
	for (i = 0; i < 4; ++i)
		d = tick(r, WARN_PM, 900);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_NO_SAFE_STEP,
		"unsupported / unknown capability => never an action");
	/* a limited (Ollama-like) runtime: only unload is supported, and that
	 * class is not allowed by default => recommend-only outcome */
	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	memset(r->sf.state, 0, sizeof(r->sf.state));
	r->sf.state[MEMBRANE_ACTION_UNLOAD_MODEL] = MEMBRANE_CAPABILITY_SUPPORTED;
	add_res(r, "a", 0, 0, 1, 0);
	for (i = 0; i < 4; ++i)
		d = tick(r, WARN_PM, 900);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_NO_SAFE_STEP && !d.request.unload[0],
		"limited runtime + default classes => no action");
	free(r);
}

static void	test_estimate_only_does_not_act(void)
{
	rig_t							*r = malloc(sizeof(*r));
	membrane_adaptive_sample_t		smp;
	membrane_adaptive_decision_t	d;
	int								i;

	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	/* force a sustained pressured STATE, then a sample whose pressure comes
	 * only from an estimate */
	for (i = 0; i < 3; ++i)
		tick(r, WARN_PM, 900);
	make_snap(&r->snap, HEALTHY_PM, 900);
	memset(&smp, 0, sizeof(smp));
	smp.t_ms = r->now + STEP_MS;
	smp.valid = 1;
	smp.measured = MEMBRANE_PRESSURE_HEALTHY;
	smp.estimate_only = 1;
	smp.ram_permille = 700;
	smp.vram_permille = 700;
	r->now += STEP_MS;
	membrane_adaptive_record_sample(&r->p, &r->s, &smp);
	r->s.pressure = MEMBRANE_PRESSURE_WARNING;		/* sticky by hysteresis */
	membrane_adaptive_decide(&r->p, &r->s, r->now, &r->snap, &r->intel, &r->lv,
		&r->sf, &d);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_RECOMMEND
		&& !strcmp(d.reason, MEMBRANE_ADAPT_REASON_ESTIMATE),
		"an estimate-only signal may advise but never acts");
	free(r);
}

/* --- generation deferral / unload candidates -------------------------- */

static void	test_active_generation_never_unloaded(void)
{
	rig_t							*r = malloc(sizeof(*r));
	membrane_adaptive_decision_t	d;
	int								i;

	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	r->p.allowed_classes = MEMBRANE_ADAPTIVE_CLASS_UNLOAD;
	add_res(r, "main", 0, 0, 100, 0);
	add_res(r, "busy", 1, 0, 900, 0);			/* generating */
	add_res(r, "pinned", 0, 1, 800, 0);
	for (i = 0; i < 4; ++i)
		d = tick(r, WARN_PM, 900);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_NO_SAFE_STEP && !d.request.unload[0],
		"generating / pinned / default are never unload candidates");
	r->lv.resident[1].generating = 0;			/* the generation ended */
	d = tick(r, WARN_PM, 900);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_MANUAL_OVERRIDE ||
		d.kind == MEMBRANE_DECISION_ACT, "(generating flag is not a manual "
			"change)");
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_ACT && !strcmp(d.request.unload,
		"busy"), "once idle it becomes the largest safe candidate");
	free(r);
}

/* --- manual override --------------------------------------------------- */

static void	test_manual_override_adopted_not_undone(void)
{
	rig_t							*r = malloc(sizeof(*r));
	membrane_adaptive_decision_t	d;

	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	d = pressure_until_decision(r, WARN_PM, 5);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_ACT, "act");
	did(r, &d);
	/* the USER raises the context back */
	snprintf(r->lv.ctx, sizeof(r->lv.ctx), "8192");
	d = tick(r, WARN_PM, 900);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_MANUAL_OVERRIDE
		&& !strcmp(d.reason, MEMBRANE_ADAPT_REASON_MANUAL)
		&& !d.request.context[0], "drift detected, nothing requested");
	TEST_ASSERT(r->s.action_count >= 2 && r->s.actions[r->s.action_count - 1]
		.manual, "recorded as a manual change in history");
	d = tick(r, WARN_PM, 900);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_COOLDOWN, "a quiet period "
		"follows, never an immediate re-reduction");
	/* a manual model change is also adopted */
	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	add_res(r, "m1", 0, 0, 1, 0);
	tick(r, HEALTHY_PM, 900);
	r->lv.resident_count = 0;
	add_res(r, "m2", 0, 0, 1, 0);
	d = tick(r, HEALTHY_PM, 900);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_MANUAL_OVERRIDE, "model swap");
	free(r);
}

/* --- anti-thrashing + restore ------------------------------------------ */

static void	test_thrash_suppression_and_restore(void)
{
	rig_t							*r = malloc(sizeof(*r));
	membrane_adaptive_decision_t	d;
	int								i;
	int								suppressed = 0;

	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	r->p.auto_restore = 1;
	r->p.cooldown_s = 30;
	r->p.restore_after_s = 60;
	r->p.max_actions_per_window = 50;
	r->p.thrash_reversals = 3;
	/* a pathological host: pressure comes back every time we restore */
	for (i = 0; i < 400 && !suppressed; ++i)
	{
		int	pressured = (i / 14) % 2 == 0;

		d = tick(r, pressured ? WARN_PM : 900, 900);
		if (d.kind == MEMBRANE_DECISION_ACT)
			did(r, &d);
		if (d.kind == MEMBRANE_DECISION_SUPPRESSED)
			suppressed = 1;
	}
	TEST_ASSERT(suppressed, "repeated opposing actions are detected");
	TEST_ASSERT(!strcmp(d.reason, MEMBRANE_ADAPT_REASON_THRASH),
		"ADAPTATION_SUPPRESSED_THRASHING");
	TEST_ASSERT(r->s.suppressed_until_ms > r->now, "bounded suppression");
	/* still suppressed on the next ticks, even under pressure */
	d = tick(r, CRIT_PM, 900);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_SUPPRESSED, "stays suppressed");
	/* it expires after suppress_s, and only then adapts again */
	r->now = r->s.suppressed_until_ms;
	d = tick(r, HEALTHY_PM, 900);
	TEST_ASSERT(d.kind != MEMBRANE_DECISION_SUPPRESSED, "suppression ends "
		"after its window");
	free(r);
}

static void	test_recovery_and_restore_when_safe(void)
{
	rig_t							*r = malloc(sizeof(*r));
	membrane_adaptive_decision_t	d;
	int								i;

	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	r->p.auto_restore = 1;
	r->p.cooldown_s = 30;
	r->p.restore_after_s = 120;
	d = pressure_until_decision(r, WARN_PM, 5);
	TEST_ASSERT(!strcmp(d.request.context, "4096"), "reduced");
	did(r, &d);
	for (i = 0; i < 3; ++i)
		d = tick(r, 300, 900);				/* healthy, exit thresholds met */
	TEST_ASSERT(r->s.pressure == MEMBRANE_PRESSURE_HEALTHY, "recovered");
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_NONE, "not restored immediately");
	for (i = 0; i < 6; ++i)
	{
		d = tick(r, 300, 900);
		if (d.kind == MEMBRANE_DECISION_ACT)
			break ;
	}
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_ACT && d.is_restore
		&& !strcmp(d.request.context, "automatic"), "restore to the value it "
			"replaced, once healthy long enough");
	/* the user changed it meanwhile => leave it alone */
	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	r->p.auto_restore = 1;
	r->p.restore_after_s = 30;
	r->p.cooldown_s = 30;
	d = pressure_until_decision(r, WARN_PM, 5);
	did(r, &d);
	snprintf(r->lv.ctx, sizeof(r->lv.ctx), "3000");
	for (i = 0; i < 12; ++i)
	{
		d = tick(r, 300, 900);
		TEST_ASSERT(!d.is_restore, "never restore over a user's value");
	}
	free(r);
}

/* --- history ------------------------------------------------------------ */

static void	test_history_is_bounded(void)
{
	rig_t	*r = malloc(sizeof(*r));
	int		i;

	rig_init(r, MEMBRANE_ADAPTIVE_OBSERVE);
	r->p.history_max_samples = 50;
	r->p.history_max_age_s = 1000000;
	for (i = 0; i < 1000; ++i)
		tick(r, HEALTHY_PM, 900);
	TEST_ASSERT(r->s.sample_count == 50, "count bound");
	TEST_ASSERT(r->s.samples[49].t_ms == r->now, "newest kept");
	r->p.history_max_samples = 256;
	r->p.history_max_age_s = 300;
	for (i = 0; i < 100; ++i)
		tick(r, HEALTHY_PM, 900);
	TEST_ASSERT(r->s.sample_count <= 11, "age bound (300 s / 30 s steps)");
	for (i = 0; i < 2000; ++i)
		tick(r, HEALTHY_PM, 900);
	TEST_ASSERT(r->s.sample_count <= MEMBRANE_ADAPTIVE_MAX_SAMPLES, "hard cap");
	/* action history is capped too */
	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	{
		membrane_adaptive_decision_t	d;

		memset(&d, 0, sizeof(d));
		snprintf(d.request.context, sizeof(d.request.context), "2048");
		for (i = 0; i < 200; ++i)
			membrane_adaptive_note_action(&r->p, &r->s, r->now + i, &d, 1,
				&r->lv);
		TEST_ASSERT(r->s.action_count == MEMBRANE_ADAPTIVE_MAX_ACTIONS,
			"action history cap");
	}
	free(r);
}

static void	test_unobservable_runtime(void)
{
	rig_t							*r = malloc(sizeof(*r));
	membrane_adaptive_sample_t		smp;
	membrane_adaptive_decision_t	d;

	rig_init(r, MEMBRANE_ADAPTIVE_APPLY);
	membrane_obs_snapshot_init(&r->snap, MEMBRANE_RUNTIME_ID_NATIVE);
	membrane_obs_snapshot_finalize(&r->snap);		/* UNAVAILABLE */
	memset(&r->intel, 0, sizeof(r->intel));
	r->now += STEP_MS;
	membrane_adaptive_make_sample(&r->snap, &r->intel, &r->lv, r->now, &smp);
	TEST_ASSERT(!smp.valid, "invalid sample");
	membrane_adaptive_record_sample(&r->p, &r->s, &smp);
	membrane_adaptive_decide(&r->p, &r->s, r->now, &r->snap, &r->intel, &r->lv,
		&r->sf, &d);
	TEST_ASSERT(d.kind == MEMBRANE_DECISION_NONE && !strcmp(d.reason,
			MEMBRANE_ADAPT_REASON_UNOBSERVABLE) && r->s.pressure
				== MEMBRANE_PRESSURE_UNKNOWN, "no data => no action");
	free(r);
}

/* --- deterministic timeline simulation (K20) ----------------------------
 * SYNTHETIC model, clearly not a performance measurement: free RAM =
 * total - other_usage - kv(ctx), kv(ctx) = ctx * 0.5 MiB. `other_usage`
 * follows a scripted spike. The question the test answers is whether the
 * controller (with the real I3 thresholds) avoids a sustained critical
 * state, recovers headroom, stays inside its rate limit, and does not
 * oscillate -- not how fast any real model runs. */

static int	sim_free_permille(uint64_t other, uint64_t ctx)
{
	uint64_t	kv = ctx * 512 * 1024ull;
	uint64_t	used = other + kv;
	uint64_t	total = 8 * GIB;

	if (used >= total)
		return (0);
	return ((int)((total - used) * 1000 / total));
}

static void	test_timeline_simulation_benefit(void)
{
	int			with_adapt;
	int			worst[2];
	long		spike_sum[2];
	int			critical_ticks[2];
	int			actions[2];
	int			reversals[2];

	for (with_adapt = 0; with_adapt < 2; ++with_adapt)
	{
		rig_t							*r = malloc(sizeof(*r));
		uint64_t						ctx = 4096;
		int								t;
		int								last_dir = 0;
		int								worst_pm = 1000;
		int								crit = 0;
		int								acts = 0;
		int								rev = 0;
		long							sum = 0;

		rig_init(r, with_adapt ? MEMBRANE_ADAPTIVE_APPLY
			: MEMBRANE_ADAPTIVE_OFF);
		r->p.cooldown_s = 60;
		snprintf(r->lv.ctx, sizeof(r->lv.ctx), "%llu", (unsigned long long)ctx);
		for (t = 0; t < 240; ++t)
		{
			/* other processes: 3 GiB, spike to 5.8 GiB for ticks 40..160 */
			uint64_t	other = (t >= 40 && t < 160) ? 5939ull * 1024 * 1024
				: 3 * GIB;
			int			pm = sim_free_permille(other, ctx);
			membrane_adaptive_decision_t	d = tick(r, pm, 900);

			if (pm < worst_pm)
				worst_pm = pm;
			if (t >= 50 && t < 160)
				sum += pm;
			if (pm < 50)
				crit++;
			if (d.kind == MEMBRANE_DECISION_ACT)
			{
				int	dir = strtoull(d.request.context, NULL, 10) < ctx
					? -1 : 1;

				ctx = strtoull(d.request.context, NULL, 10);
				if (last_dir != 0 && dir != last_dir)
					rev++;
				last_dir = dir;
				acts++;
				did(r, &d);
			}
		}
		worst[with_adapt] = worst_pm;
		spike_sum[with_adapt] = sum;
		critical_ticks[with_adapt] = crit;
		actions[with_adapt] = acts;
		reversals[with_adapt] = rev;
		free(r);
	}
	/* 4096 ctx = 2 GiB of synthetic KV + 3 GiB other is healthy; the spike
	 * to 5.8 GiB makes it ~25 permille free (critical) without adaptation. */
	TEST_ASSERT(critical_ticks[0] > 100, "baseline (adaptation off) spends "
		"most of the spike in a critical state");
	TEST_ASSERT(critical_ticks[1] < critical_ticks[0] / 2,
		"adaptation spends far fewer ticks critical");
	TEST_ASSERT(spike_sum[1] > 5 * spike_sum[0], "mean headroom during the "
		"spike is several times higher with adaptation");
	TEST_ASSERT(worst[1] >= worst[0], "never worse");
	TEST_ASSERT(actions[1] >= 1 && actions[1] <= 4, "bounded number of "
		"actions (rate limit 2/h over a 2 h simulation)");
	TEST_ASSERT(reversals[1] == 0, "no oscillation");
	TEST_ASSERT(actions[0] == 0, "off mode takes no action");
}

int	main(void)
{
	test_defaults_off_and_conservative();
	test_policy_validation();
	test_modes_never_mutate_below_apply();
	test_transient_vs_sustained();
	test_hysteresis_boundaries();
	test_cooldown_boundary();
	test_rate_limit();
	test_hierarchy_and_skips();
	test_vram_only_steps_and_unload_last();
	test_class_and_capability_gates();
	test_estimate_only_does_not_act();
	test_active_generation_never_unloaded();
	test_manual_override_adopted_not_undone();
	test_thrash_suppression_and_restore();
	test_recovery_and_restore_when_safe();
	test_history_is_bounded();
	test_unobservable_runtime();
	test_timeline_simulation_benefit();
	printf("test_adaptive: all tests passed\n");
	return (0);
}
