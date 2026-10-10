#include <string.h>

#include "action.h"
#include "test_helpers.h"

/*
 * Milestone J: action.h/.c tests -- pure compiler, stale check and the
 * transaction engine driven by a mock executor. No host, no model.
 */

static void	surface_native(membrane_apply_surface_t *s)
{
	int	i;

	memset(s, 0, sizeof(*s));
	for (i = 0; i < MEMBRANE_ACTION_CODE_COUNT; ++i)
	{
		s->state[i] = MEMBRANE_CAPABILITY_SUPPORTED;
		s->reversible[i] = 1;
	}
	s->requires_reload[MEMBRANE_ACTION_SET_GPU_LAYERS] = 1;
	s->requires_reload[MEMBRANE_ACTION_SET_KV_PRECISION] = 1;
	s->requires_reload[MEMBRANE_ACTION_SET_KV_PLACEMENT] = 1;
}

static void	add_res(membrane_apply_live_t *l, const char *name, int pinned,
				int generating, uint64_t size)
{
	membrane_apply_resident_t	*r = &l->resident[l->resident_count++];

	snprintf(r->name, sizeof(r->name), "%s", name);
	r->activity_known = 1;
	r->pinned = pinned;
	r->generating = generating;
	r->size_bytes = size;
}

static void	live_base(membrane_apply_live_t *l)
{
	memset(l, 0, sizeof(*l));
	l->runtime_reachable = 1;
	snprintf(l->default_model, sizeof(l->default_model), "main");
	l->ctx_known = l->kv_precision_known = l->kv_placement_known
		= l->gpu_layers_known = 1;
	snprintf(l->ctx, sizeof(l->ctx), "automatic");
	snprintf(l->kv_precision, sizeof(l->kv_precision), "automatic");
	snprintf(l->kv_placement, sizeof(l->kv_placement), "automatic");
	snprintf(l->gpu_layers, sizeof(l->gpu_layers), "automatic");
}

static void	intel_with(membrane_memory_intelligence_t *i, const char *code1,
				const char *code2)
{
	memset(i, 0, sizeof(*i));
	i->schema_version = 1;
	snprintf(i->runtime_id, sizeof(i->runtime_id), "membrane-native");
	if (code1 != NULL)
		snprintf(i->recommendations[i->recommendation_count++].code,
			MEMBRANE_INTEL_CODE_MAX, "%s", code1);
	if (code2 != NULL)
		snprintf(i->recommendations[i->recommendation_count++].code,
			MEMBRANE_INTEL_CODE_MAX, "%s", code2);
}

static const membrane_action_validation_t	g_ok = {1, 1, "Planner v2: feasible"};

static int	has_unapplied(const membrane_action_plan_t *p, const char *src,
				const char *reason)
{
	size_t	i;

	for (i = 0; i < p->unapplied_count; ++i)
		if (!strcmp(p->unapplied[i].source, src)
			&& !strcmp(p->unapplied[i].reason_code, reason))
			return (1);
	return (0);
}

/* --- compiler ------------------------------------------------------ */

static void	test_recommendation_without_target_is_advisory(void)
{
	membrane_memory_intelligence_t	intel;
	membrane_apply_surface_t		surf;
	membrane_apply_live_t			live;
	membrane_action_request_t		req;
	membrane_action_plan_t			plan;

	intel_with(&intel, MEMBRANE_INTEL_REC_REDUCE_CONTEXT,
		MEMBRANE_INTEL_REC_USE_LOWER_KV_PRECISION);
	surface_native(&surf);
	live_base(&live);
	memset(&req, 0, sizeof(req));
	membrane_action_compile(&intel, &surf, &live, &req, &g_ok, &plan);
	TEST_ASSERT(plan.action_count == 0, "no value => no action");
	TEST_ASSERT(has_unapplied(&plan, "REDUCE_CONTEXT",
			MEMBRANE_UNAPPLIED_NO_TARGET), "REDUCE_CONTEXT advisory");
	TEST_ASSERT(has_unapplied(&plan, "USE_LOWER_KV_PRECISION",
			MEMBRANE_UNAPPLIED_NO_TARGET), "KV advisory");
}

static void	test_user_context_becomes_action(void)
{
	membrane_memory_intelligence_t	intel;
	membrane_apply_surface_t		surf;
	membrane_apply_live_t			live;
	membrane_action_request_t		req;
	membrane_action_plan_t			plan;

	intel_with(&intel, MEMBRANE_INTEL_REC_REDUCE_CONTEXT, NULL);
	surface_native(&surf);
	live_base(&live);
	memset(&req, 0, sizeof(req));
	snprintf(req.context, sizeof(req.context), "4096");
	membrane_action_compile(&intel, &surf, &live, &req, &g_ok, &plan);
	TEST_ASSERT(plan.action_count == 1, "one action");
	TEST_ASSERT(plan.actions[0].code == MEMBRANE_ACTION_SET_CONTEXT, "code");
	TEST_ASSERT(!strcmp(plan.actions[0].requested, "4096"), "value");
	TEST_ASSERT(plan.actions[0].mutates_state == 1, "mutates");
	TEST_ASSERT(plan.actions[0].reversible && plan.actions[0].rollback_known,
		"reversible");
	TEST_ASSERT(!strcmp(plan.actions[0].source_recommendation,
			"REDUCE_CONTEXT"), "ties to recommendation");
	TEST_ASSERT(plan.actions[0].origin == MEMBRANE_ACTION_ORIGIN_USER_REQUEST,
		"user origin");
	TEST_ASSERT(plan.unapplied_count == 0, "rec consumed, no advisory dup");
}

static void	test_planner_rejection_blocks_set(void)
{
	membrane_apply_surface_t		surf;
	membrane_apply_live_t			live;
	membrane_action_request_t		req;
	membrane_action_plan_t			plan;
	membrane_action_validation_t	bad = {1, 0, "does not fit"};
	membrane_action_validation_t	none = {0, 0, ""};

	surface_native(&surf);
	live_base(&live);
	memset(&req, 0, sizeof(req));
	snprintf(req.kv_precision, sizeof(req.kv_precision), "q8");
	membrane_action_compile(NULL, &surf, &live, &req, &bad, &plan);
	TEST_ASSERT(plan.action_count == 0, "infeasible => refused");
	TEST_ASSERT(has_unapplied(&plan, "SET_KV_PRECISION",
			MEMBRANE_UNAPPLIED_NOT_VALIDATED), "reason");
	membrane_action_compile(NULL, &surf, &live, &req, &none, &plan);
	TEST_ASSERT(plan.action_count == 0, "unchecked => refused (fail closed)");
	membrane_action_compile(NULL, &surf, &live, &req, NULL, &plan);
	TEST_ASSERT(plan.action_count == 0, "NULL validation => refused");
}

static void	test_capability_unsupported_and_unknown(void)
{
	membrane_memory_intelligence_t	intel;
	membrane_apply_surface_t		surf;
	membrane_apply_live_t			live;
	membrane_action_request_t		req;
	membrane_action_plan_t			plan;

	intel_with(&intel, MEMBRANE_INTEL_REC_CHANGE_KV_PLACEMENT, NULL);
	surface_native(&surf);
	live_base(&live);
	memset(&req, 0, sizeof(req));
	snprintf(req.kv_placement, sizeof(req.kv_placement), "cpu");
	surf.state[MEMBRANE_ACTION_SET_KV_PLACEMENT]
		= MEMBRANE_CAPABILITY_UNSUPPORTED;
	membrane_action_compile(&intel, &surf, &live, &req, &g_ok, &plan);
	TEST_ASSERT(plan.action_count == 0, "unsupported => no action");
	TEST_ASSERT(has_unapplied(&plan, "SET_KV_PLACEMENT",
			MEMBRANE_UNAPPLIED_UNSUPPORTED), "unsupported reason");
	surf.state[MEMBRANE_ACTION_SET_KV_PLACEMENT] = MEMBRANE_CAPABILITY_UNKNOWN;
	membrane_action_compile(&intel, &surf, &live, &req, &g_ok, &plan);
	TEST_ASSERT(plan.action_count == 0, "unknown => no action (fail closed)");
	TEST_ASSERT(has_unapplied(&plan, "SET_KV_PLACEMENT",
			MEMBRANE_UNAPPLIED_UNKNOWN_CAP), "unknown reason");
	surf.state[MEMBRANE_ACTION_SET_KV_PLACEMENT] = MEMBRANE_CAPABILITY_PARTIAL;
	membrane_action_compile(&intel, &surf, &live, &req, &g_ok, &plan);
	TEST_ASSERT(plan.action_count == 0, "partial => no action");
	membrane_action_compile(&intel, NULL, &live, &req, &g_ok, &plan);
	TEST_ASSERT(plan.action_count == 0, "NULL surface => no action");
}

static void	test_unload_candidate_rule(void)
{
	membrane_apply_live_t	live;
	char					why[240];

	live_base(&live);
	add_res(&live, "main", 0, 0, 900);		/* default: never chosen */
	add_res(&live, "pinned", 1, 0, 800);
	add_res(&live, "busy", 0, 1, 700);
	TEST_ASSERT(membrane_action_pick_unload_candidate(&live, why,
			sizeof(why)) < 0, "nothing qualifies");
	TEST_ASSERT(strstr(why, "pinned") != NULL, "explains why");
	add_res(&live, "zeta", 0, 0, 100);
	add_res(&live, "alpha", 0, 0, 100);
	add_res(&live, "big", 0, 0, 500);
	TEST_ASSERT(!strcmp(live.resident[membrane_action_pick_unload_candidate(
				&live, why, sizeof(why))].name, "big"), "largest wins");
	live.resident_count--;
	TEST_ASSERT(!strcmp(live.resident[membrane_action_pick_unload_candidate(
				&live, why, sizeof(why))].name, "alpha"), "tie: name order");
	live.resident[4].activity_known = 0;			/* alpha unknown */
	TEST_ASSERT(!strcmp(live.resident[membrane_action_pick_unload_candidate(
				&live, why, sizeof(why))].name, "zeta"),
		"unknown activity never auto-picked");
}

static void	test_unload_from_recommendation(void)
{
	membrane_memory_intelligence_t	intel;
	membrane_apply_surface_t		surf;
	membrane_apply_live_t			live;
	membrane_action_request_t		req;
	membrane_action_plan_t			plan;

	intel_with(&intel, MEMBRANE_INTEL_REC_UNLOAD_UNUSED_MODEL, NULL);
	surface_native(&surf);
	live_base(&live);
	add_res(&live, "main", 0, 0, 900);
	add_res(&live, "spare", 0, 0, 300);
	memset(&req, 0, sizeof(req));
	membrane_action_compile(&intel, &surf, &live, &req, NULL, &plan);
	TEST_ASSERT(plan.action_count == 1, "one unload");
	TEST_ASSERT(plan.actions[0].code == MEMBRANE_ACTION_UNLOAD_MODEL, "code");
	TEST_ASSERT(!strcmp(plan.actions[0].model, "spare"), "safe candidate");
	TEST_ASSERT(plan.actions[0].origin == MEMBRANE_ACTION_ORIGIN_RECOMMENDATION,
		"recommendation origin");
	TEST_ASSERT(plan.actions[0].reason[0] != '\0', "explains choice");
}

static void	test_pinned_and_generating_protected(void)
{
	membrane_apply_surface_t	surf;
	membrane_apply_live_t		live;
	membrane_action_request_t	req;
	membrane_action_plan_t		plan;

	surface_native(&surf);
	live_base(&live);
	add_res(&live, "pin", 1, 0, 1);
	add_res(&live, "gen", 0, 1, 1);
	memset(&req, 0, sizeof(req));
	snprintf(req.unload, sizeof(req.unload), "pin");
	membrane_action_compile(NULL, &surf, &live, &req, NULL, &plan);
	TEST_ASSERT(plan.action_count == 0, "pinned never unloaded");
	TEST_ASSERT(has_unapplied(&plan, "UNLOAD_MODEL",
			MEMBRANE_UNAPPLIED_REJECTED), "refused");
	snprintf(req.unload, sizeof(req.unload), "gen");
	membrane_action_compile(NULL, &surf, &live, &req, NULL, &plan);
	TEST_ASSERT(plan.action_count == 0, "generating never unloaded");
	memset(&req, 0, sizeof(req));
	snprintf(req.unload, sizeof(req.unload), "nope");
	membrane_action_compile(NULL, &surf, &live, &req, NULL, &plan);
	TEST_ASSERT(plan.action_count == 0, "not resident rejected");
}

static void	test_deterministic_order_and_equality(void)
{
	membrane_apply_surface_t	surf;
	membrane_apply_live_t		live;
	membrane_action_request_t	req;
	membrane_action_plan_t		a;
	membrane_action_plan_t		b;

	surface_native(&surf);
	live_base(&live);
	add_res(&live, "m1", 0, 0, 5);
	add_res(&live, "m2", 0, 0, 5);
	memset(&req, 0, sizeof(req));
	snprintf(req.pin, sizeof(req.pin), "m2");
	snprintf(req.gpu_layers, sizeof(req.gpu_layers), "20");
	snprintf(req.context, sizeof(req.context), "2048");
	snprintf(req.unload, sizeof(req.unload), "m1");
	membrane_action_compile(NULL, &surf, &live, &req, &g_ok, &a);
	membrane_action_compile(NULL, &surf, &live, &req, &g_ok, &b);
	TEST_ASSERT(a.action_count == 4, "four actions");
	TEST_ASSERT(a.actions[0].code == MEMBRANE_ACTION_UNLOAD_MODEL
		&& a.actions[1].code == MEMBRANE_ACTION_SET_CONTEXT
		&& a.actions[2].code == MEMBRANE_ACTION_SET_GPU_LAYERS
		&& a.actions[3].code == MEMBRANE_ACTION_PIN_MODEL,
		"unload, settings, pin");
	TEST_ASSERT(memcmp(&a, &b, sizeof(a)) == 0, "byte-identical plans");
	TEST_ASSERT(a.any_requires_reload == 1, "reload disclosed");
}

static void	test_no_change_and_bad_values(void)
{
	membrane_apply_surface_t	surf;
	membrane_apply_live_t		live;
	membrane_action_request_t	req;
	membrane_action_plan_t		plan;

	surface_native(&surf);
	live_base(&live);
	snprintf(live.ctx, sizeof(live.ctx), "4096");
	memset(&req, 0, sizeof(req));
	snprintf(req.context, sizeof(req.context), "4096");
	snprintf(req.kv_precision, sizeof(req.kv_precision), "q9");
	snprintf(req.gpu_layers, sizeof(req.gpu_layers), "-3");
	membrane_action_compile(NULL, &surf, &live, &req, &g_ok, &plan);
	TEST_ASSERT(plan.action_count == 0, "nothing to do");
	TEST_ASSERT(has_unapplied(&plan, "SET_CONTEXT",
			MEMBRANE_UNAPPLIED_NO_CHANGE), "already set");
	TEST_ASSERT(has_unapplied(&plan, "SET_KV_PRECISION",
			MEMBRANE_UNAPPLIED_REJECTED), "bad kv");
	TEST_ASSERT(has_unapplied(&plan, "SET_GPU_LAYERS",
			MEMBRANE_UNAPPLIED_REJECTED), "bad layers");
}

/* --- stale check --------------------------------------------------- */

static void	test_stale_detection(void)
{
	membrane_apply_live_t	a;
	membrane_apply_live_t	b;
	membrane_apply_state_t	sa;
	membrane_apply_state_t	sb;
	char					drift[200];

	live_base(&a);
	add_res(&a, "x", 0, 0, 1);
	b = a;
	membrane_apply_fingerprint(&a, &sa);
	membrane_apply_fingerprint(&b, &sb);
	TEST_ASSERT(membrane_action_check_stale(&sa, &sb, drift, sizeof(drift))
		== 0, "identical => fresh");
	b.resident[0].generating = 1;
	membrane_apply_fingerprint(&b, &sb);
	TEST_ASSERT(membrane_action_check_stale(&sa, &sb, drift, sizeof(drift))
		== 1 && strstr(drift, "resident.0"), "generating drift");
	b = a;
	add_res(&b, "y", 0, 0, 1);
	membrane_apply_fingerprint(&b, &sb);
	TEST_ASSERT(membrane_action_check_stale(&sa, &sb, drift, sizeof(drift))
		>= 1, "new resident drift");
	b = a;
	snprintf(b.default_model, sizeof(b.default_model), "other");
	membrane_apply_fingerprint(&b, &sb);
	TEST_ASSERT(membrane_action_check_stale(&sa, &sb, NULL, 0) == 1,
		"config drift");
	b = a;
	b.runtime_reachable = 0;
	membrane_apply_fingerprint(&b, &sb);
	TEST_ASSERT(membrane_action_check_stale(&sa, &sb, NULL, 0) == 1,
		"runtime unavailable drift");
}

static void	test_recommendation_no_longer_holds(void)
{
	membrane_memory_intelligence_t	intel;
	membrane_apply_surface_t		surf;
	membrane_apply_live_t			live;
	membrane_action_request_t		req;
	membrane_action_plan_t			plan;
	membrane_memory_intelligence_t	fresh;
	char							missing[64];

	intel_with(&intel, MEMBRANE_INTEL_REC_UNLOAD_UNUSED_MODEL, NULL);
	surface_native(&surf);
	live_base(&live);
	add_res(&live, "main", 0, 0, 9);
	add_res(&live, "spare", 0, 0, 3);
	memset(&req, 0, sizeof(req));
	membrane_action_compile(&intel, &surf, &live, &req, NULL, &plan);
	TEST_ASSERT(plan.action_count == 1, "planned");
	TEST_ASSERT(membrane_action_recommendations_still_hold(&plan, &intel,
			missing, sizeof(missing)), "still holds");
	intel_with(&fresh, NULL, NULL);			/* pressure disappeared */
	TEST_ASSERT(!membrane_action_recommendations_still_hold(&plan, &fresh,
			missing, sizeof(missing)) && !strcmp(missing,
				"UNLOAD_UNUSED_MODEL"), "pressure gone => stale");
}

/* --- engine -------------------------------------------------------- */

typedef struct s_mock
{
	membrane_apply_live_t	live;
	int						fail_exec_at;		/* action index, -1 none */
	int						fail_rollback_for;	/* code, -1 none */
	int						exec_calls;
	int						rollback_calls;
	int						order[16];
}	mock_t;

static int	mock_read(void *ctx, membrane_apply_live_t *out)
{
	*out = ((mock_t *)ctx)->live;
	return (1);
}

static int	mock_exec(void *ctx, const membrane_action_t *a, char *err,
				size_t n)
{
	mock_t	*m = ctx;
	int		idx = m->exec_calls++;

	m->order[idx] = (int)a->code;
	if (idx == m->fail_exec_at)
	{
		snprintf(err, n, "boom");
		return (0);
	}
	return (1);
}

static int	mock_rollback(void *ctx, const membrane_action_t *a, char *err,
				size_t n)
{
	mock_t	*m = ctx;

	m->rollback_calls++;
	if ((int)a->code == m->fail_rollback_for)
	{
		snprintf(err, n, "cannot undo");
		return (0);
	}
	return (1);
}

static void	engine_plan(membrane_action_plan_t *plan, mock_t *m, int with_act)
{
	membrane_apply_surface_t	surf;
	membrane_action_request_t	req;

	surface_native(&surf);
	live_base(&m->live);
	add_res(&m->live, "m1", 0, 0, 5);
	memset(&req, 0, sizeof(req));
	snprintf(req.context, sizeof(req.context), "2048");
	snprintf(req.kv_precision, sizeof(req.kv_precision), "q8");
	snprintf(req.pin, sizeof(req.pin), "m1");
	if (with_act)
		snprintf(req.activate, sizeof(req.activate), "m9");
	membrane_action_compile(NULL, &surf, &m->live, &req, &g_ok, plan);
	m->fail_exec_at = -1;
	m->fail_rollback_for = -1;
	m->exec_calls = 0;
	m->rollback_calls = 0;
}

static void	test_engine_success(void)
{
	mock_t						m;
	membrane_action_plan_t		plan;
	membrane_action_result_t	res;
	membrane_action_executor_t	ex = {&m, mock_read, mock_exec, mock_rollback};

	engine_plan(&plan, &m, 0);
	membrane_action_run(&plan, &ex, &res);
	TEST_ASSERT(res.status == MEMBRANE_APPLY_APPLIED, "applied");
	TEST_ASSERT(m.exec_calls == 3 && m.rollback_calls == 0, "all ran");
	TEST_ASSERT(m.order[0] == MEMBRANE_ACTION_SET_CONTEXT
		&& m.order[1] == MEMBRANE_ACTION_SET_KV_PRECISION
		&& m.order[2] == MEMBRANE_ACTION_PIN_MODEL, "ordered");
}

static void	test_engine_failure_rolls_back(void)
{
	mock_t						m;
	membrane_action_plan_t		plan;
	membrane_action_result_t	res;
	membrane_action_executor_t	ex = {&m, mock_read, mock_exec, mock_rollback};

	engine_plan(&plan, &m, 0);
	m.fail_exec_at = 2;
	membrane_action_run(&plan, &ex, &res);
	TEST_ASSERT(res.status == MEMBRANE_APPLY_ROLLED_BACK, "rolled back");
	TEST_ASSERT(m.exec_calls == 3 && m.rollback_calls == 2, "2 undone");
	TEST_ASSERT(res.outcome[0] == MEMBRANE_OUTCOME_ROLLED_BACK
		&& res.outcome[2] == MEMBRANE_OUTCOME_FAILED, "outcomes");
	TEST_ASSERT(res.applied_count == 0, "nothing left applied");
}

static void	test_engine_first_action_fails(void)
{
	mock_t						m;
	membrane_action_plan_t		plan;
	membrane_action_result_t	res;
	membrane_action_executor_t	ex = {&m, mock_read, mock_exec, mock_rollback};

	engine_plan(&plan, &m, 0);
	m.fail_exec_at = 0;
	membrane_action_run(&plan, &ex, &res);
	TEST_ASSERT(res.status == MEMBRANE_APPLY_FAILED, "failed");
	TEST_ASSERT(m.exec_calls == 1, "later actions never run");
}

static void	test_engine_rollback_failure_reported(void)
{
	mock_t						m;
	membrane_action_plan_t		plan;
	membrane_action_result_t	res;
	membrane_action_executor_t	ex = {&m, mock_read, mock_exec, mock_rollback};

	engine_plan(&plan, &m, 0);
	m.fail_exec_at = 2;
	m.fail_rollback_for = MEMBRANE_ACTION_SET_KV_PRECISION;
	membrane_action_run(&plan, &ex, &res);
	TEST_ASSERT(res.status == MEMBRANE_APPLY_ROLLBACK_FAILED, "reported");
	TEST_ASSERT(res.outcome[1] == MEMBRANE_OUTCOME_ROLLBACK_FAILED, "which");
	TEST_ASSERT(m.rollback_calls == 2, "keeps undoing the rest");
	TEST_ASSERT(res.outcome[0] == MEMBRANE_OUTCOME_ROLLED_BACK, "others undone");
	TEST_ASSERT(res.error[1][0] != '\0', "error text kept");
}

static void	test_engine_partial_with_irreversible(void)
{
	mock_t						m;
	membrane_action_plan_t		plan;
	membrane_action_result_t	res;
	membrane_action_executor_t	ex = {&m, mock_read, mock_exec, mock_rollback};

	engine_plan(&plan, &m, 1);
	/* actions: ctx, kv, ACTIVATE m9 (irreversible), pin m1 */
	TEST_ASSERT(plan.action_count == 4, "four");
	TEST_ASSERT(plan.actions[2].code == MEMBRANE_ACTION_ACTIVATE_MODEL
		&& !plan.actions[2].reversible, "activate irreversible");
	m.fail_exec_at = 3;
	membrane_action_run(&plan, &ex, &res);
	TEST_ASSERT(res.status == MEMBRANE_APPLY_PARTIALLY_APPLIED, "partial");
	TEST_ASSERT(res.outcome[2] == MEMBRANE_OUTCOME_APPLIED, "stays applied");
	TEST_ASSERT(res.outcome[0] == MEMBRANE_OUTCOME_ROLLED_BACK, "rev undone");
}

static void	test_engine_stale_refuses_before_mutation(void)
{
	mock_t						m;
	membrane_action_plan_t		plan;
	membrane_action_result_t	res;
	membrane_action_executor_t	ex = {&m, mock_read, mock_exec, mock_rollback};

	engine_plan(&plan, &m, 0);
	m.live.resident[0].generating = 1;		/* drift after planning */
	membrane_action_run(&plan, &ex, &res);
	TEST_ASSERT(res.status == MEMBRANE_APPLY_STALE_PRECONDITION, "stale");
	TEST_ASSERT(m.exec_calls == 0, "nothing touched");
	TEST_ASSERT(strstr(res.detail, "rerun") != NULL, "tells user to rerun");
	engine_plan(&plan, &m, 0);
	m.live.runtime_reachable = 0;
	membrane_action_run(&plan, &ex, &res);
	TEST_ASSERT(res.status == MEMBRANE_APPLY_REFUSED && m.exec_calls == 0,
		"unreachable => refused");
}

static void	test_engine_empty_plan(void)
{
	membrane_action_plan_t		plan;
	membrane_action_result_t	res;
	mock_t						m;
	membrane_action_executor_t	ex = {&m, mock_read, mock_exec, mock_rollback};

	memset(&plan, 0, sizeof(plan));
	memset(&m, 0, sizeof(m));
	membrane_action_run(&plan, &ex, &res);
	TEST_ASSERT(res.status == MEMBRANE_APPLY_NOTHING_TO_APPLY, "empty");
	TEST_ASSERT(m.exec_calls == 0, "no calls");
}

static void	test_reload_restore_on_failure(void)
{
	mock_t						m;
	membrane_apply_surface_t	surf;
	membrane_action_request_t	req;
	membrane_action_plan_t		plan;
	membrane_action_result_t	res;
	membrane_action_executor_t	ex = {&m, mock_read, mock_exec, mock_rollback};

	surface_native(&surf);
	live_base(&m.live);
	add_res(&m.live, "m1", 0, 0, 5);
	memset(&req, 0, sizeof(req));
	snprintf(req.gpu_layers, sizeof(req.gpu_layers), "10");
	req.reload = 1;
	membrane_action_compile(NULL, &surf, &m.live, &req, &g_ok, &plan);
	TEST_ASSERT(plan.action_count == 2, "set + reload");
	TEST_ASSERT(plan.actions[1].code == MEMBRANE_ACTION_ACTIVATE_MODEL
		&& plan.actions[1].restore_on_failure, "reload restores itself");
	m.fail_exec_at = 1;
	m.fail_rollback_for = -1;
	m.exec_calls = m.rollback_calls = 0;
	membrane_action_run(&plan, &ex, &res);
	TEST_ASSERT(res.status == MEMBRANE_APPLY_ROLLED_BACK, "rolled back");
	TEST_ASSERT(res.outcome[1] == MEMBRANE_OUTCOME_ROLLED_BACK,
		"model restored");
	TEST_ASSERT(m.rollback_calls == 2, "settings undone, then model restored");
}

int	main(void)
{
	test_recommendation_without_target_is_advisory();
	test_user_context_becomes_action();
	test_planner_rejection_blocks_set();
	test_capability_unsupported_and_unknown();
	test_unload_candidate_rule();
	test_unload_from_recommendation();
	test_pinned_and_generating_protected();
	test_deterministic_order_and_equality();
	test_no_change_and_bad_values();
	test_stale_detection();
	test_recommendation_no_longer_holds();
	test_engine_success();
	test_engine_failure_rolls_back();
	test_engine_first_action_fails();
	test_engine_rollback_failure_reported();
	test_engine_partial_with_irreversible();
	test_engine_stale_refuses_before_mutation();
	test_engine_empty_plan();
	test_reload_restore_on_failure();
	printf("test_action: all tests passed\n");
	return (0);
}
