#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "adaptive_controller.h"
#include "test_helpers.h"

using json = nlohmann::json;

/*
 * Milestone K: the controller tick, driven with a fake clock, fake
 * observations and a fake J executor. No sleeps (except the one thread
 * start/stop test, which only proves the loop does not spin), no host, no
 * service, no model. The pure decision logic is covered by
 * tools/membrane-run/test_adaptive.c.
 */

#define GIB	(1024ull * 1024 * 1024)

struct s_world
{
	membrane_adaptive_policy_t	policy;
	std::string					config_error;
	int							ram_pm = 900;
	int							vram_pm = 900;
	bool						observable = true;
	bool						reachable = true;
	std::string					ctx = "automatic";
	int64_t						now = 5000000;
	int							observe_calls = 0;
	int							execute_calls = 0;
	bool						exec_ok = true;
	membrane_apply_request_t	last_req;
	membrane_apply_surface_t	surface;
	std::vector<std::string>	residents;
};

static s_world	*g_w;

static void	make_snap(membrane_observation_snapshot_t *s, int ram_pm, int vram_pm)
{
	membrane_obs_snapshot_init(s, MEMBRANE_RUNTIME_ID_NATIVE);
	s->runtime_observable = 1;
	membrane_obs_str_set(&s->runtime_availability, "available",
		MEMBRANE_OBS_PROV_STATIC_METADATA, "t");
	membrane_obs_u64_set(&s->ram_total_bytes, 8 * GIB,
		MEMBRANE_OBS_PROV_MEASURED, "t");
	membrane_obs_u64_set(&s->ram_available_bytes,
		(8 * GIB * (uint64_t)ram_pm + 999) / 1000,
		MEMBRANE_OBS_PROV_MEASURED, "t");
	membrane_obs_u64_set(&s->gpu_device_count, 1, MEMBRANE_OBS_PROV_MEASURED,
		"t");
	membrane_obs_u64_set(&s->vram_total_bytes, 8 * GIB,
		MEMBRANE_OBS_PROV_MEASURED, "t");
	membrane_obs_u64_set(&s->vram_free_bytes,
		(8 * GIB * (uint64_t)vram_pm + 999) / 1000,
		MEMBRANE_OBS_PROV_MEASURED, "t");
	membrane_obs_u64_set(&s->context_planned, 8192,
		MEMBRANE_OBS_PROV_ESTIMATED, "planner_v2");
	membrane_obs_str_set(&s->kv_planned_precision, "native",
		MEMBRANE_OBS_PROV_ESTIMATED, "planner_v2");
	membrane_obs_snapshot_finalize(s);
}

static membrane_adaptive_deps_t	make_deps(void)
{
	membrane_adaptive_deps_t	d;

	d.now_ms = []() -> int64_t { return (g_w->now); };
	d.observe = [](const std::string &, membrane_observation_snapshot_t *snap,
			membrane_memory_intelligence_t *intel, std::string *err) -> bool
	{
		membrane_runtime_capabilities_t	caps;

		g_w->observe_calls++;
		if (!g_w->observable)
		{
			*err = "down";
			return (false);
		}
		memset(&caps, 0, sizeof(caps));
		make_snap(snap, g_w->ram_pm, g_w->vram_pm);
		membrane_memory_intelligence_assemble(snap, &caps, intel);
		return (true);
	};
	d.read_live = [](const std::string &, membrane_apply_live_t *l,
			std::string *) -> bool
	{
		memset(l, 0, sizeof(*l));
		l->runtime_reachable = g_w->reachable ? 1 : 0;
		l->ctx_known = l->kv_precision_known = l->kv_placement_known
			= l->gpu_layers_known = 1;
		snprintf(l->ctx, sizeof(l->ctx), "%s", g_w->ctx.c_str());
		snprintf(l->kv_precision, sizeof(l->kv_precision), "automatic");
		snprintf(l->kv_placement, sizeof(l->kv_placement), "automatic");
		snprintf(l->gpu_layers, sizeof(l->gpu_layers), "automatic");
		snprintf(l->default_model, sizeof(l->default_model), "main");
		for (const auto &r : g_w->residents)
		{
			membrane_apply_resident_t	*x = &l->resident[l->resident_count++];

			snprintf(x->name, sizeof(x->name), "%s", r.c_str());
			x->activity_known = 1;
			x->size_bytes = 100;
		}
		return (true);
	};
	d.surface = [](const std::string &, membrane_apply_surface_t *s) -> bool
	{
		*s = g_w->surface;
		return (true);
	};
	d.load_policy = [](membrane_adaptive_policy_t *p, std::string *e) -> bool
	{
		*p = g_w->policy;
		*e = g_w->config_error;
		return (true);
	};
	d.execute = [](const membrane_apply_request_t &r,
			membrane_apply_outcome_t *o, std::string *) -> bool
	{
		g_w->execute_calls++;
		g_w->last_req = r;
		*o = membrane_apply_outcome_t();
		o->attempted = true;
		o->applied = g_w->exec_ok;
		o->status = g_w->exec_ok ? "applied" : "rolled_back";
		o->detail = g_w->exec_ok ? "all actions applied" : "boom";
		if (g_w->exec_ok && r.req.context[0])
			g_w->ctx = r.req.context;
		return (true);
	};
	return (d);
}

static void	world_init(s_world *w, membrane_adaptive_mode_t mode)
{
	w->policy = membrane_adaptive_policy_t();
	membrane_adaptive_policy_defaults(&w->policy);
	w->policy.enabled = mode != MEMBRANE_ADAPTIVE_OFF;
	w->policy.mode = mode;
	w->policy.cooldown_s = 90;
	for (int i = 0; i < MEMBRANE_ACTION_CODE_COUNT; ++i)
	{
		w->surface.state[i] = MEMBRANE_CAPABILITY_SUPPORTED;
		w->surface.reversible[i] = 1;
	}
	g_w = w;
}

static void	ticks(membrane_adaptive_controller_t &c, int n, int ram_pm)
{
	for (int i = 0; i < n; ++i)
	{
		g_w->ram_pm = ram_pm;
		g_w->now += 30000;
		c.tick();
	}
}

static void	test_off_by_default_does_nothing(void)
{
	s_world		w;

	world_init(&w, MEMBRANE_ADAPTIVE_OFF);
	w.policy.enabled = 0;
	membrane_adaptive_controller_t	c(make_deps());
	ticks(c, 20, 100);						/* even under critical pressure */
	TEST_ASSERT(w.observe_calls == 0 && w.execute_calls == 0,
		"off: no observation, no mutation");
	json	st = c.status_json();

	TEST_ASSERT(st["effective_mode"] == "off" && st["enabled"] == false,
		"status says off");
}

static void	test_observe_and_recommend_never_execute(void)
{
	s_world		w;

	world_init(&w, MEMBRANE_ADAPTIVE_OBSERVE);
	{
		membrane_adaptive_controller_t	c(make_deps());

		ticks(c, 12, 100);
		TEST_ASSERT(w.observe_calls == 12 && w.execute_calls == 0,
			"observe mode samples only");
		TEST_ASSERT(c.status_json()["decision"] == "observed", "decision");
		TEST_ASSERT(c.status_json()["samples"] == 12, "history kept");
	}
	world_init(&w, MEMBRANE_ADAPTIVE_RECOMMEND);
	{
		membrane_adaptive_controller_t	c(make_deps());

		ticks(c, 12, 100);
		TEST_ASSERT(w.execute_calls == 0, "recommend mode never executes");
		json	st = c.status_json();

		TEST_ASSERT(st["decision"] == "recommend" && st["step"].is_string()
			&& st["reason_chain"].size() >= 3,
			"a concrete, explained suggestion");
	}
}

static void	test_apply_runs_the_j_pipeline_once_per_cooldown(void)
{
	s_world		w;

	world_init(&w, MEMBRANE_ADAPTIVE_APPLY);
	membrane_adaptive_controller_t	c(make_deps());

	ticks(c, 2, 100);
	TEST_ASSERT(w.execute_calls == 0, "not sustained yet");
	ticks(c, 1, 100);
	TEST_ASSERT(w.execute_calls == 1, "sustained => exactly one apply");
	TEST_ASSERT(w.last_req.source == "adaptive"
		&& w.last_req.runtime == "membrane-native"
		&& w.last_req.req.context[0] == '4' , "J request: halve 8192 -> 4096, "
			"source adaptive");
	TEST_ASSERT(w.last_req.audit_extra["adaptive"]["reason_chain"].size() >= 3
		&& w.last_req.audit_extra["adaptive"]["step"].is_string(),
		"the reason chain is handed to the audit record");
	ticks(c, 2, 100);						/* 60 s < 90 s cooldown */
	TEST_ASSERT(w.execute_calls == 1 && c.status_json()["decision"]
		== "cooldown", "cooldown honored");
	json	st = c.status_json();

	TEST_ASSERT(st["last_action"]["applied"] == true
		&& st["last_action"]["status"] == "applied", "last action reported");
	ticks(c, 2, 100);
	TEST_ASSERT(w.execute_calls == 2, "next step after the cooldown");
}

static void	test_failed_apply_still_cools_down(void)
{
	s_world		w;

	world_init(&w, MEMBRANE_ADAPTIVE_APPLY);
	w.exec_ok = false;
	membrane_adaptive_controller_t	c(make_deps());

	ticks(c, 3, 100);
	TEST_ASSERT(w.execute_calls == 1, "attempted");
	ticks(c, 2, 100);
	TEST_ASSERT(w.execute_calls == 1, "a failure is not retried at once");
	TEST_ASSERT(c.status_json()["last_action"]["applied"] == false
		&& c.status_json()["last_action"]["status"] == "rolled_back",
		"failure surfaced as-is (J's rollback status)");
}

static void	test_unobservable_runtime_is_safe(void)
{
	s_world		w;

	world_init(&w, MEMBRANE_ADAPTIVE_APPLY);
	w.observable = false;
	membrane_adaptive_controller_t	c(make_deps());

	ticks(c, 10, 100);
	TEST_ASSERT(w.execute_calls == 0 && c.status_json()["reason"]
		== "RUNTIME_UNOBSERVABLE", "no data => no action");
	w.observable = true;
	w.reachable = false;
	ticks(c, 10, 100);
	TEST_ASSERT(w.execute_calls == 0, "unreachable runtime => no action");
}

static void	test_restart_does_not_replay(void)
{
	s_world		w;

	world_init(&w, MEMBRANE_ADAPTIVE_APPLY);
	{
		membrane_adaptive_controller_t	c(make_deps());

		ticks(c, 3, 100);
		TEST_ASSERT(w.execute_calls == 1, "acted before the 'crash'");
	}
	/* a brand-new controller (service restart): no history, no cooldown
	 * carried over, and it must re-establish sustained pressure itself */
	membrane_adaptive_controller_t	c2(make_deps());

	ticks(c2, 1, 100);
	ticks(c2, 1, 100);
	TEST_ASSERT(w.execute_calls == 1, "nothing replayed on restart");
	TEST_ASSERT(c2.status_json()["samples"] == 2, "history restarts empty");
	ticks(c2, 1, 100);
	TEST_ASSERT(w.execute_calls == 2, "acts again only after a fresh "
		"sustained observation, from the CURRENT state (ctx is 4096 now)");
	TEST_ASSERT(w.last_req.req.context[0] == '2', "next halving from the "
		"real current value (2048), not from stale history");
}

static void	test_manual_change_between_ticks(void)
{
	s_world		w;

	world_init(&w, MEMBRANE_ADAPTIVE_APPLY);
	membrane_adaptive_controller_t	c(make_deps());

	ticks(c, 3, 100);
	TEST_ASSERT(w.execute_calls == 1, "acted");
	w.ctx = "16384";						/* the user raised it again */
	ticks(c, 1, 100);
	TEST_ASSERT(w.execute_calls == 1 && c.status_json()["decision"]
		== "manual_override", "the user's value is adopted, not undone");
	TEST_ASSERT(w.ctx == "16384", "untouched");
}

static void	test_limited_runtime_cannot_act(void)
{
	s_world		w;

	world_init(&w, MEMBRANE_ADAPTIVE_APPLY);
	memset(w.surface.state, 0, sizeof(w.surface.state));
	w.surface.state[MEMBRANE_ACTION_UNLOAD_MODEL] = MEMBRANE_CAPABILITY_SUPPORTED;
	w.residents = {"a", "b"};
	membrane_adaptive_controller_t	c(make_deps());

	ticks(c, 8, 100);
	TEST_ASSERT(w.execute_calls == 0, "an Ollama-like surface offers nothing "
		"the default policy allows");
	TEST_ASSERT(c.status_json()["decision"] == "no_safe_step",
		"reported as 'ask the user', not silently ignored");
}

static void	test_policy_from_config(void)
{
	membrane_server_adaptive_t	c;
	membrane_adaptive_policy_t	p;

	TEST_ASSERT(membrane_adaptive_policy_from_config(c, &p).empty()
		&& membrane_adaptive_effective_mode(&p) == MEMBRANE_ADAPTIVE_OFF,
		"absent block => OFF");
	c.present = true;
	c.enabled = true;
	TEST_ASSERT(membrane_adaptive_policy_from_config(c, &p).empty()
		&& p.mode == MEMBRANE_ADAPTIVE_RECOMMEND, "enabled defaults to "
			"recommend, never apply");
	c.mode = "apply";
	c.allow = {"context", "unload"};
	TEST_ASSERT(membrane_adaptive_policy_from_config(c, &p).empty()
		&& (p.allowed_classes & MEMBRANE_ADAPTIVE_CLASS_UNLOAD), "allow list");
	for (const char *bad : {"turbo", ""})
	{
		c.mode = bad;
		TEST_ASSERT(!membrane_adaptive_policy_from_config(c, &p).empty(),
			"bad mode rejected");
	}
	c.mode = "apply";
	c.allow = {"context", "everything"};
	TEST_ASSERT(!membrane_adaptive_policy_from_config(c, &p).empty(),
		"bad class rejected");
	c.allow = {};
	c.interval_s = 1;
	TEST_ASSERT(!membrane_adaptive_policy_from_config(c, &p).empty(),
		"interval floor");
	c.interval_s = 30;
	TEST_ASSERT(membrane_adaptive_policy_from_config(c, &p).empty()
		&& p.allowed_classes == 0, "empty allow list = nothing may act");
	/* round trip for `membrane adaptive enable` */
	membrane_adaptive_policy_t	d;
	membrane_adaptive_policy_defaults(&d);
	d.enabled = 1;
	d.mode = MEMBRANE_ADAPTIVE_APPLY;
	d.allowed_classes = MEMBRANE_ADAPTIVE_CLASS_CONTEXT
		| MEMBRANE_ADAPTIVE_CLASS_UNLOAD;
	membrane_server_adaptive_t	back = membrane_adaptive_config_from_policy(d);
	membrane_adaptive_policy_t	d2;

	TEST_ASSERT(membrane_adaptive_policy_from_config(back, &d2).empty()
		&& d2.allowed_classes == d.allowed_classes && d2.mode == d.mode,
		"config <-> policy round trip");
}

static void	test_invalid_config_means_off(void)
{
	s_world		w;

	world_init(&w, MEMBRANE_ADAPTIVE_APPLY);
	membrane_adaptive_policy_defaults(&w.policy);		/* what load_policy
								 * returns for an invalid block */
	w.config_error = "adaptive.mode must be observe|recommend|apply";
	membrane_adaptive_controller_t	c(make_deps());

	ticks(c, 10, 100);
	TEST_ASSERT(w.observe_calls == 0 && w.execute_calls == 0,
		"invalid saved policy => OFF (fail closed)");
	TEST_ASSERT(c.status_json()["config_error"].is_string(), "and says why");
}

static void	test_policy_change_takes_effect_without_restart(void)
{
	s_world		w;

	world_init(&w, MEMBRANE_ADAPTIVE_OFF);
	w.policy.enabled = 0;
	membrane_adaptive_controller_t	c(make_deps());

	ticks(c, 5, 100);
	TEST_ASSERT(w.observe_calls == 0, "off");
	w.policy.enabled = 1;
	w.policy.mode = MEMBRANE_ADAPTIVE_RECOMMEND;
	ticks(c, 1, 100);
	TEST_ASSERT(w.observe_calls == 1 && c.status_json()["effective_mode"]
		== "recommend", "enable is picked up on the next tick");
	w.policy.enabled = 0;
	ticks(c, 1, 100);
	TEST_ASSERT(c.status_json()["effective_mode"] == "off", "and disable too");
	w.policy.enabled = 1;
	ticks(c, 1, 100);
	TEST_ASSERT(c.status_json()["samples"] == 1, "re-enabling starts a clean "
		"history (no replay)");
}

static void	test_no_content_in_status(void)
{
	s_world		w;

	world_init(&w, MEMBRANE_ADAPTIVE_APPLY);
	membrane_adaptive_controller_t	c(make_deps());

	ticks(c, 6, 100);
	std::string	dump = c.status_json().dump();

	for (const char *bad : {"prompt", "\"content\"", "message", "token",
			"api_key", "password"})
		TEST_ASSERT(dump.find(bad) == std::string::npos,
			"status carries no prompt/content/secret fields");
}

static void	test_thread_does_not_spin(void)
{
	s_world		w;

	world_init(&w, MEMBRANE_ADAPTIVE_RECOMMEND);
	membrane_adaptive_controller_t	c(make_deps());
	auto		t0 = std::chrono::steady_clock::now();

	c.start();
	/* status is read from this thread while the controller thread runs:
	 * TSan must see no race on the published state */
	for (int i = 0; i < 300; ++i)
	{
		json	st = c.status_json();

		TEST_ASSERT(st.contains("effective_mode"), "status always well-formed");
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	c.stop();
	auto		dt = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - t0).count();

	TEST_ASSERT(w.observe_calls == 1, "one tick on start, then it sleeps "
		"for the interval (no busy loop)");
	TEST_ASSERT(dt < 2000, "stop() wakes the sleeping thread promptly");
	c.stop();						/* idempotent */
	TEST_ASSERT(c.status_json()["running"] == false, "stopped");
}

int	main(void)
{
	test_off_by_default_does_nothing();
	test_observe_and_recommend_never_execute();
	test_apply_runs_the_j_pipeline_once_per_cooldown();
	test_failed_apply_still_cools_down();
	test_unobservable_runtime_is_safe();
	test_restart_does_not_replay();
	test_manual_change_between_ticks();
	test_limited_runtime_cannot_act();
	test_policy_from_config();
	test_invalid_config_means_off();
	test_policy_change_takes_effect_without_restart();
	test_no_content_in_status();
	test_thread_does_not_spin();
	printf("test_adaptive_controller: all tests passed\n");
	return (0);
}
