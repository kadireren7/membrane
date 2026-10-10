#include "adaptive_controller.h"

#include <chrono>
#include <cstring>

#include "advise_cmd.h"
#include "product_cli.h"

using json = nlohmann::json;

/* See adaptive_controller.h's top comment. */

/* ------------------------------------------------------------------ */
/* config <-> policy                                                    */
/* ------------------------------------------------------------------ */

std::string	membrane_adaptive_policy_from_config(
				const membrane_server_adaptive_t &c,
				membrane_adaptive_policy_t *out)
{
	std::string	joined;

	membrane_adaptive_policy_defaults(out);
	if (!c.present)
		return ("");
	out->enabled = c.enabled ? 1 : 0;
	if (!membrane_adaptive_mode_parse(c.mode.c_str(), &out->mode))
		return ("adaptive.mode must be observe|recommend|apply");
	if (c.runtime.empty() || c.runtime.size() >= MEMBRANE_RUNTIME_ID_MAX)
		return ("adaptive.runtime is invalid");
	snprintf(out->runtime_scope, sizeof(out->runtime_scope), "%s",
		c.runtime.c_str());
	out->interval_s = c.interval_s;
	out->cooldown_s = c.cooldown_s;
	out->max_actions_per_window = c.max_actions_per_window;
	out->auto_restore = c.auto_restore ? 1 : 0;
	if (c.min_pressure == "warning")
		out->min_pressure = MEMBRANE_PRESSURE_WARNING;
	else if (c.min_pressure == "critical")
		out->min_pressure = MEMBRANE_PRESSURE_CRITICAL;
	else
		return ("adaptive.min_pressure must be warning|critical");
	for (size_t i = 0; i < c.allow.size(); ++i)
		joined += (i ? "," : "") + c.allow[i];
	if (c.allow.empty())
		out->allowed_classes = 0;
	else
	{
		int	mask = membrane_adaptive_classes_parse(joined.c_str());

		if (mask < 0)
			return ("adaptive.allow has an unknown action class");
		out->allowed_classes = mask;
	}
	return (membrane_adaptive_policy_validate(out));
}

membrane_server_adaptive_t	membrane_adaptive_config_from_policy(
				const membrane_adaptive_policy_t &p)
{
	membrane_server_adaptive_t	c;
	char						buf[128];
	std::string					cur;

	c.present = true;
	c.enabled = p.enabled != 0;
	c.mode = membrane_adaptive_mode_name(p.mode);
	c.runtime = p.runtime_scope;
	c.interval_s = p.interval_s;
	c.cooldown_s = p.cooldown_s;
	c.max_actions_per_window = p.max_actions_per_window;
	c.auto_restore = p.auto_restore != 0;
	c.min_pressure = membrane_pressure_name(p.min_pressure);
	c.allow.clear();
	membrane_adaptive_classes_format(p.allowed_classes, buf, sizeof(buf));
	for (const char *s = buf; *s;)
	{
		const char	*e = strchr(s, ',');

		cur = e ? std::string(s, e - s) : std::string(s);
		c.allow.push_back(cur);
		s = e ? e + 1 : s + strlen(s);
	}
	return (c);
}

/* ------------------------------------------------------------------ */
/* default dependencies                                                 */
/* ------------------------------------------------------------------ */

membrane_adaptive_deps_t	membrane_adaptive_default_deps(void)
{
	membrane_adaptive_deps_t	d;

	d.now_ms = []() -> int64_t
	{
		return ((int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::system_clock::now().time_since_epoch()).count());
	};
	d.observe = [](const std::string &runtime,
			membrane_observation_snapshot_t *snap,
			membrane_memory_intelligence_t *intel, std::string *err) -> bool
	{
		membrane_runtime_capabilities_t	caps;

		return (membrane_advise_collect(runtime, snap, &caps, intel, err));
	};
	d.read_live = [](const std::string &runtime, membrane_apply_live_t *live,
			std::string *err) -> bool
	{
		return (membrane_apply_read_live(runtime, live, err));
	};
	d.surface = [](const std::string &runtime,
			membrane_apply_surface_t *surface) -> bool
	{
		return (membrane_apply_surface(runtime, surface));
	};
	d.load_policy = [](membrane_adaptive_policy_t *p, std::string *cerr)
			-> bool
	{
		std::string						path
				= membrane_server_config_resolve_path();
		membrane_server_config_t		cfg = membrane_server_config_defaults();
		membrane_server_config_error_t	e;

		membrane_adaptive_policy_defaults(p);
		cerr->clear();
		if (path.empty())
			return (true);
		if (!membrane_server_config_load(path, &cfg, &e))
		{
			*cerr = e.message;		/* unreadable config => OFF (fail closed) */
			return (true);
		}
		*cerr = membrane_adaptive_policy_from_config(cfg.adaptive, p);
		if (!cerr->empty())
			membrane_adaptive_policy_defaults(p);	/* invalid => OFF */
		return (true);
	};
	d.execute = [](const membrane_apply_request_t &r,
			membrane_apply_outcome_t *out, std::string *err) -> bool
	{
		return (membrane_apply_execute(r, membrane_apply_io_t(), out, err));
	};
	return (d);
}

/* ------------------------------------------------------------------ */
/* controller                                                           */
/* ------------------------------------------------------------------ */

membrane_adaptive_controller_t::membrane_adaptive_controller_t(
		const membrane_adaptive_deps_t &deps)
	: deps_(deps), state_(new membrane_adaptive_state_t()),
	  state_valid_(false), last_action_(nullptr), interval_s_(30),
	  stop_(false), running_(false)
{
	membrane_adaptive_policy_defaults(&policy_);
	status_ = {{"schema_version", MEMBRANE_ADAPTIVE_SCHEMA_VERSION},
		{"enabled", false}, {"mode", "off"}, {"effective_mode", "off"},
		{"running", false}};
}

membrane_adaptive_controller_t::~membrane_adaptive_controller_t(void)
{
	stop();
	delete state_;
}

json	membrane_adaptive_controller_t::status_json(void)
{
	std::lock_guard<std::mutex>	l(mtx_);
	json						j = status_;

	j["running"] = running_.load();
	return (j);
}

void	membrane_adaptive_controller_t::publish(
			const membrane_adaptive_decision_t &d, const std::string &runtime,
			const json &last_action)
{
	json	chain = json::array();
	json	j;
	char	allow[128];

	for (size_t i = 0; i < d.chain_count; ++i)
		chain.push_back({{"stage", d.chain[i].stage},
			{"text", d.chain[i].text}});
	membrane_adaptive_classes_format(policy_.allowed_classes, allow,
		sizeof(allow));
	j["schema_version"] = MEMBRANE_ADAPTIVE_SCHEMA_VERSION;
	j["enabled"] = policy_.enabled != 0;
	j["mode"] = membrane_adaptive_mode_name(policy_.mode);
	j["effective_mode"] = membrane_adaptive_mode_name(
			membrane_adaptive_effective_mode(&policy_));
	j["runtime"] = runtime;
	j["config_error"] = config_error_.empty() ? json(nullptr)
		: json(config_error_);
	j["last_observation_unix_ms"] = state_->sample_count
		? json(state_->samples[state_->sample_count - 1].t_ms) : json(nullptr);
	j["pressure"] = membrane_pressure_name(state_->pressure);
	j["decision"] = membrane_adaptive_decision_name(d.kind);
	j["reason"] = d.reason[0] ? json(d.reason) : json(nullptr);
	j["step"] = d.step_text[0] ? json(d.step_text) : json(nullptr);
	j["reason_chain"] = chain;
	j["cooldown_remaining_s"] = d.cooldown_remaining_s;
	j["suppressed"] = d.kind == MEMBRANE_DECISION_SUPPRESSED;
	j["suppress_reason"] = d.kind == MEMBRANE_DECISION_SUPPRESSED
		? json(d.reason) : json(nullptr);
	j["pending_reload"] = state_->pending_reload != 0;
	j["deferred"] = d.kind == MEMBRANE_DECISION_DEFERRED;
	j["samples"] = state_->sample_count;
	j["actions_in_history"] = state_->action_count;
	j["last_action"] = last_action;
	j["policy"] = {{"interval_s", policy_.interval_s},
		{"cooldown_s", policy_.cooldown_s}, {"allow", allow},
		{"min_pressure", membrane_pressure_name(policy_.min_pressure)},
		{"max_actions_per_window", policy_.max_actions_per_window},
		{"auto_restore", policy_.auto_restore != 0}};
	std::lock_guard<std::mutex>	l(mtx_);
	status_ = j;
}

void	membrane_adaptive_controller_t::tick(void)
{
	membrane_adaptive_policy_t		p;
	std::string						cerr;
	membrane_adaptive_decision_t	d;
	int64_t							now;

	if (!deps_.load_policy(&p, &cerr))
		membrane_adaptive_policy_defaults(&p);
	config_error_ = cerr;
	if (!state_valid_ || memcmp(&p, &policy_, sizeof(p)) != 0)
	{
		bool	was_off = membrane_adaptive_effective_mode(&policy_)
			== MEMBRANE_ADAPTIVE_OFF;
		bool	scope_changed = strcmp(p.runtime_scope,
				policy_.runtime_scope) != 0;

		policy_ = p;
		interval_s_ = p.interval_s;
		if (!state_valid_ || was_off || scope_changed)
		{
			/* a (re)start of adaptation never replays old history or old
			 * cooldowns: fail closed, re-observe, adopt the current
			 * state as the baseline */
			membrane_adaptive_state_init(state_, deps_.now_ms());
			state_valid_ = true;
		}
	}
	memset(&d, 0, sizeof(d));
	if (membrane_adaptive_effective_mode(&policy_) == MEMBRANE_ADAPTIVE_OFF)
	{
		state_valid_ = false;			/* next enable starts clean */
		publish(d, policy_.runtime_scope, last_action_);
		return ;
	}
	now = deps_.now_ms();

	std::unique_ptr<membrane_observation_snapshot_t>	snap(
			new membrane_observation_snapshot_t());
	std::unique_ptr<membrane_memory_intelligence_t>	intel(
			new membrane_memory_intelligence_t());
	membrane_apply_live_t			live;
	membrane_apply_surface_t		surface;
	membrane_adaptive_sample_t		smp;
	std::string						err;
	std::string						runtime = policy_.runtime_scope;
	bool							observed = deps_.observe(runtime,
			snap.get(), intel.get(), &err);
	bool							have_live;

	memset(&live, 0, sizeof(live));
	memset(&surface, 0, sizeof(surface));
	have_live = observed && deps_.read_live(runtime, &live, &err)
		&& live.runtime_reachable;
	deps_.surface(runtime, &surface);
	if (!observed || !have_live)
	{
		memset(&smp, 0, sizeof(smp));
		smp.t_ms = now;				/* invalid sample: no data, no action */
		membrane_adaptive_record_sample(&policy_, state_, &smp);
		membrane_adaptive_decide(&policy_, state_, now, NULL, NULL, &live,
			&surface, &d);
		publish(d, runtime, last_action_);
		return ;
	}
	membrane_adaptive_make_sample(snap.get(), intel.get(), &live, now, &smp);
	membrane_adaptive_record_sample(&policy_, state_, &smp);
	membrane_adaptive_decide(&policy_, state_, now, snap.get(), intel.get(),
		&live, &surface, &d);
	if (d.kind == MEMBRANE_DECISION_ACT)
	{
		membrane_apply_request_t	rq;
		membrane_apply_outcome_t	out;
		membrane_apply_live_t		after;
		json						chain = json::array();
		bool						ok;

		for (size_t i = 0; i < d.chain_count; ++i)
			chain.push_back({{"stage", d.chain[i].stage},
				{"text", d.chain[i].text}});
		rq.runtime = runtime;
		rq.req = d.request;
		snprintf(rq.req.runtime_id, sizeof(rq.req.runtime_id), "%s",
			runtime.c_str());
		rq.source = "adaptive";
		rq.audit_extra = {{"adaptive", {{"pressure",
			membrane_pressure_name(state_->pressure)},
			{"step", d.step_text}, {"reason_chain", chain}}}};
		ok = deps_.execute(rq, &out, &err) && out.applied;
		memset(&after, 0, sizeof(after));
		if (!deps_.read_live(runtime, &after, &err))
			after = live;
		membrane_adaptive_note_action(&policy_, state_, now, &d, ok ? 1 : 0,
			&after);
		last_action_ = {{"time_unix_ms", now}, {"step", d.step_text},
			{"status", out.status.empty() ? json("failed") : json(out.status)},
			{"applied", ok}, {"detail", out.detail.empty() ? err : out.detail},
			{"pending_reload", state_->pending_reload != 0}};
	}
	publish(d, runtime, last_action_);
}

void	membrane_adaptive_controller_t::run(void)
{
	while (!stop_.load())
	{
		int	wait_s;

		tick();
		wait_s = membrane_adaptive_effective_mode(&policy_)
			== MEMBRANE_ADAPTIVE_OFF ? 30 : interval_s_;
		if (wait_s < MEMBRANE_ADAPTIVE_MIN_INTERVAL_S)
			wait_s = MEMBRANE_ADAPTIVE_MIN_INTERVAL_S;
		std::unique_lock<std::mutex>	l(wake_mtx_);

		wake_cv_.wait_for(l, std::chrono::seconds(wait_s),
			[this] { return (stop_.load()); });
	}
}

void	membrane_adaptive_controller_t::start(void)
{
	if (running_.load())
		return ;
	stop_.store(false);
	running_ = true;
	thread_ = std::thread([this] { run(); });
}

void	membrane_adaptive_controller_t::stop(void)
{
	if (!running_.load())
		return ;
	stop_.store(true);
	{
		std::lock_guard<std::mutex>	l(wake_mtx_);
	}
	wake_cv_.notify_all();
	if (thread_.joinable())
		thread_.join();
	running_ = false;
}
