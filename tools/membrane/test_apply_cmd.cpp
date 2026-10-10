#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "apply_cmd.h"
#include "product_cli.h"
#include "runtime_ollama.h"
#include "server_config.h"
#include "test_helpers.h"

using json = nlohmann::json;

/*
 * Milestone J: `membrane apply`'s CLI-level tests. The compiler and the
 * transaction engine are covered host-independently by tools/membrane-run/
 * test_action.c; this file proves the wiring against REAL HTTP clients
 * talking to in-process mock servers (a fake native service and a fake
 * Ollama daemon) -- never a real service, never a real Ollama, never a
 * model load or an inference. Intelligence and the Planner v2 check are
 * injected through membrane_apply_io_t so results do not depend on this
 * host's memory.
 */

/* ---------------------------------------------------------------- */
/* helpers                                                           */
/* ---------------------------------------------------------------- */

static std::string	read_file(const std::string &path)
{
	std::ifstream		f(path, std::ios::binary);
	std::ostringstream	ss;

	ss << f.rdbuf();
	return (ss.str());
}

static bool	file_exists(const std::string &p)
{
	struct stat	sb;

	return (stat(p.c_str(), &sb) == 0);
}

static std::string	g_dir;

static std::string	capture(const std::vector<std::string> &args,
						bool want_json, const membrane_apply_io_t &io, int *rc)
{
	std::string	tmpl = g_dir + "/capture-XXXXXX";
	int			fd = mkstemp(&tmpl[0]);

	TEST_ASSERT(fd >= 0, "capture file");
	fflush(stdout);
	int	saved = dup(STDOUT_FILENO);

	dup2(fd, STDOUT_FILENO);
	close(fd);
	*rc = membrane_apply_cmd_dispatch_ex(args, want_json, io);
	fflush(stdout);
	dup2(saved, STDOUT_FILENO);
	close(saved);
	std::string	out = read_file(tmpl);

	unlink(tmpl.c_str());
	return (out);
}

/* ---------------------------------------------------------------- */
/* Mock native service                                               */
/* ---------------------------------------------------------------- */

typedef struct s_mres
{
	std::string	name;
	bool		pinned;
	bool		busy;
	uint64_t	bytes;
}	mres_t;

static httplib::Server			*g_native;
static std::thread				*g_native_thread;
static int						g_native_port;
static std::mutex				g_mtx;
static std::vector<mres_t>		g_res;
static std::vector<std::string>	g_log;			/* "METHOD path body" */
static std::string				g_fail_route;	/* e.g. "pin" */
static int						g_trap_hits;

static bool	failing(const char *route)
{
	return (g_fail_route.find(route) != std::string::npos);
}

static std::vector<std::string>	log_copy(void)
{
	std::lock_guard<std::mutex>	l(g_mtx);

	return (g_log);
}

static size_t	count_prefix(const std::string &prefix)
{
	size_t	n = 0;

	for (const auto &s : log_copy())
		if (s.compare(0, prefix.size(), prefix) == 0)
			n++;
	return (n);
}

static void	err_reply(httplib::Response &res, int status, const char *code)
{
	res.status = status;
	res.set_content(json{{"ok", false}, {"error", {{"code", code},
		{"message", code}}}}.dump(), "application/json");
}

static mres_t	*find_res(const std::string &n)
{
	for (auto &r : g_res)
		if (r.name == n)
			return (&r);
	return (NULL);
}

static void	start_native(void)
{
	g_native = new httplib::Server();
	g_native->Get("/v1/status", [](const httplib::Request &req,
			httplib::Response &res)
	{
		std::lock_guard<std::mutex>	l(g_mtx);
		json						arr = json::array();

		g_log.push_back("GET /v1/status");
		for (auto &r : g_res)
			arr.push_back({{"model", r.name}, {"state", "ready"},
				{"pinned", r.pinned},
				{"evictable", !r.pinned && !r.busy},
				{"estimated_model_bytes", r.bytes}});
		res.set_content(json{{"running", true}, {"resident_models", arr},
			{"default_model", "main"}}.dump(), "application/json");
		(void)req;
	});
	auto	body_model = [](const httplib::Request &req) -> std::string
	{
		json	j = json::parse(req.body, nullptr, false);

		return (j.is_object() && j.contains("model") ? j["model"]
			.get<std::string>() : "");
	};
	g_native->Post("/membrane/v1/models/activate", [body_model](
			const httplib::Request &req, httplib::Response &res)
	{
		std::lock_guard<std::mutex>	l(g_mtx);
		std::string					m = body_model(req);

		g_log.push_back("POST activate " + m);
		if (failing("activate"))
		{
			err_reply(res, 500, "MODEL_LOAD_FAILED");
			return ;
		}
		if (find_res(m) == NULL)
			g_res.push_back({m, false, false, 1000});
		res.set_content(json{{"ok", true}, {"already_active", false},
			{"active_model", m}}.dump(), "application/json");
	});
	g_native->Post("/membrane/v1/models/pin", [body_model](
			const httplib::Request &req, httplib::Response &res)
	{
		std::lock_guard<std::mutex>	l(g_mtx);
		std::string					m = body_model(req);
		mres_t						*r = find_res(m);

		g_log.push_back("POST pin " + m);
		if (failing("pin") || r == NULL)
		{
			err_reply(res, 409, "MODEL_NOT_RESIDENT");
			return ;
		}
		r->pinned = true;
		res.set_content(json{{"ok", true}, {"model", m}, {"pinned", true}}
			.dump(), "application/json");
	});
	g_native->Post("/membrane/v1/models/unpin", [body_model](
			const httplib::Request &req, httplib::Response &res)
	{
		std::lock_guard<std::mutex>	l(g_mtx);
		std::string					m = body_model(req);
		mres_t						*r = find_res(m);

		g_log.push_back("POST unpin " + m);
		if (r == NULL)
		{
			err_reply(res, 409, "MODEL_NOT_RESIDENT");
			return ;
		}
		r->pinned = false;
		res.set_content(json{{"ok", true}, {"model", m}, {"pinned", false}}
			.dump(), "application/json");
	});
	g_native->Post("/membrane/v1/models/unload", [body_model](
			const httplib::Request &req, httplib::Response &res)
	{
		std::lock_guard<std::mutex>	l(g_mtx);
		std::string					m = body_model(req);
		mres_t						*r = find_res(m);

		g_log.push_back("POST unload " + m);
		if (failing("unload"))
		{
			err_reply(res, 500, "UNLOAD_FAILED");
			return ;
		}
		if (r == NULL)
		{
			err_reply(res, 409, "MODEL_NOT_RESIDENT");
			return ;
		}
		if (r->pinned)
		{
			err_reply(res, 409, "MODEL_PINNED");
			return ;
		}
		if (r->busy)
		{
			err_reply(res, 409, "MODEL_BUSY");
			return ;
		}
		for (size_t i = 0; i < g_res.size(); ++i)
			if (g_res[i].name == m)
				g_res.erase(g_res.begin() + (long)i);
		res.set_content(json{{"ok", true}, {"unloaded", true}}.dump(),
			"application/json");
	});
	/* Any inference route is a trap: apply must never reach one. */
	g_native->Post("/v1/chat/completions", [](const httplib::Request &,
			httplib::Response &res)
	{
		std::lock_guard<std::mutex>	l(g_mtx);

		g_trap_hits++;
		g_log.push_back("TRAP chat");
		err_reply(res, 500, "TRAP");
	});
	g_native_port = g_native->bind_to_any_port("127.0.0.1");
	TEST_ASSERT(g_native_port > 0, "native mock bound");
	g_native_thread = new std::thread([]() { g_native->listen_after_bind(); });
	for (int i = 0; i < 100 && !g_native->is_running(); ++i)
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	TEST_ASSERT(g_native->is_running(), "native mock ready");
}

/* ---------------------------------------------------------------- */
/* Mock Ollama                                                       */
/* ---------------------------------------------------------------- */

static httplib::Server			*g_oll;
static std::thread				*g_oll_thread;
static int						g_oll_port;
static std::vector<std::string>	g_oll_models;		/* loaded */
static std::vector<std::string>	g_oll_log;			/* "METHOD path body" */
static bool						g_oll_stuck;		/* unload accepted, no effect */

static void	start_ollama(void)
{
	g_oll = new httplib::Server();
	g_oll->set_logger([](const httplib::Request &req,
			const httplib::Response &)
	{
		std::lock_guard<std::mutex>	l(g_mtx);

		g_oll_log.push_back(req.method + " " + req.path + " " + req.body);
	});
	g_oll->Get("/api/version", [](const httplib::Request &,
			httplib::Response &res)
	{
		res.set_content("{\"version\":\"0.34.3\"}", "application/json");
	});
	g_oll->Get("/api/ps", [](const httplib::Request &, httplib::Response &res)
	{
		std::lock_guard<std::mutex>	l(g_mtx);
		json						arr = json::array();

		for (auto &m : g_oll_models)
			arr.push_back({{"name", m}, {"model", m}, {"size", 5000},
				{"size_vram", 0}, {"digest", "abc"}, {"context_length", 4096}});
		res.set_content(json{{"models", arr}}.dump(), "application/json");
	});
	g_oll->Post("/api/generate", [](const httplib::Request &req,
			httplib::Response &res)
	{
		std::lock_guard<std::mutex>	l(g_mtx);
		json						j = json::parse(req.body, nullptr, false);

		if (j.is_object() && j.value("keep_alive", -1) == 0 && !g_oll_stuck)
		{
			std::string	m = j.value("model", std::string());

			for (size_t i = 0; i < g_oll_models.size(); ++i)
				if (g_oll_models[i] == m)
					g_oll_models.erase(g_oll_models.begin() + (long)i);
		}
		res.set_content("{\"done\":true,\"done_reason\":\"unload\"}",
			"application/json");
	});
	for (const char *p : {"/api/chat", "/api/pull", "/api/create",
			"/api/copy", "/api/push", "/api/embed", "/v1/chat/completions"})
		g_oll->Post(p, [](const httplib::Request &, httplib::Response &res)
		{
			err_reply(res, 500, "TRAP");
		});
	g_oll->Delete("/api/delete", [](const httplib::Request &,
			httplib::Response &res) { err_reply(res, 500, "TRAP"); });
	g_oll_port = g_oll->bind_to_any_port("127.0.0.1");
	TEST_ASSERT(g_oll_port > 0, "ollama mock bound");
	g_oll_thread = new std::thread([]() { g_oll->listen_after_bind(); });
	for (int i = 0; i < 100 && !g_oll->is_running(); ++i)
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	TEST_ASSERT(g_oll->is_running(), "ollama mock ready");
}

/* ---------------------------------------------------------------- */
/* fixtures                                                          */
/* ---------------------------------------------------------------- */

static std::string	cfg_path(void) { return (g_dir + "/server.json"); }
static std::string	audit_path(void) { return (g_dir + "/audit.jsonl"); }

static void	write_cfg(const std::string &listen = "127.0.0.1",
				const std::string &extra = "")
{
	std::string	body = "{\"schema_version\":1,\"listen_address\":\"" + listen
		+ "\",\"port\":" + std::to_string(g_native_port)
		+ ",\"default_model\":\"main\"" + extra + "}";
	std::ofstream	f(cfg_path(), std::ios::binary | std::ios::trunc);

	f << body;
}

static void	reset_world(void)
{
	std::lock_guard<std::mutex>	l(g_mtx);

	g_res.clear();
	g_res.push_back({"main", false, false, 9000});
	g_res.push_back({"spare", false, false, 3000});
	g_log.clear();
	g_oll_log.clear();
	g_fail_route.clear();
	g_trap_hits = 0;
	g_oll_stuck = false;
	g_oll_models = {"qwen:7b", "tiny:1b"};
	unlink(audit_path().c_str());
	unlink((audit_path() + ".1").c_str());
}

static membrane_memory_intelligence_t	g_intel;
static int								g_intel_calls;
static bool								g_intel_drop_after_first;

static void	set_recs(std::vector<std::string> codes, const char *runtime)
{
	memset(&g_intel, 0, sizeof(g_intel));
	g_intel.schema_version = 1;
	snprintf(g_intel.runtime_id, sizeof(g_intel.runtime_id), "%s", runtime);
	for (const auto &c : codes)
	{
		membrane_intel_recommendation_t	*r
			= &g_intel.recommendations[g_intel.recommendation_count++];

		snprintf(r->code, sizeof(r->code), "%s", c.c_str());
		snprintf(r->affected_runtime, sizeof(r->affected_runtime), "%s",
			runtime);
		r->applicability = MEMBRANE_INTEL_APPLICABILITY_CONTROLLABLE;
	}
	g_intel_calls = 0;
	g_intel_drop_after_first = false;
}

static membrane_apply_io_t	make_io(bool interactive = false)
{
	membrane_apply_io_t	io;

	io.interactive_known = true;
	io.interactive = interactive;
	io.audit_path = audit_path();
	io.intelligence = [](const std::string &, membrane_memory_intelligence_t *i,
			std::string *) -> bool
	{
		*i = g_intel;
		if (g_intel_drop_after_first && g_intel_calls >= 1)
			i->recommendation_count = 0;
		g_intel_calls++;
		return (true);
	};
	io.planner_check = [](const std::string &, const membrane_plan_override_request_t &ov,
			membrane_plan_override_check_t *c) -> bool
	{
		c->checked = true;
		c->feasible = !(ov.has_ctx && ov.ctx > 100000);
		c->detail = c->feasible ? "Planner v2: feasible"
			: "Planner v2: not feasible (HOST_MEMORY_LIMIT)";
		return (true);
	};
	return (io);
}

static void	env_setup(void)
{
	char	tmpl[] = "/tmp/membrane-apply-test-XXXXXX";

	TEST_ASSERT(mkdtemp(tmpl) != NULL, "temp dir");
	g_dir = tmpl;
	setenv("MEMBRANE_SERVER_CONFIG_PATH", cfg_path().c_str(), 1);
	setenv("MEMBRANE_APPLY_AUDIT_PATH", audit_path().c_str(), 1);
	setenv("MEMBRANE_OLLAMA_ENDPOINT", ("http://127.0.0.1:"
		+ std::to_string(g_oll_port)).c_str(), 1);
}

static json	run_json(const std::vector<std::string> &args,
				const membrane_apply_io_t &io, int *rc)
{
	std::string	out = capture(args, true, io, rc);
	json		j = json::parse(out, nullptr, false);

	TEST_ASSERT(!j.is_discarded(), ("stdout is one JSON document: " + out)
		.c_str());
	return (j);
}

static membrane_server_overrides_t	cfg_overrides(void)
{
	membrane_server_config_t		c;
	membrane_server_config_error_t	e;

	TEST_ASSERT(membrane_server_config_load(cfg_path(), &c, &e), "cfg loads");
	return (c.overrides);
}

/* ---------------------------------------------------------------- */
/* tests                                                             */
/* ---------------------------------------------------------------- */

static void	test_dry_run_never_mutates(void)
{
	int			rc;
	std::string	before;
	json		j;

	reset_world();
	write_cfg();
	set_recs({"REDUCE_CONTEXT", "UNLOAD_UNUSED_MODEL"}, "membrane-native");
	before = read_file(cfg_path());
	j = run_json({}, make_io(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "plain dry run exits 0");
	TEST_ASSERT(j["dry_run"] == true && j["mutates_state"] == false
		&& j["applied"] == false, "dry_run/mutates/applied flags");
	TEST_ASSERT(j["mode"] == "apply" && j["runtime"] == "membrane-native",
		"shape");
	TEST_ASSERT(j["actions"].size() == 1 && j["actions"][0]["code"]
		== "UNLOAD_MODEL" && j["actions"][0]["model"] == "spare"
		&& j["actions"][0]["mutates_state"] == true
		&& j["actions"][0]["origin"] == "recommendation",
		"unload candidate proposed");
	bool	ctx_advisory = false;

	for (const auto &u : j["unapplied"])
		if (u["source"] == "REDUCE_CONTEXT"
			&& u["reason_code"] == "NO_DETERMINISTIC_TARGET")
			ctx_advisory = true;
	TEST_ASSERT(ctx_advisory, "REDUCE_CONTEXT stays advisory, never guessed");
	TEST_ASSERT(read_file(cfg_path()) == before, "config untouched");
	TEST_ASSERT(count_prefix("POST") == 0, "no mutating call was sent");
	TEST_ASSERT(!file_exists(audit_path()), "dry run writes no audit record");
	TEST_ASSERT(g_trap_hits == 0, "no inference");

	j = run_json({"--context", "2048", "--kv", "q8"}, make_io(), &rc);
	TEST_ASSERT(j["dry_run"] == true && j["actions"].size() == 3
		&& count_prefix("POST") == 0, "explicit requests also dry-run");
}

static void	test_non_tty_requires_yes(void)
{
	int		rc;
	json	j;

	reset_world();
	write_cfg();
	set_recs({}, "membrane-native");
	std::string	before = read_file(cfg_path());

	j = run_json({"--context", "2048"}, make_io(false), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "mutation w/o --yes off a TTY "
		"fails closed (exit 2)");
	TEST_ASSERT(j["applied"] == false && j["mutates_state"] == false
		&& j["confirmation"] == "not_interactive", "reported as refused");
	TEST_ASSERT(read_file(cfg_path()) == before && count_prefix("POST") == 0,
		"nothing changed");
}

static void	test_interactive_confirmation(void)
{
	int			rc;
	std::string	out;
	bool		asked = false;

	reset_world();
	write_cfg();
	set_recs({}, "membrane-native");
	membrane_apply_io_t	io = make_io(true);

	io.confirm = [&](const std::string &) { asked = true; return (false); };
	out = capture({"--context", "2048"}, false, io, &rc);
	TEST_ASSERT(asked && rc == MEMBRANE_EXIT_SUCCESS, "prompt shown, declined");
	TEST_ASSERT(!cfg_overrides().has_context, "declined => unchanged");
	TEST_ASSERT(out.find("Declined") != std::string::npos, "says declined");
	io.confirm = [&](const std::string &) { return (true); };
	out = capture({"--context", "2048"}, false, io, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS && cfg_overrides().has_context
		&& cfg_overrides().context == 2048, "accepted => applied");
	/* --json never prompts, even on a TTY */
	reset_world();
	asked = false;
	io.confirm = [&](const std::string &) { asked = true; return (true); };
	json	j = run_json({"--context", "1024"}, io, &rc);
	TEST_ASSERT(!asked && j["applied"] == false, "--json never prompts");
}

static void	test_native_setting_actions(void)
{
	int		rc;
	json	j;

	reset_world();
	write_cfg();
	set_recs({}, "membrane-native");
	j = run_json({"--context", "4096", "--kv", "q8", "--kv-placement", "cpu",
		"--gpu-layers", "20", "--yes"}, make_io(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS && j["status"] == "applied"
		&& j["applied"] == true && j["dry_run"] == false
		&& j["mutates_state"] == true, "applied");
	TEST_ASSERT(j["actions"].size() == 4, "four settings");
	TEST_ASSERT(j["actions"][0]["code"] == "SET_CONTEXT"
		&& j["actions"][1]["code"] == "SET_KV_PRECISION"
		&& j["actions"][2]["code"] == "SET_KV_PLACEMENT"
		&& j["actions"][3]["code"] == "SET_GPU_LAYERS",
		"deterministic order");
	TEST_ASSERT(j["requires_reload"] == true, "reload disclosed");
	membrane_server_overrides_t	o = cfg_overrides();

	TEST_ASSERT(o.has_context && o.context == 4096 && o.has_kv_precision
		&& o.kv_precision == "q8" && o.has_kv_placement
		&& o.kv_placement == "cpu" && o.has_gpu_layers && o.gpu_layers == 20,
		"persisted overrides");
	TEST_ASSERT(count_prefix("POST") == 0, "settings need no admin call");
	std::string	audit = read_file(audit_path());

	TEST_ASSERT(!audit.empty() && audit.find("SET_CONTEXT") != std::string::npos
		&& audit.find("\"before\":\"automatic\"") != std::string::npos
		&& audit.find("\"after\":\"4096\"") != std::string::npos
		&& audit.find("prompt") == std::string::npos, "audit record");
	/* back to automatic */
	j = run_json({"--context", "auto", "--kv", "auto", "--kv-placement",
		"auto", "--gpu-layers", "auto", "--yes"}, make_io(), &rc);
	TEST_ASSERT(j["status"] == "applied" && membrane_server_overrides_empty(
		cfg_overrides()), "auto clears every override");
}

static void	test_planner_rejection(void)
{
	int		rc;
	json	j;

	reset_world();
	write_cfg();
	set_recs({}, "membrane-native");
	std::string	before = read_file(cfg_path());

	j = run_json({"--context", "9999999", "--yes"}, make_io(), &rc);
	TEST_ASSERT(j["actions"].size() == 0 && j["status"] == "nothing_to_apply",
		"Planner v2 says infeasible => no action");
	TEST_ASSERT(j["unapplied"][0]["reason_code"] == "PLANNER_VALIDATION_FAILED",
		"reason");
	TEST_ASSERT(read_file(cfg_path()) == before, "nothing written");
	membrane_apply_io_t	io = make_io();

	io.planner_check = [](const std::string &, const membrane_plan_override_request_t &,
			membrane_plan_override_check_t *) { return (false); };
	j = run_json({"--context", "2048", "--yes"}, io, &rc);
	TEST_ASSERT(j["actions"].size() == 0, "validation unavailable => refused "
		"(fail closed)");
}

static void	test_unload_flow_and_protections(void)
{
	int		rc;
	json	j;

	reset_world();
	write_cfg();
	set_recs({"UNLOAD_UNUSED_MODEL"}, "membrane-native");
	j = run_json({"--yes"}, make_io(), &rc);
	TEST_ASSERT(j["status"] == "applied" && count_prefix("POST unload spare")
		== 1 && count_prefix("POST unload main") == 0, "safe unload");
	TEST_ASSERT(find_res("spare") == NULL && find_res("main") != NULL,
		"only the candidate left");
	/* pinned candidate => no auto action */
	reset_world();
	g_res[1].pinned = true;
	j = run_json({}, make_io(), &rc);
	TEST_ASSERT(j["actions"].size() == 0 && j["unapplied"][0]["reason_code"]
		== "NO_SAFE_CANDIDATE", "pinned model never auto-selected");
	j = run_json({"--unload", "spare", "--yes"}, make_io(), &rc);
	TEST_ASSERT(count_prefix("POST unload") == 0 && j["actions"].size() == 0,
		"explicit unload of a pinned model refused before any call");
	/* generating candidate */
	reset_world();
	g_res[1].busy = true;
	j = run_json({"--unload", "spare", "--yes"}, make_io(), &rc);
	TEST_ASSERT(count_prefix("POST unload") == 0 && j["actions"].size() == 0,
		"generating model never unloaded");
	/* the default model is never an automatic candidate */
	reset_world();
	g_res.erase(g_res.begin() + 1);
	j = run_json({}, make_io(), &rc);
	TEST_ASSERT(j["actions"].size() == 0, "default model not auto-unloaded");
}

static void	test_stale_refuses(void)
{
	int		rc;
	json	j;

	/* (a) pressure disappears between plan and apply */
	reset_world();
	write_cfg();
	set_recs({"UNLOAD_UNUSED_MODEL"}, "membrane-native");
	g_intel_drop_after_first = true;
	j = run_json({"--yes"}, make_io(), &rc);
	TEST_ASSERT(j["status"] == "stale_precondition" && j["applied"] == false
		&& count_prefix("POST") == 0, "recommendation gone => refused");
	TEST_ASSERT(std::string(j["detail"]).find("rerun") != std::string::npos,
		"tells the user to rerun");
	/* (b) the live state changes while the user is deciding */
	reset_world();
	set_recs({}, "membrane-native");
	membrane_apply_io_t	io = make_io(true);

	io.confirm = [&](const std::string &)
	{
		std::lock_guard<std::mutex>	l(g_mtx);

		g_res[1].busy = true;			/* a generation starts */
		return (true);
	};
	std::string	out = capture({"--unload", "spare"}, false, io, &rc);

	TEST_ASSERT(out.find("stale_precondition") != std::string::npos
		&& count_prefix("POST unload") == 0 && find_res("spare") != NULL,
		"live drift after the plan => refused, nothing unloaded");
	/* (c) config changed underneath */
	reset_world();
	write_cfg();
	set_recs({}, "membrane-native");
	io = make_io(true);
	io.confirm = [&](const std::string &)
	{
		write_cfg("127.0.0.1", ",\"overrides\":{\"context\":512}");
		return (true);
	};
	out = capture({"--kv", "q8"}, false, io, &rc);
	TEST_ASSERT(out.find("stale_precondition") != std::string::npos
		&& !cfg_overrides().has_kv_precision, "config drift => refused");
}

static void	test_partial_failure_and_rollback(void)
{
	int		rc;
	json	j;

	/* context applied, then pin fails => context rolled back */
	reset_world();
	write_cfg();
	set_recs({}, "membrane-native");
	g_fail_route = "pin";
	j = run_json({"--context", "2048", "--pin", "main", "--yes"}, make_io(),
		&rc);
	TEST_ASSERT(j["status"] == "rolled_back" && rc != MEMBRANE_EXIT_SUCCESS,
		"rolled back");
	TEST_ASSERT(!cfg_overrides().has_context, "setting restored");
	TEST_ASSERT(j["result"]["actions"][0]["outcome"] == "rolled_back"
		&& j["result"]["actions"][1]["outcome"] == "failed",
		"per-action outcomes");
	/* unload applied, then pin fails => the unload is undone by reloading */
	reset_world();
	g_fail_route = "pin";
	j = run_json({"--unload", "spare", "--pin", "main", "--yes"}, make_io(),
		&rc);
	TEST_ASSERT(j["status"] == "rolled_back" && j["result"]["actions"][0]
		["outcome"] == "rolled_back" && count_prefix("POST activate spare")
		== 1 && find_res("spare") != NULL, "unload undone by reloading");
	/* the reload itself fails => rollback_failed is reported, not hidden */
	reset_world();
	g_fail_route = "pin,activate";
	j = run_json({"--unload", "spare", "--pin", "main", "--yes"}, make_io(),
		&rc);
	TEST_ASSERT(j["status"] == "rollback_failed" && j["result"]["actions"][0]
		["outcome"] == "rollback_failed" && !j["result"]["actions"][0]
			["error"].is_null(), "rollback failure is reported");
	TEST_ASSERT(j["result"]["actions"][1]["outcome"] == "failed",
		"the original failure is still shown");
	/* an irreversible action stays applied => partially_applied */
	reset_world();
	write_cfg();
	g_fail_route = "pin";
	j = run_json({"--activate", "extra", "--pin", "main", "--yes"}, make_io(),
		&rc);
	TEST_ASSERT(j["status"] == "partially_applied" && find_res("extra")
		!= NULL, "activate (not reversible) is reported as still applied");
}

static void	test_deterministic_json(void)
{
	int			rc;
	std::string	a;
	std::string	b;

	reset_world();
	write_cfg();
	set_recs({"UNLOAD_UNUSED_MODEL", "REDUCE_CONTEXT"}, "membrane-native");
	a = capture({"--kv", "q8"}, true, make_io(), &rc);
	b = capture({"--kv", "q8"}, true, make_io(), &rc);
	TEST_ASSERT(!a.empty() && a == b, "byte-identical JSON for identical "
		"inputs");
}

static void	test_native_non_loopback_refused(void)
{
	int		rc;
	json	j;

	reset_world();
	write_cfg("0.0.0.0");
	set_recs({}, "membrane-native");
	j = run_json({"--context", "2048", "--yes"}, make_io(), &rc);
	TEST_ASSERT(j["status"] == "refused" && rc != MEMBRANE_EXIT_SUCCESS
		&& count_prefix("GET") == 0 && count_prefix("POST") == 0,
		"a non-loopback listen address is never contacted or controlled");
	write_cfg();
}

static void	test_runtime_unreachable(void)
{
	int		rc;
	json	j;

	reset_world();
	{
		std::ofstream	f(cfg_path(), std::ios::trunc);

		f << "{\"schema_version\":1,\"listen_address\":\"127.0.0.1\","
			"\"port\":1,\"default_model\":\"main\"}";
	}
	set_recs({}, "membrane-native");
	j = run_json({"--context", "2048", "--yes"}, make_io(), &rc);
	TEST_ASSERT(j["status"] == "refused" && j["applied"] == false
		&& rc == MEMBRANE_EXIT_RUNTIME_ERROR, "unavailable runtime => "
			"refused");
	TEST_ASSERT(!file_exists(audit_path()), "nothing happened, nothing logged");
	write_cfg();
}

static void	test_ollama_unload(void)
{
	int		rc;
	json	j;

	reset_world();
	set_recs({"UNLOAD_UNUSED_MODEL"}, "ollama");
	/* auto candidate: Ollama does not report activity => never chosen */
	j = run_json({"--runtime", "ollama"}, make_io(), &rc);
	TEST_ASSERT(j["actions"].size() == 0 && j["unapplied"][0]["reason_code"]
		== "NO_SAFE_CANDIDATE", "no automatic Ollama unload candidate");
	/* explicit unload, dry run */
	j = run_json({"--runtime", "ollama", "--unload", "qwen:7b"}, make_io(),
		&rc);
	TEST_ASSERT(j["actions"].size() == 1 && j["actions"][0]["reversible"]
		== false && j["dry_run"] == true, "proposed, not reversible");
	for (const auto &l : g_oll_log)
		TEST_ASSERT(l.compare(0, 3, "GET") == 0, "dry run sent only GETs");
	/* explicit unload, applied */
	j = run_json({"--runtime", "ollama", "--unload", "qwen:7b", "--yes"},
		make_io(), &rc);
	TEST_ASSERT(j["status"] == "applied" && rc == MEMBRANE_EXIT_SUCCESS,
		"applied");
	TEST_ASSERT(g_oll_models.size() == 1 && g_oll_models[0] == "tiny:1b",
		"only the named model unloaded");
	size_t	posts = 0;

	for (const auto &l : g_oll_log)
		if (l.compare(0, 4, "POST") == 0)
		{
			posts++;
			TEST_ASSERT(l == "POST /api/generate "
				+ membrane_ollama_make_unload_body("qwen:7b"),
				"the only POST is the exact keep_alive=0 unload (no prompt)");
		}
	TEST_ASSERT(posts == 1, "exactly one mutating request");
	/* accepted but ineffective => reported as failure */
	reset_world();
	g_oll_stuck = true;
	j = run_json({"--runtime", "ollama", "--unload", "qwen:7b", "--yes"},
		make_io(), &rc);
	TEST_ASSERT(j["status"] == "failed" && j["applied"] == false,
		"effect is verified, not assumed");
	/* not loaded => rejected, no request */
	reset_world();
	j = run_json({"--runtime", "ollama", "--unload", "ghost:1b", "--yes"},
		make_io(), &rc);
	TEST_ASSERT(j["actions"].size() == 0 && count_prefix("POST") == 0, "x");
	for (const auto &l : g_oll_log)
		TEST_ASSERT(l.compare(0, 4, "POST") != 0, "no POST for an unloaded "
			"model");
}

static void	test_ollama_unsupported_controls(void)
{
	int		rc;
	json	j;

	reset_world();
	set_recs({"REDUCE_CONTEXT", "USE_LOWER_KV_PRECISION",
		"CHANGE_KV_PLACEMENT", "REDUCE_GPU_LAYERS"}, "ollama");
	j = run_json({"--runtime", "ollama", "--context", "4096", "--kv", "q8",
		"--kv-placement", "cpu", "--gpu-layers", "10", "--pin", "qwen:7b",
		"--unpin", "qwen:7b", "--activate", "other:1b", "--yes"}, make_io(),
		&rc);
	TEST_ASSERT(j["actions"].size() == 0 && j["status"] == "nothing_to_apply",
		"no Ollama setting is controllable => no action objects");
	size_t	unsupported = 0;

	for (const auto &u : j["unapplied"])
		if (u["reason_code"] == "RUNTIME_CANNOT_CONTROL")
			unsupported++;
	TEST_ASSERT(unsupported >= 7, "each request explained as unsupported");
	TEST_ASSERT(j["unapplied"].dump().find("per-request") != std::string::npos
		, "distinguishes per-request from persistent control");
	for (const auto &l : g_oll_log)
		TEST_ASSERT(l.compare(0, 3, "GET") == 0, "route trap: nothing but "
			"GETs reached the daemon");
}

static void	test_audit_rotation_and_content(void)
{
	int		rc;

	reset_world();
	write_cfg();
	set_recs({}, "membrane-native");
	{
		std::ofstream	f(audit_path(), std::ios::binary);

		f << std::string(1024 * 1024 + 10, 'x');
	}
	run_json({"--context", "2048", "--yes"}, make_io(), &rc);
	TEST_ASSERT(file_exists(audit_path() + ".1"), "oversized log rotated");
	std::string	line = read_file(audit_path());
	json		rec = json::parse(line, nullptr, false);

	TEST_ASSERT(!rec.is_discarded() && rec["runtime"] == "membrane-native"
		&& rec["success"] == true && rec["actions"][0]["before"]
			== "automatic" && rec["actions"][0]["after"] == "2048"
		&& rec.contains("timestamp"), "record fields");
	TEST_ASSERT(line.find("prompt") == std::string::npos
		&& line.find("content") == std::string::npos, "no prompt/content");
	struct stat	sb;

	TEST_ASSERT(stat(audit_path().c_str(), &sb) == 0
		&& (sb.st_mode & 0777) == 0600, "audit file is private (0600)");
}

static void	test_cli_errors(void)
{
	int		rc;
	json	j;

	reset_world();
	write_cfg();
	set_recs({}, "membrane-native");
	j = run_json({"--bogus"}, make_io(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR && j["ok"] == false, "unknown "
		"option");
	j = run_json({"--context", "-5"}, make_io(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "bad context value");
	j = run_json({"--runtime", "vllm"}, make_io(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "unsupported runtime");
	j = run_json({"--kv", "q9"}, make_io(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "bad kv");
}

int	main(void)
{
	start_native();
	start_ollama();
	env_setup();
	test_dry_run_never_mutates();
	test_non_tty_requires_yes();
	test_interactive_confirmation();
	test_native_setting_actions();
	test_planner_rejection();
	test_unload_flow_and_protections();
	test_stale_refuses();
	test_native_non_loopback_refused();
	test_runtime_unreachable();
	test_partial_failure_and_rollback();
	test_deterministic_json();
	test_ollama_unload();
	test_ollama_unsupported_controls();
	test_audit_rotation_and_content();
	test_cli_errors();
	g_native->stop();
	g_oll->stop();
	g_native_thread->join();
	g_oll_thread->join();
	{
		std::string	cmd = "rm -rf '" + g_dir + "'";

		if (system(cmd.c_str()) != 0)
			fprintf(stderr, "warning: cleanup failed\n");
	}
	printf("test_apply_cmd: all tests passed\n");
	return (0);
}
