#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "adaptive_cmd.h"
#include "product_cli.h"
#include "server_config.h"
#include "test_helpers.h"

using json = nlohmann::json;

/*
 * Milestone K: `membrane adaptive` CLI tests -- isolated config file, an
 * in-process mock of GET /membrane/v1/adaptive, no real service.
 */

static std::string	g_dir;
static httplib::Server	*g_srv;
static std::thread		*g_thr;
static int				g_port;
static json				g_live;

static std::string	read_file(const std::string &p)
{
	std::ifstream		f(p, std::ios::binary);
	std::ostringstream	ss;

	ss << f.rdbuf();
	return (ss.str());
}

static std::string	cfg_path(void) { return (g_dir + "/server.json"); }

static void	write_cfg(const std::string &extra = "", int port = -1)
{
	std::ofstream	f(cfg_path(), std::ios::binary | std::ios::trunc);

	f << "{\"schema_version\":1,\"listen_address\":\"127.0.0.1\",\"port\":"
		<< (port < 0 ? g_port : port) << extra << "}";
}

static std::string	capture(const std::vector<std::string> &args, bool want_json,
						const membrane_adaptive_cmd_io_t &io, int *rc)
{
	std::string	tmpl = g_dir + "/cap-XXXXXX";
	int			fd = mkstemp(&tmpl[0]);

	TEST_ASSERT(fd >= 0, "capture file");
	fflush(stdout);
	int	saved = dup(STDOUT_FILENO);

	dup2(fd, STDOUT_FILENO);
	close(fd);
	*rc = membrane_adaptive_cmd_dispatch_ex(args, want_json, io);
	fflush(stdout);
	dup2(saved, STDOUT_FILENO);
	close(saved);
	std::string	out = read_file(tmpl);

	unlink(tmpl.c_str());
	return (out);
}

static membrane_adaptive_cmd_io_t	tty(bool yes_answer = true)
{
	membrane_adaptive_cmd_io_t	io;

	io.interactive_known = true;
	io.interactive = true;
	io.confirm = [yes_answer](const std::string &) { return (yes_answer); };
	return (io);
}

static membrane_adaptive_cmd_io_t	notty(void)
{
	membrane_adaptive_cmd_io_t	io;

	io.interactive_known = true;
	io.interactive = false;
	return (io);
}

static membrane_server_adaptive_t	saved(void)
{
	membrane_server_config_t		c;
	membrane_server_config_error_t	e;

	TEST_ASSERT(membrane_server_config_load(cfg_path(), &c, &e), "cfg loads");
	return (c.adaptive);
}

static void	test_default_is_off(void)
{
	int			rc;
	std::string	out;

	write_cfg();
	out = capture({"status"}, false, notty(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS && out.find("Adaptive runtime: off")
		!= std::string::npos, "off on a fresh config");
	TEST_ASSERT(out.find("Next: membrane adaptive enable") != std::string::npos,
		"next-action guidance");
	TEST_ASSERT(!saved().present && !saved().enabled, "nothing persisted");
	json	j = json::parse(capture({"status"}, true, notty(), &rc));

	TEST_ASSERT(j["enabled"] == false && j["effective_mode"] == "off"
		&& j["live"].is_null(), "JSON: off, no live section");
}

static void	test_enable_recommend_and_disable(void)
{
	int			rc;
	std::string	out;

	write_cfg();
	out = capture({"enable"}, false, notty(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "recommend needs no confirmation");
	TEST_ASSERT(out.find("Adaptive runtime: recommend") != std::string::npos
		&& out.find("never changes anything") != std::string::npos,
		"says it is recommend-only");
	membrane_server_adaptive_t	a = saved();

	TEST_ASSERT(a.enabled && a.mode == "recommend"
		&& a.runtime == "membrane-native" && a.cooldown_s == 600
		&& a.interval_s == 30 && a.allow.size() == 2, "conservative defaults "
			"persisted");
	out = capture({"disable"}, false, notty(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS && !saved().enabled
		&& saved().mode == "recommend" && out.find("off") != std::string::npos,
		"disable turns it off and keeps preferences");
	/* the rest of the config is untouched */
	TEST_ASSERT(read_file(cfg_path()).find("\"default_model\"")
		!= std::string::npos, "other config fields preserved");
}

static void	test_apply_mode_needs_explicit_consent(void)
{
	int			rc;
	std::string	out;
	std::string	before;

	write_cfg();
	before = read_file(cfg_path());
	out = capture({"enable", "--mode", "apply"}, false, notty(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "non-interactive apply without "
		"--yes fails closed");
	TEST_ASSERT(read_file(cfg_path()) == before, "nothing saved");
	json	j = json::parse(capture({"enable", "--mode", "apply"}, true, tty(),
			&rc));

	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR && j["error"]["code"]
		== "CONFIRMATION_REQUIRED", "--json never prompts: needs --yes");
	/* interactive: declined */
	out = capture({"enable", "--mode", "apply"}, false, tty(false), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS && read_file(cfg_path()) == before
		&& out.find("WARNING") != std::string::npos
		&& out.find("change runtime state") != std::string::npos
		&& out.find("Declined") != std::string::npos,
		"the warning is explicit; declining changes nothing");
	/* interactive: accepted */
	out = capture({"enable", "--mode", "apply"}, false, tty(true), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS && saved().mode == "apply"
		&& saved().enabled, "accepted");
	TEST_ASSERT(out.find("may now change runtime state") != std::string::npos,
		"confirmation text states the consequence");
	/* --yes */
	write_cfg();
	out = capture({"enable", "--mode", "apply", "--yes", "--allow",
		"context,unload"}, false, notty(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS && saved().mode == "apply"
		&& saved().allow.size() == 2 && saved().allow[1] == "unload",
		"--yes works non-interactively; --allow honored");
}

static void	test_enable_validation(void)
{
	int			rc;
	std::string	before;

	write_cfg();
	before = read_file(cfg_path());
	capture({"enable", "--mode", "turbo"}, false, notty(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "bad mode");
	capture({"enable", "--interval", "1"}, false, notty(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "interval floor");
	capture({"enable", "--allow", "context,warpdrive"}, false, notty(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "bad class");
	capture({"enable", "--runtime", "vllm"}, false, notty(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "unsupported runtime");
	capture({"enable", "--bogus"}, false, notty(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "unknown option");
	capture({"frobnicate"}, false, notty(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "unknown subcommand");
	capture({}, false, notty(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "no subcommand");
	TEST_ASSERT(read_file(cfg_path()) == before, "no failed attempt wrote "
		"anything");
	capture({"enable", "--runtime", "ollama", "--mode", "observe"}, false,
		notty(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS && saved().runtime == "ollama",
		"ollama scope accepted (capability matrix limits what it can do)");
}

static void	test_status_with_live_service(void)
{
	int			rc;
	std::string	out;
	int64_t		now = (int64_t)std::chrono::duration_cast<
			std::chrono::milliseconds>(std::chrono::system_clock::now()
				.time_since_epoch()).count();

	write_cfg();
	capture({"enable", "--mode", "recommend"}, false, notty(), &rc);
	g_live = {{"pressure", "warning"}, {"decision", "recommend"},
		{"reason", nullptr}, {"step", "SET_CONTEXT 8192 -> 4096"},
		{"cooldown_remaining_s", 0}, {"suppressed", false},
		{"pending_reload", false}, {"last_observation_unix_ms", now - 5000},
		{"last_action", nullptr}, {"reason_chain", json::array(
			{{{"stage", "observed"}, {"text", "VRAM 4% free"}}})}};
	out = capture({"status"}, false, notty(), &rc);
	TEST_ASSERT(out.find("Adaptive runtime: recommend") != std::string::npos
		&& out.find("Runtime:           membrane-native") != std::string::npos
		&& out.find("Last observation:  5s ago") != std::string::npos
		&& out.find("Pressure:          warning") != std::string::npos
		&& out.find("Last action:       none") != std::string::npos
		&& out.find("Cooldown:          none") != std::string::npos
		&& out.find("Suppressed:        no") != std::string::npos,
		"the status block from the spec");
	TEST_ASSERT(out.find("Suggested step") != std::string::npos
		&& out.find("observed:") != std::string::npos, "step + why");
	TEST_ASSERT(out.find("{") == std::string::npos, "no JSON walls by default");
	g_live["suppressed"] = true;
	g_live["suppress_reason"] = "ADAPTATION_SUPPRESSED_THRASHING";
	g_live["cooldown_remaining_s"] = 90;
	g_live["last_action"] = {{"step", "SET_CONTEXT 8192 -> 4096"},
		{"status", "applied"}};
	out = capture({"status"}, false, notty(), &rc);
	TEST_ASSERT(out.find("Suppressed:        yes (ADAPTATION_SUPPRESSED_"
			"THRASHING)") != std::string::npos
		&& out.find("Last action:       SET_CONTEXT 8192 -> 4096 [applied]")
			!= std::string::npos, "suppression + last action shown");
	json	j = json::parse(capture({"status"}, true, notty(), &rc));

	TEST_ASSERT(j["service_reachable"] == true && j["live"]["pressure"]
		== "warning", "--json carries the live state");
	/* service not running */
	write_cfg(",\"adaptive\":{\"enabled\":true,\"mode\":\"recommend\"}", 1);
	out = capture({"status"}, false, notty(), &rc);
	TEST_ASSERT(out.find("not running") != std::string::npos
		&& out.find("Next: membrane serve") != std::string::npos,
		"stopped service => saved policy + next step");
}

static void	test_invalid_saved_policy_is_reported_and_off(void)
{
	int			rc;
	std::string	out;

	write_cfg(",\"adaptive\":{\"enabled\":true,\"mode\":\"turbo\"}");
	out = capture({"status"}, false, notty(), &rc);
	TEST_ASSERT(out.find("Adaptive runtime: off") != std::string::npos
		&& out.find("invalid") != std::string::npos, "invalid => off + says why");
	write_cfg(",\"adaptive\":{\"enabled\":\"yes\"}");
	out = capture({"status"}, true, notty(), &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_RUNTIME_ERROR, "bad JSON type fails the "
		"config load (fail closed)");
}

int	main(void)
{
	char	tmpl[] = "/tmp/membrane-adaptive-test-XXXXXX";

	TEST_ASSERT(mkdtemp(tmpl) != NULL, "temp dir");
	g_dir = tmpl;
	setenv("MEMBRANE_SERVER_CONFIG_PATH", cfg_path().c_str(), 1);
	g_srv = new httplib::Server();
	g_srv->Get("/membrane/v1/adaptive", [](const httplib::Request &,
			httplib::Response &res)
	{
		res.set_content(g_live.dump(), "application/json");
	});
	g_port = g_srv->bind_to_any_port("127.0.0.1");
	TEST_ASSERT(g_port > 0, "bound");
	g_thr = new std::thread([]() { g_srv->listen_after_bind(); });
	for (int i = 0; i < 100 && !g_srv->is_running(); ++i)
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	test_default_is_off();
	test_enable_recommend_and_disable();
	test_apply_mode_needs_explicit_consent();
	test_enable_validation();
	test_status_with_live_service();
	test_invalid_saved_policy_is_reported_and_off();
	g_srv->stop();
	g_thr->join();
	{
		std::string	cmd = "rm -rf '" + g_dir + "'";

		if (system(cmd.c_str()) != 0)
			fprintf(stderr, "warning: cleanup failed\n");
	}
	printf("test_adaptive_cmd: all tests passed\n");
	return (0);
}
