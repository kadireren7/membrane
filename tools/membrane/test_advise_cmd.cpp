#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "advise_cmd.h"
#include "observe_cmd.h"
#include "runtime_ollama.h"
#include "product_cli.h"
#include "service_state.h"
#include "test_helpers.h"

using json = nlohmann::json;

/*
 * Milestone I3: `membrane advise`'s own CLI-level tests. The pure
 * assembly logic (thresholds, findings, recommendations) is already
 * exhaustively covered, host-independent, by tools/membrane-run/
 * test_memory_intelligence.c -- this file only proves the CLI layer
 * wires that logic up correctly: routes to the right provider, never
 * touches anything beyond what `membrane observe` already touches
 * (test_observe_cmd.cpp/test_observe_ollama.cpp already prove THOSE
 * providers are read-only), produces a stable JSON shape, and is
 * deterministic given the same inputs.
 *
 * Same isolated-env / in-process-mock-server pattern as
 * test_observe_cmd.cpp (native) and test_observe_ollama.cpp (Ollama) --
 * never a real service, never a real Ollama daemon.
 */

/* ---------------------------------------------------------------- */
/* Isolated environment (native) -- same variables as                */
/* test_observe_cmd.cpp's own s_isolated_env                          */
/* ---------------------------------------------------------------- */

static std::string	make_temp_dir(void)
{
	char	tmpl[] = "/tmp/membrane-advise-test-XXXXXX";
	char	*dir = mkdtemp(tmpl);

	TEST_ASSERT(dir != NULL, "mkdtemp succeeded");
	return (std::string(dir));
}

static void	rmdir_recursive(const std::string &dir)
{
	std::string	cmd = "rm -rf '" + dir + "'";

	if (system(cmd.c_str()) != 0)
		fprintf(stderr, "warning: cleanup of %s may have failed\n",
			dir.c_str());
}

struct s_isolated_env
{
	std::string	dir;

	s_isolated_env()
	{
		dir = make_temp_dir();
		setenv("MEMBRANE_MODELS_PATH", (dir + "/models.json").c_str(), 1);
		setenv("MEMBRANE_MODELS_INSTALL_DIR", (dir + "/models").c_str(), 1);
		setenv("MEMBRANE_SERVER_CONFIG_PATH", (dir + "/server.json").c_str(),
			1);
		setenv("MEMBRANE_SYSTEMD_USER_DIR", (dir + "/systemd").c_str(), 1);
		setenv("MEMBRANE_LAUNCHD_USER_DIR", (dir + "/launchd").c_str(), 1);
		setenv(MEMBRANE_SERVICE_PROBE_OVERRIDE_ENV, "not_installed", 1);
	}
	~s_isolated_env()
	{
		unsetenv("MEMBRANE_MODELS_PATH");
		unsetenv("MEMBRANE_MODELS_INSTALL_DIR");
		unsetenv("MEMBRANE_SERVER_CONFIG_PATH");
		unsetenv("MEMBRANE_SYSTEMD_USER_DIR");
		unsetenv("MEMBRANE_LAUNCHD_USER_DIR");
		unsetenv(MEMBRANE_SERVICE_PROBE_OVERRIDE_ENV);
		rmdir_recursive(dir);
	}
};

static std::string	read_file(const std::string &p)
{
	FILE		*f = fopen(p.c_str(), "rb");
	std::string	out;
	char		buf[4096];
	size_t		n;

	if (f == NULL)
		return ("<absent>");
	while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
		out.append(buf, n);
	fclose(f);
	return (out);
}

static std::string	capture_dispatch(const std::vector<std::string> &args,
				bool want_json, int *rc)
{
	char	tmpl[] = "/tmp/membrane-advise-test-capture-XXXXXX";
	int		fd = mkstemp(tmpl);

	TEST_ASSERT(fd >= 0, "capture file created");
	fflush(stdout);
	int	saved = dup(STDOUT_FILENO);

	dup2(fd, STDOUT_FILENO);
	close(fd);
	*rc = membrane_advise_cmd_dispatch(args, want_json);
	fflush(stdout);
	dup2(saved, STDOUT_FILENO);
	close(saved);
	std::string	out = read_file(tmpl);

	unlink(tmpl);
	return (out);
}

/* ---------------------------------------------------------------- */
/* Mock Ollama server -- GET /api/version + GET /api/ps only, every  */
/* other route traps (500) so a route violation fails loudly          */
/* ---------------------------------------------------------------- */

static httplib::Server			*g_mock;
static std::thread				*g_mock_thread;
static int						g_mock_port;
static std::mutex				g_mock_mutex;
static std::vector<std::string>	g_mock_log;
static std::string				g_version_body = R"({"version": "0.34.3"})";
static int						g_version_status = 200;
static std::string				g_ps_body = R"({"models": []})";
static int						g_ps_status = 200;

static void	reset_mock(void)
{
	g_version_body = R"({"version": "0.34.3"})";
	g_version_status = 200;
	g_ps_body = R"({"models": []})";
	g_ps_status = 200;
	g_mock_log.clear();
}

static void	trap(const httplib::Request &, httplib::Response &res)
{
	res.status = 500;
	res.set_content("{\"error\":\"trap route hit\"}", "application/json");
}

static void	start_mock(void)
{
	reset_mock();
	g_mock = new httplib::Server();
	g_mock->set_logger([](const httplib::Request &req,
			const httplib::Response &)
	{
		std::lock_guard<std::mutex>	lock(g_mock_mutex);

		g_mock_log.push_back(req.method + " " + req.path);
	});
	g_mock->Get("/api/version", [](const httplib::Request &,
			httplib::Response &res)
	{
		std::lock_guard<std::mutex>	lock(g_mock_mutex);

		res.status = g_version_status;
		res.set_content(g_version_body, "application/json");
	});
	g_mock->Get("/api/ps", [](const httplib::Request &, httplib::Response &res)
	{
		std::lock_guard<std::mutex>	lock(g_mock_mutex);

		res.status = g_ps_status;
		res.set_content(g_ps_body, "application/json");
	});
	for (const char *p : {"/api/generate", "/api/chat", "/api/embed",
			"/api/embeddings", "/api/pull", "/api/push", "/api/create",
			"/api/copy"})
		g_mock->Post(p, trap);
	g_mock->Delete("/api/delete", trap);
	g_mock->Get("/api/tags", trap);
	g_mock->Post("/api/show", trap);
	g_mock_port = g_mock->bind_to_any_port("127.0.0.1");
	TEST_ASSERT(g_mock_port > 0, "mock bound an ephemeral port");
	g_mock_thread = new std::thread([]() { g_mock->listen_after_bind(); });

	int	attempts = 0;

	while (!g_mock->is_running() && attempts < 100)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		attempts++;
	}
	TEST_ASSERT(g_mock->is_running(), "mock became ready");
}

static void	stop_mock(void)
{
	g_mock->stop();
	g_mock_thread->join();
	delete g_mock_thread;
	delete g_mock;
	g_mock = NULL;
	g_mock_thread = NULL;
}

static std::string	mock_url(void)
{
	return ("http://127.0.0.1:" + std::to_string(g_mock_port));
}

static std::vector<std::string>	mock_log(void)
{
	std::lock_guard<std::mutex>	lock(g_mock_mutex);

	return (g_mock_log);
}

static int	closed_port(void)
{
	int					fd = socket(AF_INET, SOCK_STREAM, 0);
	struct sockaddr_in	addr;
	socklen_t			len = sizeof(addr);

	TEST_ASSERT(fd >= 0, "socket()");
	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	addr.sin_port = 0;
	TEST_ASSERT(bind(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0,
		"bind ephemeral port");
	TEST_ASSERT(getsockname(fd, (struct sockaddr *)&addr, &len) == 0,
		"getsockname");
	close(fd);
	return ((int)ntohs(addr.sin_port));
}

static void	use_mock_endpoint(void)
{
	setenv(MEMBRANE_OLLAMA_ENDPOINT_ENV, mock_url().c_str(), 1);
}

static void	use_closed_endpoint(void)
{
	std::string	url = "http://127.0.0.1:" + std::to_string(closed_port());

	setenv(MEMBRANE_OLLAMA_ENDPOINT_ENV, url.c_str(), 1);
}

/* ---------------------------------------------------------------- */
/* Native path                                                       */
/* ---------------------------------------------------------------- */

static void	test_native_json_shape(void)
{
	s_isolated_env	env;
	int				rc;
	std::string		out = capture_dispatch({}, true, &rc);
	json			j = json::parse(out);

	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS,
		"native advise (available runtime, no model) exits 0");
	TEST_ASSERT(j["mode"] == "advise", "mode is advise");
	TEST_ASSERT(j["runtime"] == MEMBRANE_RUNTIME_ID_NATIVE, "runtime id");
	TEST_ASSERT(j.contains("status"), "top-level status present");
	TEST_ASSERT(j.contains("observation_summary"), "observation_summary "
		"present");
	TEST_ASSERT(j["findings"].is_array(), "findings is an array");
	TEST_ASSERT(j["recommendations"].is_array(),
		"recommendations is an array");
	TEST_ASSERT(j["reasons"].is_array(), "reasons is an array");
	TEST_ASSERT(j["mutates_state"] == false,
		"mutates_state is always false");
	for (const auto &rec : j["recommendations"])
		TEST_ASSERT(rec["mutates_state"] == false,
			"every recommendation echoes mutates_state false");
}

static void	test_native_human_output(void)
{
	s_isolated_env	env;
	int				rc;
	std::string		out = capture_dispatch({}, false, &rc);

	TEST_ASSERT(out.find("Memory intelligence") != std::string::npos,
		"human header present");
	TEST_ASSERT(out.find("Runtime:") != std::string::npos, "runtime line");
	TEST_ASSERT(out.find("Findings") != std::string::npos, "findings "
		"section");
	TEST_ASSERT(out.find("Recommendations") != std::string::npos,
		"recommendations section");
}

/* On a real, live host, RAM available can genuinely shift by a few
 * hundred KiB between two calls a few milliseconds apart (this repo's
 * own dev host runs with only ~5.6 GiB total -- see the "dev machine
 * memory constraints" project note), so byte-identical evidence TEXT
 * is not a meaningful determinism check here -- unlike test_memory_
 * intelligence.c's own fixture-based determinism test, which holds
 * every input fixed and DOES assert byte-for-byte identical output.
 * This CLI-level check instead asserts the part that must stay
 * identical regardless of a few hundred KiB of live drift: which
 * finding/recommendation CODES fire, in which order, and the overall
 * status/reasons -- i.e. the classification is stable even though the
 * exact measured number underneath it is not. */
static std::vector<std::string>	extract_codes(const json &arr)
{
	std::vector<std::string>	out;

	for (const auto &e : arr)
		out.push_back(e["code"].get<std::string>());
	return (out);
}

static void	test_native_deterministic(void)
{
	s_isolated_env	env;
	int				rc1;
	int				rc2;
	std::string		out1 = capture_dispatch({}, true, &rc1);
	std::string		out2 = capture_dispatch({}, true, &rc2);

	TEST_ASSERT(rc1 == rc2, "same exit code across repeated calls");
	json	j1 = json::parse(out1);
	json	j2 = json::parse(out2);

	TEST_ASSERT(j1["status"] == j2["status"],
		"same overall status across repeated calls milliseconds apart");
	TEST_ASSERT(extract_codes(j1["findings"]) == extract_codes(j2["findings"]),
		"same finding codes, same order");
	TEST_ASSERT(extract_codes(j1["recommendations"])
		== extract_codes(j2["recommendations"]),
		"same recommendation codes, same order");
	TEST_ASSERT(j1["reasons"] == j2["reasons"], "same reasons[] union");
}

/* ---------------------------------------------------------------- */
/* Ollama path                                                       */
/* ---------------------------------------------------------------- */

static void	test_ollama_unavailable_no_socket_error(void)
{
	int			rc;
	std::string	out;

	use_closed_endpoint();
	out = capture_dispatch({"--runtime", "ollama"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_RUNTIME_ERROR,
		"unreachable Ollama -> RUNTIME_ERROR, not a CLI error");
	json	j = json::parse(out);

	TEST_ASSERT(j["status"] == "insufficient_data",
		"unreachable runtime is insufficient_data, never a fabricated ok");
	TEST_ASSERT(out.find("Connection refused") == std::string::npos
		&& out.find("ECONNREFUSED") == std::string::npos
		&& out.find("errno") == std::string::npos,
		"no raw socket error text is ever printed");

	out = capture_dispatch({"--runtime", "ollama"}, false, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_RUNTIME_ERROR, "same in human mode");
	unsetenv(MEMBRANE_OLLAMA_ENDPOINT_ENV);
}

static void	test_ollama_healthy_zero_models_read_only(void)
{
	int			rc;
	std::string	out;

	start_mock();
	use_mock_endpoint();
	out = capture_dispatch({"--runtime", "ollama"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "healthy zero-model -> 0");

	json	j = json::parse(out);

	TEST_ASSERT(j["runtime"] == "ollama", "runtime id");
	TEST_ASSERT(j["mutates_state"] == false, "mutates_state false");

	std::vector<std::string>	log = mock_log();

	for (const std::string &line : log)
		TEST_ASSERT(line == "GET /api/version" || line == "GET /api/ps",
			("advise never requests anything beyond I2's allowlist: "
				+ line).c_str());
	unsetenv(MEMBRANE_OLLAMA_ENDPOINT_ENV);
	stop_mock();
}

static void	test_ollama_capability_limited_finding(void)
{
	int			rc;
	std::string	out;

	start_mock();
	{
		std::lock_guard<std::mutex>	lock(g_mock_mutex);

		g_ps_body = R"({"models": [{
			"name": "qwen2.5:7b", "model": "qwen2.5:7b",
			"size": 4683087332, "digest": "845dbda0ea48",
			"details": {"family": "qwen2", "quantization_level": "Q4_K_M"},
			"size_vram": 4683087332, "context_length": 4096
		}]})";
	}
	use_mock_endpoint();
	out = capture_dispatch({"--runtime", "ollama"}, true, &rc);
	TEST_ASSERT(rc == MEMBRANE_EXIT_SUCCESS, "one resident model -> 0");

	json	j = json::parse(out);
	bool	saw_capability_finding = false;

	for (const auto &f : j["findings"])
		if (f["dimension"] == "runtime_capability")
			saw_capability_finding = true;
	TEST_ASSERT(saw_capability_finding,
		"Ollama's real, static capability matrix (kv precision/placement "
		"control both unsupported) surfaces a runtime_capability finding "
		"end to end");
	unsetenv(MEMBRANE_OLLAMA_ENDPOINT_ENV);
	stop_mock();
}

/* ---------------------------------------------------------------- */
/* CLI-error paths                                                   */
/* ---------------------------------------------------------------- */

static void	test_unknown_runtime(void)
{
	int			rc;
	std::string	out = capture_dispatch({"--runtime", "vllm"}, true, &rc);

	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR,
		"an unimplemented/unknown runtime is a CLI error");
	json	j = json::parse(out);

	TEST_ASSERT(j["ok"] == false, "error envelope");
}

static void	test_unknown_option(void)
{
	int			rc;
	std::string	out = capture_dispatch({"--bogus"}, true, &rc);

	TEST_ASSERT(rc == MEMBRANE_EXIT_CLI_ERROR, "unknown option is a CLI "
		"error");
	(void)out;
}

int	main(void)
{
	test_native_json_shape();
	test_native_human_output();
	test_native_deterministic();
	test_ollama_unavailable_no_socket_error();
	test_ollama_healthy_zero_models_read_only();
	test_ollama_capability_limited_finding();
	test_unknown_runtime();
	test_unknown_option();
	printf("test_advise_cmd: all tests passed\n");
	return (0);
}
