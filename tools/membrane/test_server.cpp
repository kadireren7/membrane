#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "server.h"
#include "test_helpers.h"

using json = nlohmann::json;

/*
 * Mega Phase A, PR A3: black-box HTTP tests against a REAL
 * membrane_server_run() instance (run in a background thread, stopped via
 * membrane_server_request_stop() -- the same control surface a real
 * SIGTERM handler uses). CI-safe by construction: an EMPTY, isolated
 * temp-directory registry (no real GGUF model anywhere in this file), so
 * every test here covers request validation, routing, and error mapping
 * -- never real generation (that needs a real model; covered by the
 * separate, non-ctest-registered dev-local end-to-end smoke, same
 * precedent as runtime-session-smoke/context-recommender-dryrun).
 *
 * A fixed test port is used rather than engineering port-0 (OS-assigned)
 * discovery back out of membrane_server_run() -- simpler, and CI runners
 * are isolated enough that a collision is not a realistic risk (same
 * judgment call other test suites in this project already make for
 * fixed resource names).
 */

# define TEST_PORT	18942

static std::thread	*g_server_thread = NULL;

static std::string	make_temp_registry_dir(void)
{
	char	tmpl[] = "/tmp/membrane-server-test-XXXXXX";
	char	*dir = mkdtemp(tmpl);

	TEST_ASSERT(dir != NULL, "mkdtemp succeeded");
	return (std::string(dir));
}

static void	start_test_server(const std::string &registry_dir)
{
	membrane_server_options_t	opts;

	opts.bind_address = "127.0.0.1";
	opts.port = TEST_PORT;
	opts.allow_non_loopback = false;
	opts.registry_path = registry_dir + "/models.json";
	g_server_thread = new std::thread([opts]()
		{ membrane_server_run(opts); });
	/* No race-free "ready" signal is exposed across the process boundary
	 * this test uses (httplib::Client, real sockets) -- poll /health
	 * with a bounded retry budget instead of a fixed sleep guess. */
	httplib::Client	probe("127.0.0.1", TEST_PORT);
	int				attempts = 0;

	probe.set_connection_timeout(0, 50000);
	while (attempts < 100)
	{
		auto	res = probe.Get("/health");

		if (res && res->status == 200)
			return ;
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		attempts++;
	}
	TEST_ASSERT(false, "server did not become ready within 5s");
}

static void	stop_test_server(void)
{
	membrane_server_request_stop();
	if (g_server_thread != NULL)
	{
		g_server_thread->join();
		delete g_server_thread;
		g_server_thread = NULL;
	}
}

static void	test_health(void)
{
	httplib::Client	cli("127.0.0.1", TEST_PORT);
	auto			res = cli.Get("/health");

	TEST_ASSERT(res != nullptr, "got an HTTP response");
	TEST_ASSERT(res->status == 200, "GET /health returns 200");
	json	body = json::parse(res->body);

	TEST_ASSERT(body["status"] == "ok", "health status is \"ok\"");
	TEST_ASSERT(body.contains("version"), "health reports a version");
}

static void	test_models_empty_registry(void)
{
	httplib::Client	cli("127.0.0.1", TEST_PORT);
	auto			res = cli.Get("/v1/models");

	TEST_ASSERT(res != nullptr && res->status == 200,
		"GET /v1/models returns 200 even for an empty registry");
	json	body = json::parse(res->body);

	TEST_ASSERT(body["object"] == "list", "response object is \"list\"");
	TEST_ASSERT(body["data"].is_array() && body["data"].empty(),
		"data is an empty array for an empty registry");
}

static void	test_status_no_model_loaded(void)
{
	httplib::Client	cli("127.0.0.1", TEST_PORT);
	auto			res = cli.Get("/v1/status");

	TEST_ASSERT(res != nullptr && res->status == 200,
		"GET /v1/status returns 200");
	json	body = json::parse(res->body);

	TEST_ASSERT(body["running"] == true, "running is true");
	TEST_ASSERT(body["loaded_model"].is_null(),
		"loaded_model is null before any generation request");
	TEST_ASSERT(body.contains("endpoint"), "endpoint field is present");
	TEST_ASSERT(body["context_policy"] == "automatic",
		"context_policy is \"automatic\"");
}

static void	test_chat_unknown_model(void)
{
	httplib::Client	cli("127.0.0.1", TEST_PORT);
	json			req = {{"model", "nonexistent"},
			{"messages", json::array({{{"role", "user"},
				{"content", "hi"}}})}};
	auto			res = cli.Post("/v1/chat/completions", req.dump(),
			"application/json");

	TEST_ASSERT(res != nullptr, "got an HTTP response");
	TEST_ASSERT(res->status == 404, "an unknown model returns 404");
	json	body = json::parse(res->body);

	TEST_ASSERT(body["error"]["code"] == "MODEL_NOT_FOUND",
		"error code is MODEL_NOT_FOUND");
	/* Section 27: no local filesystem leak -- the registry is empty, so
	 * there is no real path to leak here, but the error message itself
	 * must never contain an absolute path shape. */
	TEST_ASSERT(body["error"]["message"].get<std::string>().find("/tmp/")
		== std::string::npos, "error message contains no filesystem path");
}

static void	test_chat_invalid_json(void)
{
	httplib::Client	cli("127.0.0.1", TEST_PORT);
	auto			res = cli.Post("/v1/chat/completions", "not json at all",
			"application/json");

	TEST_ASSERT(res != nullptr, "got an HTTP response");
	TEST_ASSERT(res->status == 400, "invalid JSON returns 400");
	json	body = json::parse(res->body);

	TEST_ASSERT(body["error"]["code"] == "INVALID_REQUEST",
		"error code is INVALID_REQUEST");
}

static void	test_chat_missing_required_fields(void)
{
	httplib::Client	cli("127.0.0.1", TEST_PORT);
	json			req = {{"model", "smol"}};	/* no "messages" */
	auto			res = cli.Post("/v1/chat/completions", req.dump(),
			"application/json");

	TEST_ASSERT(res != nullptr && res->status == 400,
		"a request missing \"messages\" returns 400");
}

static void	test_chat_stream_true_rejected(void)
{
	httplib::Client	cli("127.0.0.1", TEST_PORT);
	json			req = {{"model", "smol"},
			{"messages", json::array({{{"role", "user"},
				{"content", "hi"}}})}, {"stream", true}};
	auto			res = cli.Post("/v1/chat/completions", req.dump(),
			"application/json");

	TEST_ASSERT(res != nullptr, "got an HTTP response");
	TEST_ASSERT(res->status == 400, "stream=true returns 400 (Section 29: "
		"never fake streaming)");
	json	body = json::parse(res->body);

	TEST_ASSERT(body["error"]["code"] == "STREAMING_NOT_SUPPORTED",
		"error code is STREAMING_NOT_SUPPORTED");
}

static void	test_chat_malformed_message_shape(void)
{
	httplib::Client	cli("127.0.0.1", TEST_PORT);
	json			req = {{"model", "smol"},
			{"messages", json::array({{{"role", "user"}}})}};
			/* missing "content" */
	auto			res = cli.Post("/v1/chat/completions", req.dump(),
			"application/json");

	TEST_ASSERT(res != nullptr && res->status == 400,
		"a message missing \"content\" returns 400");
}

int	main(void)
{
	std::string	dir = make_temp_registry_dir();

	start_test_server(dir);
	test_health();
	test_models_empty_registry();
	test_status_no_model_loaded();
	test_chat_unknown_model();
	test_chat_invalid_json();
	test_chat_missing_required_fields();
	test_chat_stream_true_rejected();
	test_chat_malformed_message_shape();
	stop_test_server();
	{
		std::string	cmd = "rm -rf '" + dir + "'";
		int			rc = system(cmd.c_str());

		if (rc != 0)
			fprintf(stderr, "warning: cleanup of %s may have failed "
				"(rc=%d)\n", dir.c_str(), rc);
	}
	printf("test_server: all tests passed\n");
	return (0);
}
